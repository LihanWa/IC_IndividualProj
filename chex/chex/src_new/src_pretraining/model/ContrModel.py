from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import einops
from model.components.pooling import GlobalAvgPool
from omegaconf import MISSING
from model.img_encoder.chexzero_img_encoder import ImageEncoder, ImageEncoderConfig
from model.txt_decoder.ptuned_decoder import PTunedDecoderConfig, PTunedDecoderModel
from model.txt_encoder.chexzero_txt_encoder import ChexzeroTextEncoder, ChexzeroTextEncoderConfig

import torch.nn.functional as F
import torch
import numpy as np
from deepdiff import DeepDiff

from metrics.training_metrics import TrainingMetrics
from metrics.contr_training_metrics import ContrastTrainingMetrics
from model.components.bbox_prediction import clip_bboxes
from model.components.classification_losses import classify_features
from model.detector.token_decoder_detector import PostDecoder, TokenDetectorOutput, TokenDetector, TokenDetectorConfig
from model.eval.anat_explainer import AnatomyExplainerEvaluator
from model.eval.box_explainer import BoxExplainerEvaluator
from model.eval.pathology_detection import PathologyDetectionEvaluator
from model.eval.report_generation import ReportEvaluator
from model.eval.sentence_grounding import SentenceGroundingEvaluator
from model.img_encoder import ImageEncoderOutput
from model.supervisors.anatomy_token_supervisor import AnatomyTokenConfig, AnatomyTokenSupervisor
from model.supervisors.patho_token_supervisor import PathologyTokenConfig, PathologyTokenSupervisor
from model.supervisors.sentence_token_supervisor import SentenceTokenConfig, SentenceTokenSupervisor
from model.txt_encoder import TextEncoderOutput
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from util.plot_grounding import plot_grounding

from util.model_utils import BaseModel, BaseModelOutput, MainModelConfig, load_model_by_name
from util.plot_utils import wandb_plot_text
from util.prompt_utils import fill_prompt_templates, flatten_prompts, flatten_prompts_2, localized_prompt_templates
from util.train_utils import EvalConfig, Evaluator
import ipdb
import tracemalloc
import deepspeed
import torch.distributed as dist

log = logging.getLogger(__name__)

@dataclass
class ContrastTrainOutput(BaseModelOutput):
    sample_id: List[str] = field(default_factory=list)
    x: Tensor = MISSING

    encoded_img: ImageEncoderOutput = MISSING


    @property
    def device(self):
        return self.x.device
    



@dataclass
class ContrastModelConfig(MainModelConfig):
    # --- Model ---
    name: str = MISSING
    img_encoder: ImageEncoderConfig = MISSING
    txt_encoder: ChexzeroTextEncoderConfig = MISSING

    load_components_from: Dict[str, Optional[str]] = field(default_factory=dict)
    freeze_loaded_components: List[str] = field(default_factory=list)

    # --- Post-decoder layers ---
    n_post_decoder_layers: int = 0
    post_decoder_patches: bool = False
    post_decoder_cls: bool = False
    post_decoder_droppath: bool = False
    post_decoder_gate: bool = False
    post_decoder_multibox_drop: float = 0.0

    # --- Prompts ---
    anatomy_prompts: Dict[str, List[str]] = field(default_factory=dict)
    pathology_pos_prompts: Dict[str, List[str]] = field(default_factory=dict)
    pathology_neg_prompts: Dict[str, List[str]] = field(default_factory=dict)
    no_finding_prompts: List[str] = field(default_factory=list)
    pathology_neg_prompt_mode: str = 'neg'  # neg, pos_centroid, no_finding
    randomize_prompts: bool = False


    cache_encoded_sentences: bool = True    
    drop_sentence_prob: float = 0.0

    temperature: float = 0.1
def memory_monitor(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB")
        return result
    return wrapper

class ContrastModel(BaseModel):
    CONFIG_CLS = ContrastModelConfig

    def __init__(self, config: ContrastModelConfig, from_checkpoint=False) -> None:
        self.config: ContrastModelConfig
        super().__init__(config)

        # --- Components ---
        self.img_encoder = ImageEncoder(self.config.img_encoder, main_config=self.config)
        self.txt_encoder = ChexzeroTextEncoder(self.config.txt_encoder, main_config=self.config)


    def gather_features(self, feature, dp_group):
        """
        将 feature (shape: [B, D]) 在数据并行组内 all_gather,
        返回拼接后的 tensor (shape: [world_size * B, D])
        """
        world_size = dist.get_world_size(group=dp_group)
        gathered = [torch.zeros_like(feature) for _ in range(world_size)]
        dist.all_gather(gathered, feature, group=dp_group)
        return torch.cat(gathered, dim=0)




            

    
    # @memory_monitor
    def forward(self, 
                x: FloatTensor, 
                sample_id: List[str],
                epoch=None,
                data_config=None,
                sentences: List[List[str]] = None,
                **kwargs): #samples
        
        # 在主要入口点添加内存追踪
        # tracemalloc.start()

        # ---------- Encode the image features (i.e. patches) and all required prompts ----------
        # 只在第一个epoch的第一个批次打印信息
        should_log = (epoch == 0 or epoch is None) and not hasattr(self, '_logged_contrast_info')
        
        encoded_image: ImageEncoderOutput = self.img_encoder(x, epoch=epoch,sample_id=sample_id)
        global_features = F.normalize(encoded_image.global_features, dim=-1)
        text_features = F.normalize(self.encode_sentences(sentences, device=x.device, epoch=epoch).sentence_features, dim=-1)
        
        # 收集所有进程的特征
        if dist.is_initialized():
            dp_group = getattr(self, 'dp_group', dist.group.WORLD)
            global_features_all = self.gather_features(global_features, dp_group)
            text_features_all = self.gather_features(text_features, dp_group)
            
            # 获取当前进程的rank和world_size
            rank = dist.get_rank(dp_group)
            world_size = dist.get_world_size(dp_group)
            local_bs = global_features.size(0)
            
            # 验证特征收集是否正确
            expected_total_samples = world_size * local_bs
            actual_total_samples = global_features_all.size(0)
            assert actual_total_samples == expected_total_samples, f"特征收集错误: 预期{expected_total_samples}个样本，实际有{actual_total_samples}个样本"
            
            # 记录负样本数量 - 只在需要时打印
            if should_log and rank == 0:
                negative_samples_count = actual_total_samples - 1
                log.info(f"分布式训练: 每个样本的负样本数量为 {negative_samples_count} (world_size={world_size}, local_bs={local_bs})")
                # 标记已经打印过
                self._logged_contrast_info = True
            
            # 确保所有进程使用相同的batch_size计算标签
            all_batch_sizes = [torch.tensor([local_bs], device=x.device) for _ in range(world_size)]
            dist.all_gather(all_batch_sizes, torch.tensor([local_bs], device=x.device), group=dp_group)
            
            # 计算每个进程的起始索引
            start_indices = [0]
            for i in range(world_size-1):
                start_indices.append(start_indices[-1] + all_batch_sizes[i].item())
            
            # 当前进程的起始和结束索引
            start_idx = start_indices[rank]
            end_idx = start_idx + local_bs
        else:
            # 单节点情况：直接使用当前特征
            global_features_all = global_features
            text_features_all = text_features
            start_idx, end_idx = 0, global_features_all.shape[0]
            
            # 记录负样本数量 - 只在需要时打印
            if should_log:
                batch_size = global_features_all.size(0)
                negative_samples_count = batch_size - 1
                log.info(f"单节点训练: 每个样本的负样本数量为 {negative_samples_count} (batch_size={batch_size})")
                # 标记已经打印过
                self._logged_contrast_info = True

        logits = global_features_all @ text_features_all.T/self.config.temperature
        
        # 生成标签
        labels = torch.arange(start_idx, end_idx, device=logits.device)
        
        if dist.is_initialized():
            logits_local = logits[start_idx:end_idx, :]
        else:
            logits_local = logits
        
        # 在关键操作前添加同步点
        if dist.is_initialized():
            dist.barrier(group=dp_group)
        
        # 计算对比损失 - 不使用torch.no_grad()
        loss_img2txt = nn.functional.cross_entropy(logits_local, labels)
        loss_txt2img = nn.functional.cross_entropy(logits[:, start_idx:end_idx].T, labels)
        
        # 在损失计算后添加同步点
        if dist.is_initialized():
            dist.barrier(group=dp_group)
        
        # ---------- Supervisors ----------
        # 对比损失
        contrastive_loss = (5 * loss_img2txt + loss_txt2img) / 6
        
        # 计算中心损失 - 同样不使用torch.no_grad()
        center_loss = torch.mean(torch.sum((global_features - text_features) ** 2, dim=1)) / 4
        
        # 组合损失
        loss = contrastive_loss + 0.05 * center_loss
        
        output = ContrastTrainOutput(
            sample_id=sample_id,
            x=x, 
            encoded_img=encoded_image,
        )
        output.loss = loss

        return output


    def encode_sentences(self, sentences: List[List[str]], device, epoch=None) -> TextEncoderOutput:
        flattened_sentences, sentence_mask = flatten_prompts(sentences, device=device)
        flat_sentence_features = self.txt_encoder.encode_sentences(flattened_sentences, cache=self.config.cache_encoded_sentences, epoch=epoch)
        N, S = sentence_mask.shape
        d = flat_sentence_features.shape[-1]
        # (N x S x d)
        sentence_features = flat_sentence_features.new_zeros((N, S, d))
        sentence_features[sentence_mask] = flat_sentence_features
        
        pooled_features = torch.zeros(N, d, device=device)
        # 对每个样本进行处理
        for i in range(N):
            # 获取当前样本的有效特征
            valid_features = sentence_features[i, sentence_mask[i], :]
            # 如果有有效特征，则计算平均值
            if valid_features.size(0) > 0:
                pooled_features[i] = valid_features.mean(dim=0)
        
        return TextEncoderOutput(
            sentence_features=pooled_features, # (N x d) 经过mean pooling后的特征
            sentence_mask=sentence_mask, # (N x S)
            sentences=sentences,
            flattened_sentences=flattened_sentences
        )


    def build_evaluator(self, task: EvalConfig, **kwargs) -> Evaluator:
        if task.task is None:
            return TrainEvaluator(task, self, **kwargs)
        else:
            raise ValueError(f'Unknown task {task.task}')




class TrainEvaluator(Evaluator):
    def __init__(self, task: EvalConfig, model: ContrastModel, **kwargs):
        super().__init__(task, EvalConfig, **kwargs)
        self.model = model
        self.task=task
        self._register_metric(ContrastTrainingMetrics())

    def eval_step(self, **kwargs) -> 'ContrastTrainOutput':
        output: ContrastTrainOutput = self.model(**kwargs, 
                                                validate=True,
                                                task=self.task)
        self._update_metric(model_output=output)
        return output
 
    
    def plot_metrics(self, output: ContrastTrainOutput):
        """只返回数值指标，不生成图像"""
        metrics = {}
        
        # 添加您需要的数值指标
        # 例如：准确率、损失值等
        if hasattr(output, 'loss') and output.loss is not None:
            metrics['loss'] = output.loss
            
        if hasattr(output, 'accuracy') and output.accuracy is not None:
            metrics['accuracy'] = output.accuracy
            
        # 如果有其他指标，可以继续添加
        # 例如：precision, recall, f1 等
        
        return metrics
    
    def plot(self, output: ContrastTrainOutput, max_samples: int, target_dir: str, plot_local):
        """保留原方法以兼容现有代码，但只返回数值指标"""
        return self.plot_metrics(output)
