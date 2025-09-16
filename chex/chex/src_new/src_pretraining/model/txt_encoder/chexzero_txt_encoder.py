from dataclasses import dataclass
import os
from typing import List
from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from model.components.mlp import MLP

from util.model_utils import BaseModel, BaseModelConfig, MainModelConfig
import chexzero.clip
import logging
import threading

log = logging.getLogger(__name__)

@dataclass
class ChexzeroTextEncoderConfig(BaseModelConfig):
    # model_path: str = os.path.expanduser("~/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt")
    model_path: str = os.path.join(os.environ.get('MODELS_DIR'),"best_64_5e-05_original_22000_0.864.pt")

    frozen_language_model: bool = False
    frozen_token_embedding: bool = True
    frozen_positional_embedding: bool = True
    frozen_transformer: bool = True
    freeze_layers: int = 8
    frozen_ln_final: bool = True
    frozen_text_projection: bool = False
    frozen_projection: bool = False
    # Additional projection layers (after the language model projection)
    # 0 = no projection, 1 = linear, 2 = one hidden layer
    n_projection_layers: int = 0
    # whether to use batch norm in the addtional projection layers
    projection_bn: bool = False
    normalize_projected: bool = False


class ChexzeroTextEncoder(BaseModel):
    CONFIG_CLS = ChexzeroTextEncoderConfig
    MODIFYABLE_CONFIGS = ('frozen_backbone', )

    def __init__(self, config: ChexzeroTextEncoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: ChexzeroTextEncoderConfig

        self.d = main_config.d_model

        model, _ = chexzero.clip.load("ViT-B/32", device='cpu', jit=False) 
        
        # 在分布式环境中同步加载模型
        if torch.distributed.is_initialized():
            # 确保所有进程都等待rank 0加载完成
            if torch.distributed.get_rank() == 0:
                # 主进程验证模型路径存在
                model_path = os.path.join(os.environ.get('CHEX_DIR'),'chex/chex/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt')
                assert os.path.exists(model_path), f"模型路径不存在: {model_path}"
            # 同步所有进程
            torch.distributed.barrier()
        
        # 加载模型权重
        try:
            model.load_state_dict(torch.load(os.path.join(os.environ.get('CHEX_DIR'),'chex/chex/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt'), map_location='cpu'))
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                log.info(f"成功加载txt模型权重")
        except Exception as e:
            log.error(f"加载模型权重失败: {e}")
            raise RuntimeError(f"模型权重加载失败: {e}")
        
        self.d = main_config.d_model

        self.transformer = model.transformer
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        d_backbone = self.text_projection.shape[1]
        
        self.projection = MLP(
            self.config.n_projection_layers, 
            d_in=d_backbone, 
            d_out=self.d, 
            use_bn=self.config.projection_bn,
            act=main_config.act,
            dropout=main_config.dropout)
            
        # 冻结设置
        if self.config.frozen_language_model:
            for param in self.parameters():
                param.requires_grad = False
        else:
            # 冻结token embedding（一般建议冻结）
            if self.config.frozen_token_embedding:
                for param in self.token_embedding.parameters():
                    param.requires_grad = False

            # 位置编码一般建议冻结
            if self.config.frozen_positional_embedding:
                self.positional_embedding.requires_grad = False
            else:
                self.positional_embedding.requires_grad = True

            # Transformer层选择性冻结
            if self.config.frozen_transformer:
                freeze_layers = self.config.freeze_layers  # 比如冻结前8层
                for idx, block in enumerate(self.transformer.resblocks):
                    requires_grad = idx >= freeze_layers  # 后面几层解冻
                    for param in block.parameters():
                        param.requires_grad = requires_grad

            # 最终LayerNorm（一般冻结）
            if self.config.frozen_ln_final:
                for param in self.ln_final.parameters():
                    param.requires_grad = False

            # text_projection（看具体情况，一般解冻）
            if self.config.frozen_text_projection:
                self.text_projection.requires_grad = False

            # projection模块（一般需要微调，不冻结）
            if self.config.frozen_projection:
                for param in self.projection.parameters():
                    param.requires_grad = False

        # 创建共享缓存
        self.cached_sentence_embeddings = {}
        self.cache_lock = threading.Lock()  # 用于多线程环境
        
        # 如果是分布式环境，确保只有rank 0进程计算和缓存嵌入
        self.is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

        # 只在主进程或非分布式环境中打印模型状态
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.print_model_freeze_status()

    def print_model_freeze_status(self):
        """
        打印模型各部分的冻结状态
        """
        print("="*50)
        print("文本编码器冻结状态:")
        print("-"*50)
        
        # 检查token_embedding
        token_frozen = all(not p.requires_grad for p in self.token_embedding.parameters())
        print(f"token_embedding: {'已冻结' if token_frozen else '未冻结'}")
        
        # 检查positional_embedding
        pos_frozen = not self.positional_embedding.requires_grad
        print(f"positional_embedding: {'已冻结' if pos_frozen else '未冻结'}")
        
        # 检查transformer各层
        print(f"transformer各层状态:")
        for i, block in enumerate(self.transformer.resblocks):
            block_frozen = all(not p.requires_grad for p in block.parameters())
            print(f"  - 第{i+1}层: {'已冻结' if block_frozen else '未冻结'}")
        
        # 检查ln_final
        ln_frozen = all(not p.requires_grad for p in self.ln_final.parameters())
        print(f"ln_final: {'已冻结' if ln_frozen else '未冻结'}")
        
        # 检查text_projection
        text_proj_frozen = not self.text_projection.requires_grad
        print(f"text_projection: {'已冻结' if text_proj_frozen else '未冻结'}")
        
        # 检查projection
        proj_frozen = all(not p.requires_grad for p in self.projection.parameters())
        print(f"projection: {'已冻结' if proj_frozen else '未冻结'}")
        # 添加调试信息
        print("Initial projection parameters status:")
        for name, param in self.projection.named_parameters():
            print(f"{name}: requires_grad = {param.requires_grad}")

        # 统计参数数量
        total_params = 0
        trainable_params = 0
        for param in self.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print("-"*50)
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params:.2%})")
        print("="*50)
    @property
    def dtype(self):
        return self.token_embedding.weight.dtype

    @property
    def device(self):
        return self.token_embedding.weight.device

    def forward(self, 
        input_ids: torch.Tensor,
        project: bool = True,
        **kwargs) -> Tensor:

        # 确保输入在正确的设备上
        input_ids = input_ids.to(device=self.device)

        # Encode image using backbone
        with torch.set_grad_enabled(not self.config.frozen_language_model):
            x = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x).type(self.dtype)

            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # (N_sentences x d)
            features = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection

        if project:
            features = self.projection(features)
        return features

    
    def encode_sentences(self, sentences: List[str], cache=False, **kwargs) -> Tensor:
        """
        编码一组句子，支持缓存和分布式环境
        
        Args:
            sentences: 要编码的句子列表
            cache: 是否使用缓存
            **kwargs: 传递给forward方法的额外参数
            
        Returns:
            编码后的特征张量
        """
        # 处理空输入
        if not sentences:
            return torch.empty((0, self.projection.out_features), 
                              dtype=self.dtype, device=self.device)
        
        # 初始化结果列表和跟踪变量
        final_features = [None] * len(sentences)
        indices_to_compute = []
        sentences_to_compute = []
        
        # 1. 检查缓存并确定需要计算的句子
        if cache and self.config.frozen_language_model:
            for i, s in enumerate(sentences):
                if s in self.cached_sentence_embeddings:
                    # 将缓存的张量移动到正确的设备
                    final_features[i] = self.cached_sentence_embeddings[s].to(self.device)
                else:
                    indices_to_compute.append(i)
                    sentences_to_compute.append(s)
        else:
            indices_to_compute = list(range(len(sentences)))
            sentences_to_compute = sentences.copy()
        
        # 2. 计算新特征（如果需要）
        if sentences_to_compute:
            input_ids = chexzero.clip.tokenize(sentences_to_compute, context_length=77).to(self.device)
            new_features = self(input_ids, project=False, **kwargs)
            
            # 3. 更新缓存并处理分布式环境
            if cache and self.config.frozen_language_model:
                # 准备要缓存的数据
                local_cache_updates = {}
                for i, (idx, sentence) in enumerate(zip(indices_to_compute, sentences_to_compute)):
                    # 存储在CPU上以节省GPU内存
                    feature_to_cache = new_features[i].detach().float().cpu()
                    local_cache_updates[sentence] = feature_to_cache
                    
                    # 更新本地缓存
                    self.cached_sentence_embeddings[sentence] = feature_to_cache
                    
                    # 填充结果列表
                    final_features[idx] = new_features[i]
                
                # 分布式缓存同步
                if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                    # 收集所有进程的缓存更新
                    gathered_updates = [None] * torch.distributed.get_world_size()
                    torch.distributed.all_gather_object(gathered_updates, local_cache_updates)
                    
                    # 更新本地缓存
                    for proc_updates in gathered_updates:
                        if proc_updates:  # 确保不是None
                            self.cached_sentence_embeddings.update(proc_updates)
            else:
                # 如果不使用缓存，直接填充结果
                for i, idx in enumerate(indices_to_compute):
                    final_features[idx] = new_features[i]
        
        # 4. 组合特征并进行最终处理
        features = torch.stack([f for f in final_features if f is not None], dim=0)
        features = features.to(dtype=self.dtype)
        projected_features = self.projection(features)
        
        return projected_features
        