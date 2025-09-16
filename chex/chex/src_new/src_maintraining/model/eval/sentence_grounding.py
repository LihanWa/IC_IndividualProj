from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from omegaconf import MISSING
import torch
from tqdm import tqdm
from metrics.detection_metrics import SentenceDetectionMetrics
from model.components.bbox_prediction import clip_bboxes
from model.detector.token_decoder_detector import TokenDetectorOutput
from model.txt_encoder import TextEncoderOutput
import torch.nn.functional as F

from util.model_utils import BaseModelOutput
from util.train_utils import EvalConfig, Evaluator
from torchvision.ops import nms
from transformers.models.detr.modeling_detr import center_to_corners_format

import logging
import ipdb
log = logging.getLogger(__name__)

@dataclass
class SentenceGroundingOutput(BaseModelOutput):
    # (N x H x W)
    image: torch.Tensor = MISSING

    # List (N) of tensors (M_i x 4) in the (x_c, y_c, w, h) format
    pred_boxes: List[torch.Tensor] = MISSING
    # List (N) of tensors (M_i x 5) in the (x_c, y_c, w, h, class_id) format
    target_cls_boxes: Optional[List[torch.Tensor]] = None
    # (N x M_i)
    target_sentences: Optional[List[List[List[str]]]] = None
 
@dataclass
class SentenceGroundingEvalConfig(EvalConfig):
    obj_threshold: Optional[float] = 0.3
    box_scale_factor: Optional[float] = None
    clip_boxes: bool = True
    nms_threshold: float = 0.25

    skip_roi_pool_inference: bool = True
    

class SentenceGroundingEvaluator(Evaluator):
    def __init__(self, config: SentenceGroundingEvalConfig, model: 'ChEX', **kwargs):
        super().__init__(config, config_cls=SentenceGroundingEvalConfig, **kwargs)
        from model.chex import ChEX
        self.model: ChEX = model
        config = self.config

        assert self.dataset.has_class_box_sentences
        assert self.dataset.class_names is not None and len(self.dataset.class_names) > 0, 'Dataset does not have class names (missing class_names in the config)'
        self.class_names = self.dataset.class_names

        self._register_metric(SentenceDetectionMetrics(
            map_iou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ap_iou_thresholds=[0.1, 0.3, 0.5],
            mIoU_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
            obj_threshold=config.obj_threshold))

    def _predict(self, 
        x: torch.Tensor, 
        grounded_sentences: List[List[str]],
        target_sentence_boxes: List[List[torch.FloatTensor]],
        **kwargs) -> SentenceGroundingOutput:
        """
        :param x: (N x C x H x W)
        :param grounding_sentences: List (N) of lists (S_i) of strings
        :param target_sentence_cls_boxes: List (N) of lists (S_i) of tensors (M_is x 5) in the (x_c, y_c, w, h, class_id) format
        """

        # Encode Sentences
        encoded_sentences: TextEncoderOutput = self.model.encode_sentences(grounded_sentences, device=target_sentence_boxes[0][0].device)
        # Detect
        detected_regions: TokenDetectorOutput = self.model.detect_prompts(x, encoded_sentences.sentence_features, box_prompt_mask=encoded_sentences.sentence_mask, 
                                                                          clip_boxes=self.config.clip_boxes,
                                                                          skip_roi_pool=self.config.skip_roi_pool_inference,
                                                                          use_post_decoder=False)
        # ipdb.set_trace()
        # 创建保存目录
        import os
        from PIL import Image, ImageDraw, ImageFont
        output_dir = "SG_raddino_images"
        os.makedirs(output_dir, exist_ok=True)

        # 图像张量转换为 PIL 图像的函数
        def tensor_to_pil_image(tensor):
            if tensor.dim() == 3:  # 如果是 [C, H, W]
                tensor = tensor.permute(1, 2, 0)
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]
            tensor = (tensor * 255).byte()  # 转为 uint8
            return Image.fromarray(tensor.cpu().numpy())

        # 绘制目标框和简化句子的函数
        def draw_boxes(image, target_cls_boxes, pred_boxes, grounded_sentences):
            # 确保图片是 RGB 模式，以便显示彩色字体
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            draw = ImageDraw.Draw(image)
            font = ImageFont.load_default()
            
            # 确保预测框和句子数量匹配
            num_boxes = min(len(target_cls_boxes[0]), len(pred_boxes), len(grounded_sentences))
            
            for i in range(num_boxes):
                target_cls_box = target_cls_boxes[0][i]
                pred_box = pred_boxes[i]
                sentence = grounded_sentences[i]
                
                # 获取目标框坐标并转换为像素
                # ipdb.set_trace()
                x, y, w, h, label = target_cls_box
                x_min = int((x - w / 2) * image.width)
                y_min = int((y - h / 2) * image.height)
                x_max = int((x + w / 2) * image.width)
                y_max = int((y + h / 2) * image.height)
                
                # 获取预测框坐标并转换为像素
                x, y, w, h = pred_box
                x_min_pred = int((x - w / 2) * image.width)
                y_min_pred = int((y - h / 2) * image.height)
                x_max_pred = int((x + w / 2) * image.width)
                y_max_pred = int((y + h / 2) * image.height)    
                
                # 绘制矩形框 - 使用元组而不是列表
                draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
                draw.rectangle((x_min_pred, y_min_pred, x_max_pred, y_max_pred), outline="blue", width=2)

                # 简化句子（只显示第一个和最后一个单词）
                simplified_sentence = " ".join([sentence.split()[0], sentence.split()[-1]])

                # 直接绘制文字
                draw.text((x_min + 2, y_min + 1), simplified_sentence, fill="yellow", font=font)  # 黄色文字

            return image

        # 保存图片和文本信息
        # ipdb.set_trace()
        
        # 处理每个样本
        for i, image_tensor in enumerate(x):
            # 获取当前样本的目标框和句子
            sample_target_boxes = target_sentence_boxes[i]
            
            # 获取当前样本的预测框
            sample_pred_boxes = detected_regions.boxes[i]
            
            # 获取当前样本的句子
            sample_sentences = grounded_sentences[i]
            
            # 确保数据长度匹配
            num_items = min(len(sample_target_boxes), len(sample_sentences))
            if sample_pred_boxes.shape[0] < num_items:
                # 如果预测框数量不足，只处理有预测框的部分
                num_items = sample_pred_boxes.shape[0]
            
            # 如果没有框或句子，跳过此样本
            if num_items == 0:
                continue
                
            # 转换图像为 PIL 格式
            image = tensor_to_pil_image(image_tensor)
            
            # 准备数据
            target_boxes = sample_target_boxes[:num_items]
            pred_boxes = sample_pred_boxes[:num_items].cpu().numpy()
            sentences = sample_sentences[:num_items]
            
            # 在图像上绘制边界框和句子
            annotated_image = draw_boxes(image, target_boxes, pred_boxes, sentences)
            
            # 保存标注后的图像
            output_path = os.path.join(output_dir, f"sample_{i}.jpg")
            annotated_image.save(output_path)
            
            # 保存对应的文本信息
            text_path = os.path.join(output_dir, f"sample_{i}.txt")
            with open(text_path, "w") as f:
                f.write("目标句子:\n")
                for sentence in sentences:
                    f.write(f"- {sentence}\n")
                f.write("\n目标框:\n")
                for box in target_boxes:
                    f.write(f"- {box}\n")
                f.write("\n预测框:\n")
                for box in pred_boxes:
                    f.write(f"- {box}\n")

        return detected_regions, encoded_sentences, x, grounded_sentences, target_sentence_boxes

    def _postprocess(self,
                     detected_regions: TokenDetectorOutput, encoded_sentences: TextEncoderOutput, 
                     x, grounded_sentences, target_sentence_boxes: List[List[torch.FloatTensor]], config: SentenceGroundingEvalConfig):
        # (N x S)
        sentence_mask = encoded_sentences.sentence_mask

        # (N x S x M x 4)
        region_boxes = detected_regions.multiboxes if detected_regions.multiboxes is not None else detected_regions.boxes[:, :, None, :]

        if config.box_scale_factor is not None:
            region_boxes = region_boxes.clone()
            region_boxes[:, :, :, 2:4] *= config.box_scale_factor
            if self.config.clip_boxes:
                region_boxes = clip_bboxes(region_boxes)

        N, S, M, _ = region_boxes.shape
        # (N x S x M)
        region_weights = detected_regions.multiboxes_weights
        # (N x S x M x 5)
        boxes_with_weights = torch.cat([region_boxes, region_weights.unsqueeze(-1)], dim=-1)

        # List (N) of tensors (S_i x M x 5)
        boxes = [
            sample_boxes[sample_mask]
            for sample_boxes, sample_mask in zip(boxes_with_weights, sentence_mask)
        ]
        # List (N) of lists (S_i) of tensors (M_is x 5)
        boxes = [
            [sent_boxes for sent_boxes in sample_boxes]
            for sample_boxes in boxes
        ]

        # Apply NMS
        boxes = [
            [sent_boxes[nms(center_to_corners_format(sent_boxes[:, :4]), scores=sent_boxes[:, -1], iou_threshold=self.config.nms_threshold)] 
            for sent_boxes in sample_boxes]
            for sample_boxes in boxes]

        return SentenceGroundingOutput(
            image=x,
            pred_boxes=boxes,
            target_cls_boxes=target_sentence_boxes,
            target_sentences=grounded_sentences)

    def _update_metrics_with_output(self, output: SentenceGroundingOutput):
        self._update_metric(predicted_boxes=output.pred_boxes, target_boxes=output.target_cls_boxes)
    
    def plot(self, output: SentenceGroundingOutput, max_samples: int, target_dir: str, plot_local):
        return super().plot(output, max_samples, target_dir, plot_local)
    
    def _do_inference_for_optimization(self, predictions: list, config: SentenceGroundingEvalConfig):
        self.reset_metrics()
        for sample_pred in predictions:
            output = self._postprocess(*sample_pred, config=config)
            self._update_metrics_with_output(output)

        metrics = self._get_metric().compute_mAP()
        self.reset_metrics()
        return metrics

    def optimize_inference(self, predictions: list, optimize_scale_factor: bool = True, optimize_nms: bool = True):
        base_config = self.config
        best_config_overwrites = {}
        if optimize_scale_factor:
            log.info('Optimizing box scale factor...')
            # different evaluation datasets draw smaller/larger boxes so we scale the boxes based on the val dataset
            box_scale_sweep = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0]
            best_mAP, best_scale = 0.0, 1.0
            for box_scale in tqdm(box_scale_sweep):
                current_config = deepcopy(base_config)
                current_config.box_scale_factor = box_scale
                mAP = self._do_inference_for_optimization(predictions, current_config)['AP/mAP']
                if mAP >= best_mAP:
                    best_mAP, best_scale = mAP, box_scale
            base_config = deepcopy(base_config)
            base_config.box_scale_factor = best_scale

            best_config_overwrites = {
                **best_config_overwrites,
                'box_scale_factor': base_config.box_scale_factor,
            }
            log.info(f'Optimized box sclaes method: mAP={best_mAP}')

        if optimize_nms:
                best_map, best_config = 0.0, None
                nms_thres_sweep = [0.05, 0.1, 0.25, 0.5]
                for nms_thres in tqdm(nms_thres_sweep):
                    current_config = deepcopy(base_config)
                    current_config.nms_threshold = nms_thres
                    mAP = self._do_inference_for_optimization(predictions, current_config)['AP/mAP']
                    if mAP >= best_map:
                        best_map, best_config = mAP, current_config
                base_config = best_config
                log.info(f'Optimized nms/wbf: mAP={best_map}')

                best_config_overwrites = {
                    **best_config_overwrites,
                    'nms_threshold': best_config.nms_threshold,
                    'wbf_threshold': best_config.wbf_threshold,
                    'multi_region_postprocess': best_config.multi_region_postprocess,
                }

        log.info(f'Best config: {best_config_overwrites}')     
        return base_config
