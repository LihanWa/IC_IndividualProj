from dataclasses import dataclass
import logging
import os
from typing import Optional, Union, Literal, Tuple
import einops
from torch import Tensor
from torch import nn
import torch
import torch.nn.functional as F
from chexzero.model import VisualTransformer
from model.components.mlp import MLP
from model.img_encoder import ImageEncoderOutput
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

import ipdb
from util.model_utils import BaseModel, BaseModelConfig, MainModelConfig
import chexzero.clip

log = logging.getLogger(__name__)

@dataclass
class ImageEncoderConfig(BaseModelConfig): #有哪些选择
    # 将可修改属性定义在配置类中
    MODIFYABLE_ATTRIBUTES: Tuple[str, ...] = ('frozen_backbone', 'model_type')
    pretrain: bool = False
    unfreeze_last_layer: bool = False
    # 添加模型类型选择
    model_type: str= "chexzero"
    
    # CheXzero相关配置
    chexzero_path: str = os.path.expanduser(os.path.join(os.environ.get('CHEX_DIR', ''), 
                          'chex/chex/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt'))
    
    # Rad-DINO相关配置
    rad_dino_path: str = os.path.join(os.environ.get('CHEX_DIR', ''), "chex/chex/cache/rad-dino")
    
    # 通用配置
    frozen_backbone: bool = False
    freeze_backbone_layers: Optional[int] = None
    add_cls_features: bool = False
    use_pretrained_projection: bool = True
    additional_projection_layers: int = 0
    projection_bn: bool = False
    normalize_projected: bool = False

class ImageEncoder(BaseModel): #选择
    CONFIG_CLS = ImageEncoderConfig

    def __init__(self, config: ImageEncoderConfig, main_config: MainModelConfig):
        super().__init__(config)
        self.config: ImageEncoderConfig
        self.d = main_config.d_model
        self.model_type = config.model_type
        #先定义了patch_projection
        if self.model_type == "raddino":
            self.d_backbone = 768
        else:
            self.d_backbone = self.backbone.output_dim if self.config.use_pretrained_projection else self.backbone.proj.shape[0]
            
        self.patch_projection = MLP(
            self.config.additional_projection_layers, 
            d_in=self.d_backbone, 
            d_out=self.d, 
            use_bn=self.config.projection_bn,
            d_hidden_factor=4,
            act=main_config.act,
            dropout=main_config.dropout)
        # 打印patch_projection的结构和参数信息
        log.info(f"=== patch_projection 结构 ===")
        log.info(f"输入维度: {self.d_backbone}, 输出维度: {self.d}")
        log.info(f"额外投影层数: {self.config.additional_projection_layers}")
        log.info(f"使用批归一化: {self.config.projection_bn}")
        log.info(f"投影层结构: {self.patch_projection}")
        
        # 计算并打印参数数量
        patch_proj_params = sum(p.numel() for p in self.patch_projection.parameters())
        patch_proj_trainable = sum(p.numel() for p in self.patch_projection.parameters() if p.requires_grad)
        log.info(f"patch_projection 参数总数: {patch_proj_params}, 可训练参数: {patch_proj_trainable}")
        # 根据模型类型初始化不同的编码器
        if self.model_type == "chexzero":
            self._init_chexzero_encoder()
        elif self.model_type == "raddino":
            self._init_rad_dino_encoder()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 共享的投影层
        
        
    def _init_rad_dino_encoder(self):
        """初始化Rad-DINO编码器"""
        
        # 初始化backbone
        local_repo_path = self.config.rad_dino_path
        self.backbone = AutoModel.from_pretrained(local_repo_path)
        self.backbone = self.backbone.to('cuda')
        self.d_backbone = 768  # Rad-DINO的特征维度
        
        # 冻结设置
        if self.config.frozen_backbone:
            # 冻结所有参数
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        #     # 解冻最后一层
        #     if self.config.pretrain and self.config.unfreeze_last_layer:
        #         if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
        #             last_layer_idx = len(self.backbone.encoder.layer) - 1
        #             for name, param in self.backbone.named_parameters():
        #                 if f'encoder.layer.{last_layer_idx}' in name:
        #                   global_featuresglobal_features  param.requires_grad = True
        #             log.info(f'冻结Rad-DINO网络，但保持最后一层(layer.{last_layer_idx})可训练')
        #         else:
        #             log.info('冻结Rad-DINO网络')
        self.check_frozen_status()
        
        
    def check_frozen_status(self):
        """检查并打印模型各部分的冻结状态"""
        log.info("=== 模型冻结状态检查 ===")
        
        # 检查backbone的冻结状态
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_frozen = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        backbone_ratio = backbone_frozen / backbone_params if backbone_params > 0 else 0
        log.info(f"Backbone: 冻结参数 {backbone_frozen}/{backbone_params} ({backbone_ratio:.2%})")
        
        # 检查patch_projection的冻结状态（如果存在）
        if hasattr(self, 'patch_projection'):
            patch_proj_params = sum(p.numel() for p in self.patch_projection.parameters())
            patch_proj_frozen = sum(p.numel() for p in self.patch_projection.parameters() if not p.requires_grad)
            patch_proj_ratio = patch_proj_frozen / patch_proj_params if patch_proj_params > 0 else 0
            log.info(f'层数是{self.config.additional_projection_layers}')
            log.info(f"PatchProjection: 冻结参数 {patch_proj_frozen}/{patch_proj_params} ({patch_proj_ratio:.2%})")
        else:
            log.info("PatchProjection: 不存在")
        
        log.info("="*50)
    def _freeze_chexzero_layers(self, n_frozen_layers: int):
        """冻结CheXzero的特定层"""
        # 嵌入层
        emb_layer_requires_grad = n_frozen_layers == 0
        for param in self.backbone.conv1.parameters():
            param.requires_grad = emb_layer_requires_grad
        self.backbone.class_embedding.requires_grad = emb_layer_requires_grad
        self.backbone.positional_embedding.requires_grad = emb_layer_requires_grad
        self.backbone.ln_pre.requires_grad = emb_layer_requires_grad

        # Transformer层
        n_transformer_layers = len(self.backbone.transformer.resblocks)
        n_frozen_transformer_layers = n_frozen_layers - 1
        for i, resblock in enumerate(self.backbone.transformer.resblocks):
            layer_requires_grad = i >= n_frozen_transformer_layers
            for param in resblock.parameters():
                param.requires_grad = layer_requires_grad

        # 投影层
        proj_layer_requires_grad = n_frozen_layers > n_transformer_layers + 1
        for param in self.backbone.ln_post.parameters():
            param.requires_grad = proj_layer_requires_grad
        self.backbone.proj.requires_grad = proj_layer_requires_grad

    def forward(self, x: Union[Tensor, list], **kwargs) -> ImageEncoderOutput:
        """统一的前向传播方法，根据模型类型调用不同的实现"""
        if self.model_type == "chexzero":
            return self._forward_chexzero(x, **kwargs)
        elif self.model_type == "raddino":
            return self._forward_rad_dino(x, **kwargs)
        
    def _forward_chexzero(self, x: Tensor, **kwargs) -> ImageEncoderOutput:
        """CheXzero模型的前向传播实现"""
        if x.ndim == 3:
            x = einops.repeat(x, 'n h w -> n c h w', c=3)
        device = x.device
        dtype = x.dtype
        N, _, H, W = x.shape
        assert H == W, "只支持正方形图像"
        input_resolution = H
        assert self.backbone.input_resolution == input_resolution, f"骨干网络输入分辨率({self.backbone.input_resolution})与图像分辨率({input_resolution})不匹配"

        # 使用骨干网络编码图像
        with torch.set_grad_enabled(not self.config.frozen_backbone):
            x = self.backbone.conv1(x)  # shape = [*, width, grid, grid]
            N, _, H, W = x.shape
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat([self.backbone.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
            pos_emb = self.backbone.positional_embedding.to(x.dtype)  # shape = [grid ** 2 + 1, width]
            x = x + pos_emb
            x = self.backbone.ln_pre(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.backbone.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD

            # (N x d)
            cls_features = self.backbone.ln_post(x[:, 0, :])
            if self.config.add_cls_features:
                patch_features = self.backbone.ln_post(x[:, 1:, :] + x[:, 0, :].unsqueeze(1))
            else:
                # (N x HW x d)
                patch_features = self.backbone.ln_post(x[:, 1:, :])

            if self.config.use_pretrained_projection:
                cls_features = cls_features @ self.backbone.proj
                patch_features = patch_features @ self.backbone.proj
                pos_embeddings = pos_emb[1:] @ self.backbone.proj
            else:
                pos_embeddings = pos_emb[1:]

            # (N, H, W, d_backbone)
            patch_features = einops.rearrange(patch_features, 'n (h w) d -> n h w d', h=H, w=W)
            # (N, H, W, d_backbone) -> (N, H, W, d)
            pos_embeddings = einops.repeat(pos_embeddings, '(h w) d -> n h w d', h=H, w=W, n=N)
            
        # 投影特征
        projected_patch_features = self.patch_projection(patch_features)

        cls_features = self.patch_projection(cls_features.unsqueeze(1)).squeeze(1)
        if self.config.additional_projection_layers > 0:
            log.info(f"使用额外的投影层，投影层数量：{self.config.additional_projection_layers}")
            pos_embeddings = self.patch_projection(pos_embeddings)

        if self.config.normalize_projected:
            raise NotImplementedError("should not use normalize_projected")
            projected_patch_features = F.normalize(projected_patch_features, dim=-1)
            cls_features = F.normalize(cls_features, dim=-1)

        return ImageEncoderOutput(
            patch_features=projected_patch_features,
            pos_embeddings=pos_embeddings,
            global_features=cls_features)
    
    def _forward_rad_dino(self, x: Tensor, sample_id: list=None, **kwargs) -> ImageEncoderOutput:
        """Rad-DINO模型的前向传播实现"""
        # # 处理输入图像
        # if sample_id is not None and all(s in self.cached_image_embeddings for s in sample_id):
        #     cls_features = torch.stack([self.cached_image_embeddings[s]['cls_features'] for s in sample_id], dim=0)
        #     patch_features = torch.stack([self.cached_image_embeddings[s]['patch_features'] for s in sample_id], dim=0)
        #     pos_embeddings = torch.stack([self.cached_image_embeddings[s]['pos_embeddings'] for s in sample_id], dim=0)
        # else:
        #     cls_features, patch_features, pos_embeddings = self._forward_rad_dino_no_cache(x,**kwargs)
        #     # 如果有sample_id且需要缓存，则存储特征到缓存中
        #     if sample_id is not None and self.config.frozen_backbone:
        #         for i, sid in enumerate(sample_id):
        #             if sid not in self.cached_image_embeddings:
        #                 # 将特征打包成字典存储，便于后续提取
        #                 self.cached_image_embeddings[sid] = {
        #                     'cls_features': cls_features[i].detach().float(),
        #                     'patch_features': patch_features[i].detach().float(),
        #                     'pos_embeddings': pos_embeddings[i].detach().float()
        #                 }
        cls_features, patch_features, pos_embeddings = self._forward_rad_dino_no_cache(x,**kwargs)
        # 投影特征
        projected_patch_features = self.patch_projection(patch_features)
        cls_features = self.patch_projection(cls_features.unsqueeze(1)).squeeze(1)
        pos_embeddings = self.patch_projection(pos_embeddings)
        
        if self.config.normalize_projected:
            raise NotImplementedError("should not use normalize_projected")
            projected_patch_features = F.normalize(projected_patch_features, dim=-1)
            cls_features = F.normalize(cls_features, dim=-1)
            
        return ImageEncoderOutput(
            patch_features=projected_patch_features,
            pos_embeddings=pos_embeddings,
            global_features=cls_features)
    def _forward_rad_dino_no_cache(self, x: Tensor, **kwargs) -> ImageEncoderOutput:
        # 使用x.device而不是硬编码'cuda'
        device = x.device
        output_img = x
        
        if x.ndim == 3:
            x = einops.repeat(x, 'n h w -> n c h w', c=3)
        
        H = int(x.shape[2]/14)
        W = int(x.shape[3]/14)
        if H != W:
            ipdb.set_trace()
        assert H == W, "只支持正方形图像"
        
        # 使用骨干网络编码图像
        with torch.set_grad_enabled(not self.config.frozen_backbone):
            N = int(x.shape[0])
            
            # 提取特征
            x = self.backbone.embeddings.patch_embeddings(x)  # shape = [N, 768, H/14, W/14]
            x = x.reshape(N, 768, -1)
            x = x.permute(0, 2, 1)
            
            # 添加CLS标记
            cls_token = self.backbone.embeddings.cls_token.expand(N, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            
            # 添加位置编码
            pos_emb = self.backbone.embeddings.position_embeddings.squeeze(0)
            x = x + pos_emb
            
            # 通过transformer层
            for layer in self.backbone.encoder.layer:
                x = layer(x)[0]
                
            # 提取特征
            cls_features = self.backbone.layernorm(x[:, 0, :])

            if self.config.add_cls_features:
                patch_features = self.backbone.layernorm(x[:, 1:, :] + x[:, 0, :].unsqueeze(1))
            else:
                patch_features = self.backbone.layernorm(x[:, 1:, :])
                
            pos_embeddings = pos_emb[1:]
            
            # 重塑特征
            patch_features = einops.rearrange(patch_features, 'n (h w) d -> n h w d', h=H, w=W)
            pos_embeddings = einops.repeat(pos_embeddings, '(h w) d -> n h w d', h=H, w=W, n=N)

            return cls_features, patch_features, pos_embeddings