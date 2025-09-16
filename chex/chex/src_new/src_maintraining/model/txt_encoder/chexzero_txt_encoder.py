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

        # 第一次加载：加载基础CLIP模型
        model, _ = chexzero.clip.load("ViT-B/32", device='cpu', jit=False) 
        # txt_encoder_params = None
        # # 第二次加载：加载CheXzero预训练权重到model
        model.load_state_dict(torch.load(os.path.join(os.environ.get('CHEX_DIR'),'chex/chex/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt'), map_location='cpu'))
        # import ipdb; ipdb.set_trace()
        use_ckpt_from_pretrained = False
        if use_ckpt_from_pretrained: #如果用别的run,要设置False;如果从预训练过来，要设置True，但是需要更改run_name。run_name在wandb中
            # 第三次加载：加载另一个检查点
            ckpt_dict = torch.load('/rds/general/user/lw1824/home/chex/chex/models/contr_train_img_txt/run_2025-04-26_19-55-13/checkpoints/checkpoint_best.pth', map_location=torch.device('cpu'))
            # 检验是否正确加载模型权重
            try:
                # 提取文本编码器的参数
                txt_encoder_params = {k.replace('txt_encoder.', ''): v 
                                    for k, v in ckpt_dict['state_dict'].items() 
                                    if k.startswith('txt_encoder.')}


                # 统计参数数量
                total_params = len(txt_encoder_params)
                log.info(f"从检查点中提取了{total_params}个文本编码器参数")
                
                # 尝试加载文本编码器参数
                missing_keys, unexpected_keys = model.load_state_dict(txt_encoder_params, strict=False)

                
                # 记录加载结果
                if missing_keys:
                    log.warning(f"加载文本编码器时缺少参数: {len(missing_keys)}个")
                    log.warning(f"缺少的参数: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
                
                if unexpected_keys:
                    log.warning(f"加载文本编码器时有意外参数: {len(unexpected_keys)}个")
                    log.warning(f"意外的参数: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
                log.info("成功加载模型权重")
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
            
        if use_ckpt_from_pretrained:
            projection_params = {}
            for k, v in txt_encoder_params.items():
                if k.startswith('projection.layers.'):
                    new_key = k.replace('projection.layers.', 'layers.')
                    projection_params[new_key] = v
        
            # 加载调整后的参数到projection模块
            missing_keys, unexpected_keys = self.projection.load_state_dict(projection_params, strict=False)
            if missing_keys:
                log.warning(f"加载projection时缺少参数: {len(missing_keys)}个")
                log.warning(f"缺少的参数: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
            if unexpected_keys:
                log.warning(f"加载projection时有意外参数: {len(unexpected_keys)}个")
                log.warning(f"意外的参数: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)
        
        if True:
        # if self.config.frozen_language_model:
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

            # 添加调试信息
        # print("Initial projection parameters status:")
        # for name, param in self.projection.named_parameters():
        #     print(f"{name}: requires_grad = {param.requires_grad}")


        self.cached_sentence_embeddings = {}
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
            # import ipdb; ipdb.set_trace()
            features = x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)] @ self.text_projection

        if project:
            features = self.projection(features)
        return features

    
    def encode_sentences(self, sentences: List[str], cache=False, **kwargs) -> Tensor:
        if cache and self.config.frozen_language_model and all(s in self.cached_sentence_embeddings for s in sentences):
            features = torch.stack([self.cached_sentence_embeddings[s] for s in sentences], dim=0)
        else:
            # import ipdb; ipdb.set_trace()
            input_ids = chexzero.clip.tokenize(sentences, context_length=77) 
            features = self(input_ids, project=False, **kwargs)

            if cache and self.config.frozen_language_model: # 如果冻结了语言模型，则不更新缓存
                for s, f in zip(sentences, features):
                    self.cached_sentence_embeddings[s] = f.detach().float()
        features = features.to(dtype=self.dtype)
        projected_features = self.projection(features)
        return projected_features
        