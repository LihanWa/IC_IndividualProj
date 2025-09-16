from clip.mlp import MLP
import chexzero.clip
from chexzero.model import VisualTransformer
from collections import OrderedDict
from typing import Tuple, Union
import clip.clip as clip
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import ipdb
from settings import MAPLE_LENGTH,CHEX_DIR,MAPLE_LENGTH2
from transformers import AutoModel, AutoImageProcessor
import os
import einops
import logging
import hashlib
import urllib
import warnings
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
log = logging.getLogger(__name__)
class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim//4, bias=False)
        self.key = nn.Linear(in_dim, out_dim//4, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)
        
        # self.query.weight.data = self.query.weight.data.half()
        # self.key.weight.data = self.key.weight.data.half()
        # self.value.weight.data = self.value.weight.data.half()

    def forward(self, x, y):
        q = self.query(x)
        k = self.key(y)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        V = self.value(y)
        output = torch.bmm(attn_weights, V)
        return output + x


class CrossAttention2(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CrossAttention2, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.query = nn.Linear(in_dim, out_dim//4, bias=False)
        self.key = nn.Linear(in_dim, out_dim//4, bias=False)
        # self.query.weight.data = self.query.weight.data.half()
        # self.key.weight.data = self.key.weight.data.half()  

    def forward(self, x, y):
        # ipdb.set_trace()
        x = self.query(x)
        y = self.key(y)
        attn_scores = torch.matmul(x, y.transpose(-2, -1))
        attn_weights = F.sigmoid(attn_scores)
        # attn_scores = torch.matmul(x, y.transpose(-2, -1)) / (self.out_dim ** 0.5)
        # attn_weights = F.softmax(attn_scores, dim=-1)
        
        return attn_weights








class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



''''''''''''''''''''''''''''''''''''''''''''''''''''''
# class ResidualAttentionBlock_MaPLe(nn.Module):
#     def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None, text_layer=False, i=0):
#         super().__init__()

#         self.attn = nn.MultiheadAttention(d_model, n_head)
#         self.ln_1 = LayerNorm(d_model)
#         self.mlp = nn.Sequential(OrderedDict([
#             ("c_fc", nn.Linear(d_model, d_model * 4)),
#             ("gelu", QuickGELU()),
#             ("c_proj", nn.Linear(d_model * 4, d_model))
#         ]))
#         self.ln_2 = LayerNorm(d_model)
#         # For the first iteration i, we do not need to add the learnable parameters here
#         # as it will be added in the beginning, for both text and the vision branch
#         self.text_layer = text_layer
#         self.attn_mask = attn_mask
#         # This must be consistent with the config file prompt
#         self.compound_prompt_nctx = design_details['maple_length']
#         if i == 0:
#             self.first_layer = True
#         else:
#             self.first_layer = False
        
#         ''''''''''''''''''''''''''''''''''''
#         self.i = i
#         self.GAP = nn.AvgPool1d(77)
#         if i>=1 and i<=3:
#             self.CrossAtt = CrossAttention(768, 768)
#             self.CrossRlt = CrossAttention2(768, 768)

#     def attention(self, x: torch.Tensor):
#         self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
#         return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

#     def forward(self, inputs):
#         # For the first layer, we do not need to add any duplicate, as it is already added as the shallow version
#         x = inputs[0]
#         compound_prompts_deeper = inputs[1]
#         counter = inputs[2]
#         x1 = inputs[3]
#         x2 = inputs[4]
#         JP_prompts = inputs[5] #image A promt
#         BL_prompts = inputs[6] #image P prompt
#         xy = inputs[7]
        
#         if not self.first_layer:
#             if len(compound_prompts_deeper) > 0: 
#                 if not self.text_layer: # vision layer
#                     if not (counter > len(compound_prompts_deeper) - 1): # 2
#                         # Remove the outputs produced by learnable tokens of previous layer
#                         prefix = x[0:x.shape[0]-self.compound_prompt_nctx-2, :, :]
#                         # prefix = x[:71, :, :]
#                         JP_feature = x1.permute(1, 0, 2) # 64 77 768 
#                         BL_feature = x2.permute(1, 0, 2) # 64 77 768
#                         pmt_feature = x[x.shape[0]-self.compound_prompt_nctx-2:, :, :] # 4 64 768
#                         pmt_feature = pmt_feature.permute(1, 0, 2) # 64 4 768
#                         img_feature = compound_prompts_deeper[counter] # 2 768
#                         img_feature = img_feature.expand(x.shape[1], -1, -1) # 64 2 768
#                         # visual_context = self.CrossAtt(img_feature.half(), torch.cat([JP_feature, BL_feature, pmt_feature], dim=1).half()) # 64 2 768
#                         visual_context = self.CrossAtt(img_feature, torch.cat([JP_feature, BL_feature, pmt_feature], dim=1)) # 64 2 768
#                         visual_context = visual_context.permute(1, 0, 2) # 2 64 768

#                         JP_context = JP_prompts[counter]    
#                         JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half() # 1 64 768
#                         JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1).half() # 解剖学提示+GAP的解剖学特征
                        
#                         BL_context = BL_prompts[counter]
#                         BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
#                         BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1).half()
#                         # Add the learnable tokens of this layer with the input, replaced by previous layer learnable tokens
#                         x = torch.cat([prefix, visual_context, JP_context, BL_context], dim=0)
                        
#                         # if counter == len(compound_prompts_deeper) - 1:
#                         xyxy = self.CrossRlt(visual_context.permute(1, 0, 2), torch.cat([JP_feature, BL_feature], dim=1))
#                         # xyxy = self.CrossRlt(visual_context.permute(1, 0, 2).half(), torch.cat([JP_feature, BL_feature], dim=1).half())
#                         xy[:, counter*2:(counter+1)*2, :] = xyxy

#                         # Once done, update the counter, so that the next time, it does not use same learnable tokens
#                         counter += 1
        
#         x = x + self.attention(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
        
#         if self.i <= 3:        
#             x1 = x1 + self.attention(self.ln_1(x1))
#             x1 = x1 + self.mlp(self.ln_2(x1))
        
#             x2 = x2 + self.attention(self.ln_1(x2))
#             x2 = x2 + self.mlp(self.ln_2(x2))
        
#         return [x, compound_prompts_deeper, counter, x1, x2, JP_prompts, BL_prompts, xy]  # return again as a list, so that nn.seq can work
# class Transformer(nn.Module):
#     def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
#                  text_layer=False, design_details=None):
#         super().__init__()
#         self.width = width
#         self.layers = layers
#         # Implements respective encoder blocks for a given design choice

#         self.resblocks = nn.Sequential(
#             *[ResidualAttentionBlock_MaPLe(width, heads, attn_mask, design_details, text_layer, i)
#                 for i in range(layers)])

#     def forward(self, x: torch.Tensor):
#         return self.resblocks(x)
class Resblock_text(nn.Module):
    def __init__(self,resblock_text,design_details,text_layer,i=0):
        super().__init__()
        self.resblock_text = resblock_text
        self.text_layer = text_layer
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        self.i = i
        self.GAP = nn.AvgPool1d(77)
        if i>=1 and i<=4:
            self.CrossAtt = CrossAttention(512, 512)
            self.CrossRlt = CrossAttention2(512, 512)
        
    def forward(self,inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        x1 = inputs[3]
        x2 = inputs[4]
        JP_prompts = inputs[5]
        BL_prompts = inputs[6]
        xy = inputs[7]
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0: 
                if self.text_layer: # text layer
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix = x[:1, :, :]                             
                        suffix = x[1+self.compound_prompt_nctx+2*MAPLE_LENGTH:, :, :] 
                        pmt_feature = x[1:1+self.compound_prompt_nctx+2*MAPLE_LENGTH, :, :] # 4 64 512
                        pmt_feature = pmt_feature.permute(1, 0, 2) # 64 4 512
                        # prefix = x[:71, :, :]
                        JP_feature = x1.permute(1, 0, 2) # 64 77 512 
                        BL_feature = x2.permute(1, 0, 2) # 64 77 512
                        text_feature = compound_prompts_deeper[counter] # 2 512
                        text_feature = text_feature.expand(x.shape[1], -1, -1) # 64 2 512
                        # textual_context = self.CrossAtt(text_feature.half(), torch.cat([JP_feature, BL_feature, pmt_feature], dim=1).half()) # 64 2 512
                        textual_context = self.CrossAtt(text_feature, torch.cat([JP_feature, BL_feature, pmt_feature], dim=1)) # 64 2 512
                        # ipdb.set_trace()
                        # textual_context = textual_context.permute(1, 0, 2).half() # 2 64 512
                        textual_context = textual_context.permute(1, 0, 2) # 2 64 512

                        JP_context = JP_prompts[counter]    
                        JP_context = JP_context[:, 0:512]
                        JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2) # 1 64 512
                        JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1) # 解剖学提示+GAP的解剖学特征
                        # JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half() # 1 64 512
                        # JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1).half() # 解剖学提示+GAP的解剖学特征
                        
                        BL_context = BL_prompts[counter]
                        BL_context = BL_context[:, 0:512]
                        BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2)
                        BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1)
                        # BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1).half()
                        
                        # Add the learnable tokens of this layer with the input, replaced by previous layer learnable tokens
                        x = torch.cat([prefix, textual_context, JP_context, BL_context, suffix], dim=0)
                        
                        # if counter == len(compound_prompts_deeper) - 1:
                        # xyxy = self.CrossRlt(textual_context.permute(1, 0, 2).half(), torch.cat([JP_feature, BL_feature], dim=1).half())
                        xyxy = self.CrossRlt(textual_context.permute(1, 0, 2), torch.cat([JP_feature, BL_feature], dim=1))
                        xy[:, counter*(2*MAPLE_LENGTH):(counter+1)*(2*MAPLE_LENGTH), :] = xyxy
                        
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
        
        x = self.resblock_text(x)
        
        if self.i <= 4:        
            x1 = self.resblock_text(x1)
        
            x2 = self.resblock_text(x2)
        # ipdb.set_trace()
        return [x, compound_prompts_deeper, counter, x1, x2, JP_prompts, BL_prompts, xy]
class Layer(nn.Module):
    def __init__(self, layer,design_details=None, text_layer=False, i=0):
        super().__init__()
        self.compound_prompt_nctx = design_details['maple_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        
        ''''''''''''''''''''''''''''''''''''
        self.i = i
        self.GAP = nn.AvgPool1d(77)
        if i>=1 and i<=4:
            self.CrossAtt = CrossAttention(768, 768)
            self.CrossRlt = CrossAttention2(768, 768)
        self.layer = layer
        self.CrossMLP=nn.Linear(768,768)
        # self.CrossMLP2=nn.Linear(768,768)

    def forward(self,inputs):
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        x1 = inputs[3]
        x2 = inputs[4]
        JP_prompts = inputs[5] #image A promt
        BL_prompts = inputs[6] #image P prompt
        xy = inputs[7]
        # ipdb.set_trace()
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0: 
                if not self.text_layer: # vision layer
                    if not (counter > len(compound_prompts_deeper) - 1): # 2
                        # ipdb.set_trace()
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0]-self.compound_prompt_nctx-2*MAPLE_LENGTH2, :, :]
                        # prefix = x[:71, :, :]
                        JP_feature = x1.permute(1, 0, 2) # 64 77 768 
                        BL_feature = x2.permute(1, 0, 2) # 64 77 768
                        pmt_feature = x[x.shape[0]-self.compound_prompt_nctx-2*MAPLE_LENGTH2:, :, :] # 4 64 768
                        pmt_feature = pmt_feature.permute(1, 0, 2) # 64 4 768
                        # ipdb.set_trace()
                        pmt_feature = self.CrossMLP(pmt_feature)

                        # pmt_feature = self.CrossMLP2(pmt_feature)
                        img_feature = compound_prompts_deeper[counter] # 2 768
                        img_feature = img_feature.expand(x.shape[1], -1, -1) # 64 2 768
                        
                        # visual_context = self.CrossAtt(img_feature.half(), torch.cat([JP_feature, BL_feature, pmt_feature], dim=1).half()) # 64 2 768
                        visual_context = self.CrossAtt(img_feature, torch.cat([JP_feature, BL_feature, pmt_feature], dim=1)) # 64 2 768
                        # ipdb.set_trace()
                        visual_context = visual_context.permute(1, 0, 2) # 2 64 768

                        JP_context = JP_prompts[counter]    
                        # JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half() # 1 64 768
                        # JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1).half() # 解剖学提示+GAP的解剖学特征
                        JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2) # 1 64 768
                        JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1) # 解剖学提示+GAP的解剖学特征
                        
                        BL_context = BL_prompts[counter]
                        # BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half    
                        # BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1).half()
                        # ipdb.set_trace()
                        BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2) # 1 64 768
                        BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1) # 解剖学提示+GAP的解剖学特征
                        # ipdb.set_trace()
                        # Add the learnable tokens of this layer with the input, replaced by previous layer learnable tokens
                        x = torch.cat([prefix, visual_context, JP_context, BL_context], dim=0)
                        
                        # if counter == len(compound_prompts_deeper) - 1:
                        # xyxy = self.CrossRlt(visual_context.permute(1, 0, 2).half(), torch.cat([JP_feature, BL_feature], dim=1).half())
                        # ipdb.set_trace()
                        xyxy = self.CrossRlt(visual_context.permute(1, 0, 2), torch.cat([JP_feature, BL_feature], dim=1))
                        # 检查xyxy的第二维度是否大于2
                        if xyxy.shape[1] > 2:
                            # 使用1D卷积将第二维度从大于2的值卷积到2

                            
                            # 应用卷积
                            # 使用平均池化将第二维度从16变为2
                            xyxy = torch.mean(xyxy.reshape(xyxy.shape[0], 2, -1, xyxy.shape[2]), dim=2)

                        xy[:, counter*(2*MAPLE_LENGTH):(counter+1)*(2*MAPLE_LENGTH), :] = xyxy
                        # ipdb.set_trace()
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                        # ipdb.set_trace()
        x = self.layer(x)[0]
        
        if self.i <= 4:        
            x1 = self.layer(x1)[0]
        
            x2 = self.layer(x2)[0] 
        
        return [x, compound_prompts_deeper, counter, x1, x2, JP_prompts, BL_prompts, xy] 



        
''''''''''''''''''''''''''''''''''''''''''''''''''''''

''''''''''''''''''''''''''''''''''''''''''''''''''''''
class ResidualAttentionBlock_MaPLe_text(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None, text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        # For the first iteration i, we do not need to add the learnable parameters here
        # as it will be added in the beginning, for both text and the vision branch
        self.text_layer = text_layer
        self.attn_mask = attn_mask
        # This must be consistent with the config file prompt
        self.compound_prompt_nctx = design_details['maple_length']
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        
        ''''''''''''''''''''''''''''''''''''
        self.i = i
        self.GAP = nn.AvgPool1d(77)
        if i>=1 and i<=3:
            self.CrossAtt = CrossAttention(512, 512)
            self.CrossRlt = CrossAttention2(512, 512)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        # For the first layer, we do not need to add any duplicate, as it is already added as the shallow version
        x = inputs[0]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]
        x1 = inputs[3]
        x2 = inputs[4]
        JP_prompts = inputs[5]
        BL_prompts = inputs[6]
        xy = inputs[7]
        
        if not self.first_layer:
            if len(compound_prompts_deeper) > 0: 
                if self.text_layer: # text layer
                    # First check if the ith layer needs compound prompts or not
                    if not (counter > len(compound_prompts_deeper) - 1):
                        prefix = x[:1, :, :]                             
                        suffix = x[1+self.compound_prompt_nctx+2:, :, :] 
                        pmt_feature = x[1:1+self.compound_prompt_nctx+2, :, :] # 4 64 512
                        pmt_feature = pmt_feature.permute(1, 0, 2) # 64 4 768
                        # prefix = x[:71, :, :]
                        JP_feature = x1.permute(1, 0, 2) # 64 77 512 
                        BL_feature = x2.permute(1, 0, 2) # 64 77 512
                        text_feature = compound_prompts_deeper[counter] # 2 512
                        text_feature = text_feature.expand(x.shape[1], -1, -1) # 64 2 512
                        textual_context = self.CrossAtt(text_feature.half(), torch.cat([JP_feature, BL_feature, pmt_feature], dim=1).half()) # 64 2 512
                        textual_context = textual_context.permute(1, 0, 2).half() # 2 64 512

                        JP_context = JP_prompts[counter]    
                        JP_context = JP_context[:, 0:512]
                        JP_context = JP_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half() # 1 64 512
                        JP_context = JP_context + self.GAP(JP_feature.permute(0, 2, 1)).permute(2, 0, 1).half() # 解剖学提示+GAP的解剖学特征
                        
                        BL_context = BL_prompts[counter]
                        BL_context = BL_context[:, 0:512]
                        BL_context = BL_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        BL_context = BL_context + self.GAP(BL_feature.permute(0, 2, 1)).permute(2, 0, 1).half()
                        
                        # Add the learnable tokens of this layer with the input, replaced by previous layer learnable tokens
                        x = torch.cat([prefix, textual_context, JP_context, BL_context, suffix], dim=0)
                        
                        # if counter == len(compound_prompts_deeper) - 1:
                        xyxy = self.CrossRlt(textual_context.permute(1, 0, 2).half(), torch.cat([JP_feature, BL_feature], dim=1).half())
                        xy[:, counter*(2*MAPLE_LENGTH):(counter+1)*(2*MAPLE_LENGTH), :] = xyxy
                        
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                        
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        
        if self.i <= 3:        
            x1 = x1 + self.attention(self.ln_1(x1))
            x1 = x1 + self.mlp(self.ln_2(x1))
        
            x2 = x2 + self.attention(self.ln_1(x2))
            x2 = x2 + self.mlp(self.ln_2(x2))
        
        return [x, compound_prompts_deeper, counter, x1, x2, JP_prompts, BL_prompts, xy]  # return again as a list, so that nn.seq can work

class Transformer_text(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=0,
                 text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        # Implements respective encoder blocks for a given design choice
        current_trainer = design_details['trainer']
        if current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
                *[ResidualAttentionBlock_MaPLe_text(width, heads, attn_mask, design_details, text_layer, i)
                  for i in range(layers)])
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
''''''''''''''''''''''''''''''''''''''''''



class VisionTransformer_MaPLe(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.VPT_shallow = True
        scale = width ** -0.5
        # import ipdb; ipdb.set_trace()
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.Crosspositional_embedding = nn.Parameter(scale * torch.randn(77, width))
        self.ln_pre = LayerNorm(width)
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        import ipdb
        # ipdb.set_trace()
    def forward(self, x: torch.Tensor, shared_ctx, JP_prompts, BL_prompts, compound_deeper_prompts, imgs_JP, imgs_BL):
        x = self.conv1(x)                          # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)                     # shape = [*, grid ** 2, width]  64 196 768
        # x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = torch.cat([self.class_embedding.half() + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)                               # 64 197 768

        # After positional embeddings, we will attach prompts with the model, remember only those are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0

        # Normal code as before
        x = self.ln_pre(x)      # 64 201 768
        x = x.permute(1, 0, 2)  # NLD -> LND  201 64 768  

        x1 = imgs_JP + self.Crosspositional_embedding.to(x.dtype) #proj(txt_jp)
        x2 = imgs_BL + self.Crosspositional_embedding.to(x.dtype)
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        ''''''
        # import ipdb; ipdb.set_trace()
        # Again combine the inputs, so nn.sequential can work
        outputs = self.transformer([x, compound_deeper_prompts, 0, x1, x2, JP_prompts, BL_prompts, torch.empty(x.shape[1], 4*(2*MAPLE_LENGTH), 154)])  # third argument is counter
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:,0,:])

        if self.proj is not None:
            x = x @ self.proj
            
        xy = outputs[7]
        return x, xy



class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, num_queries=1):
        super().__init__()
        # 可学习的query向量
        self.query = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.scale = embed_dim ** -0.5

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embed_dim]
        Returns:
            pooled: Tensor, shape [batch_size, num_queries, embed_dim] 或 [batch_size, embed_dim] (如果num_queries=1)
        """
        # [batch_size, num_queries, embed_dim] × [batch_size, embed_dim, seq_len] → [batch_size, num_queries, seq_len]
        attn_weights = torch.matmul(self.query.unsqueeze(0), x.transpose(1, 2)) * self.scale
        
        # softmax归一化
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 加权求和 [batch_size, num_queries, seq_len] × [batch_size, seq_len, embed_dim] → [batch_size, num_queries, embed_dim]
        pooled = torch.matmul(attn_weights, x)
        
        if pooled.size(1) == 1:
            pooled = pooled.squeeze(1)  # 若只有一个query，则返回[batch_size, embed_dim]
        
        return pooled

class Raddino(nn.Module):
    def __init__(self, backbone,design_details):
        super().__init__()
        # self.backbone = backbone.half()
        self.backbone = backbone
        width = 768
        scale = width ** -0.5
        self.layers=[]
        for i,layer in enumerate(self.backbone.encoder.layer):
            self.layers.append(Layer(layer,design_details=design_details,i=i))
        self.layers=nn.Sequential(*self.layers)
        self.Crosspositional_embedding = nn.Parameter(scale * torch.randn(77, width))
        # DINOv2输出维度是768，需要投影到CLIP的embed_dim
        self.CrossPatch_projection = MLP(
            n_layers=1, 
            d_in=768, 
            d_out=512, 
            use_bn=False,
            dropout=0.2)
        self.CrossAttentionPooling = CrossAttentionPooling(embed_dim=768, num_queries=8)
        self.CrossFinalAttentionPooling = CrossAttentionPooling(embed_dim=768, num_queries=1)

        # ckpt_dict = torch.load('/rds/general/user/lw1824/home/chex/chex/models/contr_train_img_txt/run_2025-04-26_19-55-13/checkpoints/checkpoint_best.pth', map_location=torch.device('cpu'))
        # # 提取图像编码器的patch_projection参数
        # patch_proj_params = {k.replace('img_encoder.patch_projection.', ''): v 
        #                     for k, v in ckpt_dict['state_dict'].items() 
        #                     if k.startswith('img_encoder.patch_projection.')}
        # # raise NotImplementedError("Rad-DINO的patch_projection参数加载未实现")
        # missing, unexpected = self.CrossPatch_projection.load_state_dict(patch_proj_params, strict=False)
        # if missing:
        #     log.warning(f"加载patch_projection时缺少参数: {missing}")
        # if unexpected:
        #     log.warning(f"加载patch_projection时有意外参数: {unexpected}")
        # 打印patch_proj_params参数的一小部分
        # if patch_proj_params:
        #     print("patch_proj_params参数示例:")
        #     sample_keys = list(patch_proj_params.keys())[:3]  # 获取前3个键
        #     for key in sample_keys:
        #         print(f"  - {key}: 形状={patch_proj_params[key].shape}, 类型={patch_proj_params[key].dtype}")
        #         # 打印参数的一小部分值
        #         param_sample = patch_proj_params[key].flatten()[:5]
        #         print(f"    值样本: {param_sample}")
        log.info("成功从检查点加载patch_projection参数")
        self.ln_post = LayerNorm(768)
    def forward(self, x,shared_ctx,JP_prompts,BL_prompts,compound_deeper_prompts,imgs_JP,imgs_BL):
        x = x.to('cuda')

        if x.ndim == 3:
            x = einops.repeat(x, 'n h w -> n c h w', c=3)
        
        H = int(x.shape[2]/14)
        W = int(x.shape[3]/14)
        if H != W:
            ipdb.set_trace()
        assert H == W, "只支持正方形图像"
        
        # 使用骨干网络编码图像
        # with torch.set_grad_enabled(not self.config.frozen_backbone):
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

        visual_ctx = shared_ctx.expand(x.shape[0], -1, -1)

        x = torch.cat([x, visual_ctx], dim=1)
        x1 = imgs_JP + self.Crosspositional_embedding.to(x.dtype)
        x2 = imgs_BL + self.Crosspositional_embedding.to(x.dtype)

        # 通过transformer层
        x=x.permute(1, 0, 2)
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        outputs = self.layers([x,compound_deeper_prompts,0,x1,x2,JP_prompts,BL_prompts,torch.empty(x.shape[1], 4*(2*MAPLE_LENGTH), 154)])
        # ipdb.set_trace()
        # ipdb.set_trace()
        x = outputs[0]
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        prompt_tokens=x[:,x.shape[1]-visual_ctx.shape[1]:,:]
        image_tokens=x[:,0:x.shape[1]-visual_ctx.shape[1],:]
        pooled_image=self.CrossAttentionPooling(image_tokens)
        if pooled_image.dim() == 2:
            pooled_image = pooled_image.unsqueeze(1)  # [batch, 1, embed_dim]
        combined_tokens=torch.cat([pooled_image,prompt_tokens],dim=1)
        final_feat = self.CrossFinalAttentionPooling(combined_tokens)
        x = self.CrossPatch_projection(final_feat)
        # x = self.CrossPatch_projection(x[:,0,:])
        # x = self.CrossPatch_projection2(x)
        xy = outputs[7]
        return x,xy

class CLIP(nn.Module):
    def __init__(self,

                 ):
        super().__init__()
        design_details = {"vision_depth": 0, "language_depth": 0, "vision_ctx": 0, "language_ctx": 0, "maple_length": 2, "use_raddino": True}
        self.use_raddino = design_details.get('use_raddino', False)
        
        # 如果使用DINOv2，一些参数可以是默认值
        if self.use_raddino:
            log.info("使用Raddino模型")
            vision_layers = 12  # DINOv2-base有12层
            vision_width = 768  # DINOv2-base的宽度
            vision_patch_size = 14  # DINOv2的patch size
            image_resolution = 512  # 标准输入分辨率
            self.context_length = 77
            # trainer = design_details['trainer']

        
            local_repo_path: str = os.path.join(os.environ.get('CHEX_DIR', ''), "chex/chex/cache/rad-dino")
            # self.backbone = AutoModel.from_pretrained(local_repo_path).half()
            self.backbone = AutoModel.from_pretrained(local_repo_path)
            self.visual = Raddino(self.backbone,design_details)
            
        else:
            backbone_name = "ViT-B/32"
            # 直接使用模型URL
            model_url = "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
            model_path = clip._download(model_url)
            
            try:  # loading JIT archive
                model = torch.jit.load(model_path, map_location="cpu").eval()
                state_dict = model.state_dict()
            except RuntimeError:
                state_dict = torch.load(model_path, map_location="cpu") #model 框架建好

            embed_dim = state_dict["text_projection"].shape[1]
            context_length = state_dict["positional_embedding"].shape[0]
            vocab_size = state_dict["token_embedding.weight"].shape[0]
            transformer_width = state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64
            transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
            vision_heads = vision_width // 64
            self.visual = VisionTransformer_MaPLe(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details
            )
        #         else:
        #             self.visual = VisionTransformer(
        #                 input_resolution=image_resolution,
        #                 patch_size=vision_patch_size,
        #                 width=vision_width,
        #                 layers=vision_layers,
        #                 heads=vision_heads,
        #                 output_dim=embed_dim,
        #                 design_details=design_details
        #             )
        # hyper-parameter if need to add prompt embeddings inside to the input
        # of transformer block or not:
        prompt_till_layer_text = design_details['language_depth']
        
        self.text_encoder, _ = chexzero.clip.load("ViT-B/32", device='cpu', jit=False) 
        # # 第二次加载：加载CheXzero预训练权重到model
        self.text_encoder.load_state_dict(torch.load('/rds/general/user/lw1824/home/chex/chex/models/third_party/chexzero/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt', map_location='cpu'))
        # 只保存需要的部分
        self.token_embedding = self.text_encoder.token_embedding
        self.positional_embedding = self.text_encoder.positional_embedding
        self.ln_final = self.text_encoder.ln_final
        self.Cross_text_projection = self.text_encoder.text_projection
        # # 打印text_projection参数的一小部分
        # if hasattr(self, 'text_projection') and self.text_projection is not None:
        #     print("text_projection参数信息:")
        #     print(f"形状: {self.text_projection.shape}")
        #     print(f"数据类型: {self.text_projection.dtype}")
        #     # 只打印前5个值作为示例
        #     print(f"前5个值: {self.text_projection[:5, :5]}")
        #     # 检查是否有NaN值
        #     print(f"是否包含NaN值: {torch.isnan(self.text_projection).any()}")
        # 然后删除整个text_encoder

        self.transformer = []
        for i,resblock in enumerate(self.text_encoder.transformer.resblocks):
            self.transformer.append(Resblock_text(resblock,text_layer=True,design_details=design_details,i=i))
        self.transformer=nn.Sequential(*self.transformer)

        self.vocab_size = self.text_encoder.token_embedding.weight.shape[0]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        del self.text_encoder

        # self.initialize_parameters()

   

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        if self.use_raddino:
            return self.backbone.embeddings.patch_embeddings.projection.weight.dtype
        else:
            return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model():
    # 检查是否使用DINOv2

    # else:
    #     # 原始的代码继续判断vit还是resnet
    #     vit = "visual.proj" in state_dict
    #     if vit:
    #         vision_width = state_dict["visual.conv1.weight"].shape[0]
    #         vision_layers = len(
    #             [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    #         vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    #         grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    #         image_resolution = vision_patch_size * grid_size
    #     else:
    #         counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
    #                         [1, 2, 3, 4]]
    #         vision_layers = tuple(counts)
    #         vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    #         output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
    #         vision_patch_size = None
    #         assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    #         image_resolution = output_width * 32
    #         ipdb.set_trace()


    # model = CLIP(
    #     embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
    #     context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    # )
    model = CLIP()
    # context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details)
    # # ipdb.set_trace()
    # for key in ["input_resolution", "context_length", "vocab_size"]:
    #     if key in state_dict:
    #         del state_dict[key]

    # convert_weights(model)
    # try:
    #     model.load_state_dict(state_dict)
    # except:
    #     missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    #     # print('Weights not found for some missing keys: ', missing_keys)
        
    # # return model.eval()
    # 加载权重
    # if use_raddino:
    #     # 只加载文本编码器部分
    #     text_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('visual.')}
    #     model.load_state_dict(text_state_dict, strict=False)
        
    #     # DINOv2模型已经在创建时加载了预训练权重，不需要额外加载
    # else:
    #     # 原有的权重加载逻辑

    #     for key in ["input_resolution", "context_length", "vocab_size"]:
    #         if key in state_dict:
    #             del state_dict[key]

        # convert_weights(model)
    #     try:
    #         model.load_state_dict(state_dict)
    #     except:
    #         missing_keys, _ = model.load_state_dict(state_dict, strict=False)
    #         # print('Weights not found for some missing keys: ', missing_keys)

    return model.eval()
    



