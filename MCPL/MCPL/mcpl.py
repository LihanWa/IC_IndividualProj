import os
import copy
import datetime
# import numpy as np
from settings import MAPLE_LENGTH,MAPLE_LENGTH2
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch import distributed as dist

from dateutil import tz
# from einops import rearrange
from argparse import ArgumentParser
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
# from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
import ipdb
from mgca.datasets.data_module import DataModule
from mgca.datasets.transforms import DataTransforms
from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset, multimodal_collate_fn)
# from mgca.models.backbones.encoder import BertEncoder, ImageEncoder

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# from constants import *
import warnings
warnings.filterwarnings("ignore") 

_tokenizer = _Tokenizer()


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class MHAttention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MHAttention, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ln = LayerNorm(in_dim)
        self.query = nn.Linear(in_dim, out_dim//4, bias=False)
        self.key = nn.Linear(in_dim, out_dim//4, bias=False)
        self.value = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.ln(x.permute(1, 0, 2)).permute(1, 0, 2)
        q = self.query(x)
        k = self.key(x)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.out_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        v = self.value(x)
        out = torch.bmm(attn_weights, v)
        return out  


# 构建CLIP模型
def load_clip_to_cpu():
    # backbone_name = "ViT-B/32"
    # url = clip._MODELS[backbone_name]
    # model_path = clip._download(url)
    
    # try:  # loading JIT archive
    #     model = torch.jit.load(model_path, map_location="cpu").eval()
    #     state_dict = None
    # except RuntimeError:
    #     state_dict = torch.load(model_path, map_location="cpu") #model 框架建好

    # model = clip.build_model(state_dict or model.state_dict(), design_details)
    model = clip.build_model()
    # ipdb.set_trace()
    # for p in model.parameters(): 
    # 	p.requires_grad = False
 
    for name, p in model.named_parameters():
        if "Cross" in name:
            print("update only",name)
            p.requires_grad = True
        else:
            p.requires_grad = False     
    # ipdb.set_trace()
    return model


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_attention_mask():
    mask = torch.empty(4, 4)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask


# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer                   
        self.positional_embedding = clip_model.positional_embedding 
        self.ln_final = clip_model.ln_final                         
        self.Cross_textprojection = clip_model.Cross_text_projection           
        self.dtype = clip_model.dtype                               

    def forward(self, texts, tokenized_prompts, JP_prompts, BL_prompts, compound_prompts_text, texts_JP, texts_BL): 
        x = texts + self.positional_embedding.type(self.dtype)      
        x = x.permute(1, 0, 2)                                      
        x1 = texts_JP + self.positional_embedding.type(self.dtype)
        x2 = texts_BL + self.positional_embedding.type(self.dtype)
        x1 = x1.permute(1, 0, 2)
        x2 = x2.permute(1, 0, 2)
        # 使用全局参数MAPLE_LENGTH代替硬编码的7
          # 定义全局参数
        combined = [x, compound_prompts_text, 0, x1, x2, JP_prompts, BL_prompts, torch.empty(x.shape[1], 4*(2*MAPLE_LENGTH), 154)]     # 文本与提示组成输入
        # combined = [x, compound_prompts_text, 0, x1, x2, JP_prompts, BL_prompts, torch.empty(x.shape[1], 3*(2*7), 154)]     # 文本与提示组成输入
        outputs = self.transformer(combined)
        x = outputs[0]                                              
        xy = outputs[7]
        x = x.permute(1, 0, 2)                                      # LND -> NLD
        x = self.ln_final(x).type(self.dtype)                   
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.Cross_textprojection 

        return x, xy


# 多模态提示
class MultiModalPromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        self.n_cls = 64   # batch size
        # self.dtype = clip_model.dtype
        self.dtype = torch.float32
        self.token_embed = clip_model.token_embedding
        
        # cfg_imsize = 512
        # clip_imsize = clip_model.visual.input_resolution
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        n_ctx = 16 
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.ctx_text = nn.Parameter(torch.empty(2*MAPLE_LENGTH+2, 512, dtype=self.dtype))
        nn.init.normal_(self.ctx_text, std=0.02)
        self.ctx_img = nn.Parameter(torch.empty(2*MAPLE_LENGTH2+2, 768, dtype=self.dtype))
        nn.init.normal_(self.ctx_img, std=0.02)                   
        
        print('Multi-modal Prompt Learning')
        print(f"Number of MaPLe context words (tokens): {n_ctx+2}")
        
        self.proj_JB = nn.Linear(ctx_dim, 768)

        # self.proj_JB.half()

        self.compound_prompts_depth = 5 
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))                    
                                                       for _ in range(self.compound_prompts_depth - 1)])
        
        self.compound_prompts_img = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))                  
                                                      for _ in range(self.compound_prompts_depth - 1)])
        
        self.JP_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(1*MAPLE_LENGTH, 768))                                  
                                                 for _ in range(self.compound_prompts_depth - 1)])
        
        self.BL_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(1*MAPLE_LENGTH, 768))                    
                                                 for _ in range(self.compound_prompts_depth - 1)])
        
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.compound_prompts_img:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.JP_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.BL_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        
        self.MSA_1 = MHAttention(768, 768)
        self.MSA_2 = MHAttention(768, 768)
        self.MSA_3 = MHAttention(768, 768)
        self.MSA_4 = MHAttention(768, 768)

    def forward(self, batch):
        ctx_text = self.ctx_text #4* 可学习向量
        if ctx_text.dim() == 2:
            ctx_text = ctx_text.unsqueeze(0).expand(self.n_cls, -1, -1)
        ctx_img = self.ctx_img

        texts = self.token_embed().type(self.dtype) 
        prefix = texts[:, :1, :]
        suffix = texts[:, 1+4*MAPLE_LENGTH:77, :]
        texts = torch.cat([prefix, ctx_text, suffix], dim=1) 
        
        texts_JP = self.token_embed(batch["tokens_JP"]).type(self.dtype)  #token embedding 
        texts_BL = self.token_embed(batch["tokens_BL"]).type(self.dtype)
        
        imgs_JP = self.proj_JB(texts_JP) 
        # ipdb.set_trace()
        # 复制MAPLE_LENGTH2个序列
        # imgs_JP = imgs_JP.repeat(1, MAPLE_LENGTH2, 1)
        imgs_BL = self.proj_JB(texts_BL)
        
        textual_prompts = []    
        visual_prompts = []     
        
        joint_in_1 = torch.cat([self.compound_prompts_text[0], self.compound_prompts_img[0]], dim=0).unsqueeze(0)
        joint_out_1 = self.MSA_1(joint_in_1)
        joint_out_1 = joint_out_1.squeeze(0)
        textual_prompts.append(joint_out_1[0:2*MAPLE_LENGTH, 0:512])
        visual_prompts.append(joint_out_1[16:16+2*MAPLE_LENGTH2, :])
        
        joint_in_2 = torch.cat([self.compound_prompts_text[1], self.compound_prompts_img[1]], dim=0).unsqueeze(0)
        joint_out_2 = self.MSA_2(joint_in_2)
        joint_out_2 = joint_out_2.squeeze(0)
        textual_prompts.append(joint_out_2[0:2*MAPLE_LENGTH, 0:512])
        visual_prompts.append(joint_out_2[16:16+2*MAPLE_LENGTH2, :])
        
        joint_in_3 = torch.cat([self.compound_prompts_text[2], self.compound_prompts_img[2]], dim=0).unsqueeze(0)
        joint_out_3 = self.MSA_3(joint_in_3)
        joint_out_3 = joint_out_3.squeeze(0)
        textual_prompts.append(joint_out_3[0:2*MAPLE_LENGTH, 0:512])
        visual_prompts.append(joint_out_3[16:16+2*MAPLE_LENGTH2, :])
        
        joint_in_4 = torch.cat([self.compound_prompts_text[3], self.compound_prompts_img[3]], dim=0).unsqueeze(0)
        joint_out_4 = self.MSA_4(joint_in_4)
        joint_out_4 = joint_out_4.squeeze(0)
        textual_prompts.append(joint_out_4[0:2*MAPLE_LENGTH, 0:512])
        visual_prompts.append(joint_out_4[16:16+2*MAPLE_LENGTH2, :])
        
        return texts, texts_JP, texts_BL, imgs_JP, imgs_BL, ctx_img, self.JP_prompts_text, self.BL_prompts_text, textual_prompts, visual_prompts

class MGCA(LightningModule):
    '''Pytorch lightning implementation of MGCA'''
    def __init__(self,              
                 emb_dim: int = 128,
                 learning_rate: float = 2e-5,
                 momentum: float = 0.9,
                 weight_decay: float = 0.05,
                 batch_size: int = 4,
                 num_workers: int = 0,
                 softmax_temperature: float = 0.07,
                 *args,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # init encoders
        self.clip_model = load_clip_to_cpu() 
        
        self.image_encoder = self.clip_model.visual 
        self.text_encoder = TextEncoder(self.clip_model)

        self.prompt_learner = MultiModalPromptLearner(self.clip_model)

        # 添加滑动平均的缓冲区
        self.register_buffer('acc30_buffer', torch.zeros(10))
        self.register_buffer('acc40_buffer', torch.zeros(10))
        self.register_buffer('acc50_buffer', torch.zeros(10))
        self.register_buffer('buffer_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))

    def forward(self, batch, batch_idx, split="train"):
        '''Forward step of our method'''
        tokenized_texts = batch["tokens_77"]
        images = batch["imgs"]
        # images = batch["imgs"].half()
        texts, texts_JP, texts_BL, imgs_JP, imgs_BL, shared_ctx, JP_prompts_text, BL_prompts_text, deep_prompts_text, deep_prompts_vision = self.prompt_learner(batch)
        
        image_features, image_p = self.image_encoder(images, shared_ctx, JP_prompts_text, BL_prompts_text, deep_prompts_vision, imgs_JP, imgs_BL)
        text_features, text_p = self.text_encoder(texts, tokenized_texts, JP_prompts_text, BL_prompts_text, deep_prompts_text, texts_JP, texts_BL)
        # ipdb.set_trace()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()
        
        scores1 = logits/self.hparams.softmax_temperature
        scores2 = scores1.transpose(0, 1)
        
        bz = image_features.size(0)
        labels = torch.arange(bz).type_as(image_features).long()
        # i2t_acc1, i2t_acc5, i2t_acc20 = self.precision_at_k(scores1, labels, top_k=(30, 40,50))
        # t2i_acc1, t2i_acc5, t2i_acc20 = self.precision_at_k(scores2, labels, top_k=(30, 40,50))
        i2t_acc30, i2t_acc40, i2t_acc50 = self.precision_at_k(scores1, labels, top_k=(30, 40,50))
        t2i_acc30, t2i_acc40, t2i_acc50 = self.precision_at_k(scores2, labels, top_k=(30, 40,50))
        acc30 = (i2t_acc30 + t2i_acc30) / 2.
        acc40 = (i2t_acc40 + t2i_acc40) / 2.
        acc50 = (i2t_acc50 + t2i_acc50) / 2.
        loss1 = F.cross_entropy(scores1, labels)
        loss2 = F.cross_entropy(scores2, labels)
       
        loss1_p = F.l1_loss(image_p, text_p)
        loss2_p = F.l1_loss(text_p, image_p)
        # print('image_p:', image_p)
        # print('image_p.shape:', image_p.shape)
        # print('text_p:', text_p)
        # print('text_p.shape:', text_p.shape)
        # # 检查是否有NaN值
        # print('image_p是否包含NaN:', torch.isnan(image_p).any())
        # print('text_p是否包含NaN:', torch.isnan(text_p).any())
        # print("loss1_p:", loss1_p)
        # print("loss2_p:", loss2_p)
        loss_p = loss1_p + loss2_p
        # print("loss_p:", loss_p)
        
        loss = loss1/2 + loss2/2
        
        return loss, loss_p, acc30, acc40, acc50


    def training_step(self, batch, batch_idx):
        loss, loss_p, acc30, acc40, acc50 = self(batch, batch_idx, "train")
        loss_total = loss + loss_p
        
        # 更新滑动平均缓冲区
        idx = int(self.buffer_idx.item())
        self.acc30_buffer[idx] = acc30
        self.acc40_buffer[idx] = acc40
        self.acc50_buffer[idx] = acc50
        self.buffer_idx[0] = (idx + 1) % 10
        self.step_counter[0] += 1
        
        # 每10步才更新一次准确率
        if self.step_counter[0] % 10 == 0:
            avg_acc30 = self.acc30_buffer.mean()
            avg_acc40 = self.acc40_buffer.mean()
            avg_acc50 = self.acc50_buffer.mean()
            
            log = {
                "train_loss": loss, 
                "train_loss_p": loss_p, 
                "train_acc30": avg_acc30, 
                "train_acc40": avg_acc40, 
                "train_acc50": avg_acc50
            }
            self.log_dict(log, sync_dist=True, prog_bar=True)
            
            if batch_idx % 100 == 0:
                print('Train Batch: [%d] | Loss: %.4f | Loss_F: %.4f | Loss_P: %.4f | ACC-30: %.4f | ACC-40: %.4f | ACC-50: %.4f' % 
                      (batch_idx, loss_total, loss, loss_p, avg_acc30, avg_acc40, avg_acc50))
        else:
            # 非更新步只记录损失
            log = {
                "train_loss": loss, 
                "train_loss_p": loss_p
            }
            self.log_dict(log, sync_dist=True, prog_bar=True)
        
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        # loss, acc1, acc5 = self(batch, batch_idx, "valid")
        loss, loss_p, acc30, acc40, acc50 = self(batch, batch_idx, "valid")
        # loss, loss_p, acc1, acc5, acc20 = self(batch, batch_idx, "valid")
        loss_total = loss + loss_p
        log = {"val_loss": loss, "val_acc30": acc30, "val_acc40": acc40, "val_acc50": acc50}
        self.log_dict(log, sync_dist=True, prog_bar=True)
        print('Val Loss: %.4f | Loss_F: %.4f | Loss_P: %.4f | ACC-30: %.4f | ACC-40: %.4f | ACC-50: %.4f' % (loss_total, loss, loss_p, acc30, acc40, acc50))
        return loss_total


    @staticmethod
    def precision_at_k(output: torch.Tensor, target: torch.Tensor, top_k=(1,)):
        ''' Compute the accuracy over the k top predictions for the specified values of k'''
        with torch.no_grad():
            maxk = max(top_k)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in top_k:
                correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     self.hparams.learning_rate,
        #     betas=(self.hparams.momentum, 0.999),
        #     weight_decay=self.hparams.weight_decay)
        
        optimizer = torch.optim.SGD(
            self.parameters(),
            self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay)
        
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=self.training_steps,
            cycle_mult=1.0,
            max_lr=self.hparams.learning_rate,
            min_lr=5e-6,                                 
            warmup_steps=int(self.training_steps * 0.1)) 
        
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--emb_dim", type=int, default=128, help="128, 256")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument("--learning_rate", type=float, default=0.0005) 
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument("--weight_decay", type=float, default=0.05)  
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--softmax_temperature", type=float, default=0.01)
        parser.add_argument("--experiment_name", type=str, default="")
        parser.add_argument("--seed", type=int, default=101)
        parser.add_argument("--data_pct", type=float, default=1.)
        return parser

    @staticmethod
    def _use_ddp_or_dpp2(trainer: Trainer) -> bool:
        if trainer:
            # return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))
            return isinstance(trainer.training_type_plugin, strategy="ddp") 
        else:
            return torch.distributed.is_initialized()

    @staticmethod
    def num_training_steps(trainer, dm) -> int:
        """Total training steps inferred from datamodule and devices."""
        dataset = dm.train_dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, trainer.num_devices)
        
        # if trainer.tpu_cores:
            # num_devices = max(num_devices, trainer.tpu_cores)
        effective_batch_size = trainer.accumulate_grad_batches * num_devices

        return (dataset_size // effective_batch_size) * trainer.max_epochs


@torch.no_grad()
def concat_all_gather(tensor):
    '''Performs all_gather operation on the provided tensors'''
    tensors_gather = [torch.ones_like(tensor) for _ in range(
        torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


'''主函数'''
def cli_main():

    parser = ArgumentParser()
    # model args
    parser = MGCA.add_model_specific_args(parser)
    args = parser.parse_args()

    args.deterministic = True
    args.max_epochs = 10 # 10

    # seed
    seed_everything(args.seed)
    # ipdb.set_trace()
    datamodule = DataModule(MultimodalPretrainingDataset, multimodal_collate_fn, DataTransforms, args.data_pct, args.batch_size, args.num_workers)
    # ipdb.set_trace()
    # Add load from checkpoint
    model = MGCA(**args.__dict__)
    # 打印模型参数信息
    # print("模型参数统计:")
    # total_params = 0
    # trainable_params = 0
    # frozen_params = 0
    # pretrained_params = 0
    
    # # 检查模型中的参数
    # for name, param in model.named_parameters():
    #     param_count = param.numel()
    #     total_params += param_count
        
    #     if param.requires_grad:
    #         trainable_params += param_count
    #         print(f"可训练参数: {name}, 形状: {param.shape}, 参数量: {param_count}")
    #     else:
    #         frozen_params += param_count
    #         print(f"冻结参数: {name}, 形状: {param.shape}, 参数量: {param_count}")
    
    # # 检查是否有从checkpoint加载的参数
    # if hasattr(model, 'loaded_from_checkpoint') and model.loaded_from_checkpoint:
    #     for name in model.loaded_from_checkpoint:
    #         param = dict(model.named_parameters())[name]
    #         pretrained_params += param.numel()
    #         print(f"从checkpoint加载的参数: {name}, 形状: {param.shape}")
    
    # print(f"\n总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    # print(f"冻结参数量: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    
    # if hasattr(model, 'loaded_from_checkpoint') and model.loaded_from_checkpoint:
    #     print(f"预训练参数量: {pretrained_params:,} ({pretrained_params/total_params*100:.2f}%)")
    # ipdb.set_trace()
    # get current time
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    ckpt_dir = os.path.join(BASE_DIR, f"D:/124/MGCA-main/model/ckpts/MGCA/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [LearningRateMonitor(logging_interval="step"),
                 ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, save_last=True, mode="min", save_top_k=5),
                 EarlyStopping(monitor="val_loss", min_delta=0.,patience=5, verbose=False, mode="min") ]
    
    logger_dir = os.path.join(BASE_DIR, f"D:/124/MGCA-main/model")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(project="MGCA", save_dir=logger_dir, name=extension)
    
    # 直接在Trainer构造函数中传递参数，而不是使用add_argparse_args
    trainer = Trainer(
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=args.deterministic,
        # 添加其他需要的参数
    )
    
    model.training_steps = model.num_training_steps(trainer, datamodule)
    print(model.training_steps)
    
    trainer.fit(model, datamodule=datamodule)

    best_ckpt_path = os.path.join(ckpt_dir, "best_ckpts.yaml")
    callbacks[1].to_yaml(filepath=best_ckpt_path)


if __name__ == "__main__":
    cli_main()


