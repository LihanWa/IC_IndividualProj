import os
import math
import datetime
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from dateutil import tz
from sklearn.metrics import f1_score, average_precision_score

# ----------------- 环境设置 -----------------
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- 数据增强 ----------------
class MedicalImageTransforms(object):
    def __init__(self, is_train: bool = True, crop_size: int = 224):
        if is_train:
            self.data_transforms = transforms.Compose([
                transforms.Resize((crop_size + 32, crop_size + 32)),
                transforms.RandomCrop(crop_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.data_transforms = transforms.Compose([
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __call__(self, image):
        return self.data_transforms(image)


# ---------------- 模型定义 ----------------
class ClassificationMGCA(LightningModule):
    def __init__(self,              
                 learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4,
                 batch_size: int = 16,
                 num_workers: int = 0,
                 num_classes: int = 121,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # backbone
        self.backbone = models.resnet50(pretrained=True)
        self.in_features = self.backbone.fc.in_features
        for name, param in self.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
        self.backbone.fc = nn.Identity()

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        # 损失 & 权重
        self.criterion = nn.BCEWithLogitsLoss()
        self.pos_weight = None
        self.dynamic_thresholds = None
        self.label_smoothing = 0.05
        self.train_pbar = None  # 初始化进度条变量

    def on_train_epoch_start(self):
        # 每个 epoch 开始时新建一个 tqdm
        self.train_pbar = tqdm(total=len(self.trainer.datamodule.train_dataloader()), 
                               desc=f"Epoch {self.current_epoch}", 
                               position=0, leave=True)


    def on_train_epoch_end(self):
        if self.train_pbar is not None:
            self.train_pbar.close()
    # -------- 类别权重计算 ----------
    def setup(self, stage=None):
        if stage == "fit":
            all_labels = []
            for i, batch in enumerate(self.trainer.datamodule.train_dataloader()):
                _, labels = batch["imgs"], batch["labels"] if isinstance(batch, dict) else batch
                all_labels.append(labels)
                if i > 50:  # 只采样前 50 个 batch
                    break
            labels = torch.cat(all_labels, dim=0)

            pos_counts = labels.sum(dim=0).float()
            total_samples = labels.shape[0]
            pos_rates = pos_counts / total_samples
            neg_counts = total_samples - pos_counts
            pos_weight = neg_counts / (pos_counts + 1e-8)
            pos_weight = torch.clamp(pos_weight, min=1.0, max=15.0)

            self.pos_weight = pos_weight.to(self.device)
            self.dynamic_thresholds = 0.3 + 0.4 * pos_rates.to(self.device)
            self.dynamic_thresholds = torch.clamp(self.dynamic_thresholds, min=0.2, max=0.7)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

            print(f"[Class Weights Ready] range={self.pos_weight.min():.2f}-{self.pos_weight.max():.2f}")
            print(f"[Dynamic Thresholds] range={self.dynamic_thresholds.min():.2f}-{self.dynamic_thresholds.max():.2f}")

    # -------- 标签平滑 ----------
    def smooth_labels(self, labels, smoothing=0.05):
        with torch.no_grad():
            smoothed_labels = labels * (1 - smoothing) + 0.5 * smoothing
        return smoothed_labels

    # -------- forward ----------
    def forward(self, batch):
        images = batch["imgs"]
        features = self.backbone(images)
        logits = self.classifier(features)
        return logits

    # -------- 公共 step ----------
    def _step(self, batch, stage="train"):
        logits = self.forward(batch)

        # 原始标签 (0/1)，用于 metric
        labels = batch["labels"]
        labels_int = labels.int()  # 保证是 0/1 int

        # 训练阶段 → label smoothing
        if stage == "train":
            labels_for_loss = self.smooth_labels(labels.float(), self.label_smoothing)
        else:
            labels_for_loss = labels.float()

        # Loss
        loss = self.criterion(logits, labels_for_loss)

        # 预测概率
        preds = torch.sigmoid(logits)

        # 二值化预测
        thresholds = self.dynamic_thresholds.unsqueeze(0) if self.dynamic_thresholds is not None else 0.5
        binary_preds = (preds > thresholds).int()

        # F1 & mAP
        labels_np = labels_int.cpu().numpy()                      # 保证 0/1
        binary_preds_np = binary_preds.cpu().numpy().astype(int)  # 保证 0/1
        preds_np = preds.detach().cpu().numpy()

        f1_macro = f1_score(labels_np, binary_preds_np, average="macro", zero_division=0)
        f1_micro = f1_score(labels_np, binary_preds_np, average="micro", zero_division=0)
        mAP = average_precision_score(labels_np, preds_np, average="macro")
        per_class_acc = (binary_preds_np == labels_np).mean(axis=0)   # 每个类的accuracy
        macro_acc = per_class_acc.mean()
        # Logging
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_f1_macro', f1_macro, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_f1_micro', f1_micro, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_mAP', mAP, on_epoch=True, prog_bar=True)
        self.log(f'{stage}_macro_acc', macro_acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # -------- 优化器 ----------
    def configure_optimizers(self):
        backbone_params, classifier_params = [], []
        for name, param in self.named_parameters():
            if 'backbone' in name and param.requires_grad:
                backbone_params.append(param)
            else:
                classifier_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hparams.learning_rate * 0.1, 'weight_decay': 1e-3},
            {'params': classifier_params, 'lr': self.hparams.learning_rate, 'weight_decay': 1e-4}
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# # ---------------- 主函数 ----------------
# def cli_main():
#     parser = ArgumentParser()
#     parser.add_argument("--dataset", type=str, default="mimiccxr")
#     parser.add_argument("--data_pct", type=float, default=1.0)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--learning_rate", type=float, default=1e-3)
#     parser.add_argument("--weight_decay", type=float, default=1e-4)
#     parser.add_argument("--num_classes", type=int, default=121)
#     parser.add_argument("--max_epochs", type=int, default=50)
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--accelerator", type=str, default="gpu")
#     parser.add_argument("--devices", type=int, default=1)
#     args = parser.parse_args()

#     seed_everything(args.seed)
#     torch.set_float32_matmul_precision('medium')

#     # 加载数据模块
#     from mgca.datasets.data_module import DataModule
#     # from mgca.datasets.pretrain_dataset import (MultimodalPretrainingDataset, multimodal_collate_fn)

#     if args.dataset == "mimiccxr":
#         from mgca.datasets.classification_dataset import MIMICImageDataset
#         datamodule = DataModule(MIMICImageDataset, None, MedicalImageTransforms,
#                                 args.data_pct, args.batch_size, args.num_workers)
#         args.num_classes = 119
#     # elif args.dataset == "chexpert":
#     #     from mgca.datasets.classification_dataset import CheXpertImageDataset
#     #     datamodule = DataModule(CheXpertImageDataset, None, MedicalImageTransforms,
#     #                             args.data_pct, args.batch_size, args.num_workers)
#     #     args.num_classes = 14
#     # elif args.dataset == "rsna":
#     #     from mgca.datasets.classification_dataset import RSNAImageDataset
#     #     datamodule = DataModule(RSNAImageDataset, None, MedicalImageTransforms,
#     #                             args.data_pct, args.batch_size, args.num_workers)
#     #     args.num_classes = 2
#     # elif args.dataset == "covidx":
#     #     from mgca.datasets.classification_dataset import COVIDXImageDataset
#     #     datamodule = DataModule(COVIDXImageDataset, None, MedicalImageTransforms,
#     #                             args.data_pct, args.batch_size, args.num_workers)
#     #     args.num_classes = 2
#     # else:
#     #     datamodule = DataModule(
#     #         MultimodalPretrainingDataset,
#     #         multimodal_collate_fn,
#     #         MedicalImageTransforms,
#     #         args.data_pct,
#     #         args.batch_size,
#     #         args.num_workers
#     #     )

#     # 创建模型
#     model = ClassificationMGCA(
#         learning_rate=args.learning_rate,
#         weight_decay=args.weight_decay,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         num_classes=args.num_classes
#     )

#     # 保存路径
#     now = datetime.datetime.now(tz.tzlocal())
#     extension = now.strftime("%Y_%m_%d_%H_%M_%S")
#     ckpt_dir = os.path.join(BASE_DIR, f"checkpoints/classification/{extension}")
#     os.makedirs(ckpt_dir, exist_ok=True)

#     # 回调函数
#     callbacks = [
#         LearningRateMonitor(logging_interval="step"),
#         ModelCheckpoint(
#             monitor="val_mAP",  # ⚠️ 改成 mAP 作为保存依据
#             dirpath=ckpt_dir,
#             save_last=True,
#             mode="max",
#             save_top_k=3,
#             filename="best-{epoch:02d}-{val_mAP:.3f}"
#         ),
#         EarlyStopping(
#             monitor="val_mAP",
#             min_delta=0.001,
#             patience=10,
#             verbose=True,
#             mode="max"
#         )
#     ]

#     # Wandb 日志
#     logger_dir = os.path.join(BASE_DIR, "logs")
#     os.makedirs(logger_dir, exist_ok=True)
#     wandb_logger = WandbLogger(
#         project="MCPL-Classification-ResNet",
#         entity="lihanw",
#         save_dir=logger_dir,
#         name=f"classification_resnet_{extension}"
#     )

#     # Trainer
#     trainer = Trainer(
#         max_epochs=args.max_epochs,
#         accelerator=args.accelerator,
#         devices=args.devices,
#         callbacks=callbacks,
#         logger=wandb_logger,
#         precision=16,
#         accumulate_grad_batches=2,
#         gradient_clip_val=1.0,
#         check_val_every_n_epoch=1,
#         log_every_n_steps=10,
#         enable_progress_bar=True,
#         enable_model_summary=True,
#         enable_checkpointing=True
#     )

#     # 训练 & 测试
#     trainer.fit(model, datamodule=datamodule,ckpt_path="/rds/general/user/lw1824/home/MCPL/MCPL/checkpoints/classification/2025_09_04_17_37_06/last.ckpt")
#     trainer.test(model, datamodule=datamodule, ckpt_path="best")


# if __name__ == "__main__":
#     cli_main()
