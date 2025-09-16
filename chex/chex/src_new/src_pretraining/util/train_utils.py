
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
import dataclasses
import glob
import json
import logging
import os
import random
from typing import Any, Callable, Dict, Generic, Iterable, List, Mapping, Optional, Sequence, Collection, TypeVar, Union
import warnings
import numpy as np
from deepspeed.runtime.lr_schedules import WarmupCosineLR
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR

from timm.scheduler import CosineLRScheduler
from torch import optim
from omegaconf import MISSING, OmegaConf
import torch
from dataset.image_transform import TransformConfig
from deepspeed.ops.adam import DeepSpeedCPUAdam 
from dataset.datasets import DatasetConfig, build_dataloader
from metrics.bootstrapping import BootstrapMetricsWrapper
from util.model_utils import prepare_config
import ipdb


log = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    task: Optional[str] = None
    dataset: Dict[str, DatasetConfig] = MISSING

    plot_wandb: bool = True
    plot_local: bool = False
    plot_val_samples: int = 0
    plot_val_arguments: Dict[str, Any] = field(default_factory=dict)


def get_task_dataset(config: EvalConfig) -> DatasetConfig:
    return list(config.dataset.values())[0]

@dataclass
class ExperimentConfig:
    name: str = MISSING
    seed: int = MISSING
    use_deepspeed: bool = MISSING
    model: Any = MISSING

    continue_from_checkpoint: Optional[str] = None
    load_modules_from_checkpoint: Optional[List[str]] = None

    train_dataset: Dict[str, DatasetConfig] = field(default_factory=dict)
    val_tasks: Dict[str, Any] = field(default_factory=dict)  # Dict[str, EvalConfig]
    transform: TransformConfig = MISSING
    train: bool = True
    evaluate: bool = True
    eval_mode: str = 'val'
    eval_tasks: Dict[str, Any] = field(default_factory=dict)

    batch_size: int = MISSING
    max_steps: Optional[int] = None
    max_epochs: Optional[int] = None
    lr: float = MISSING
    min_lr: float = MISSING
    warmup_lr: Optional[float] = MISSING
    warmup_steps: int = MISSING
    weight_decay: float = MISSING
    accumulation_steps: int = MISSING
    grad_clip_norm: Optional[float] = MISSING
    early_sopping_patience: Optional[int] = None

    metric: Optional[str] = None
    metric_mode: str = 'max'

    val_freq: Optional[int] = None
    val_ep: Optional[int] = None
    print_freq: int = MISSING
    num_workers: int = MISSING
    prefetch: bool = MISSING
    device: str = MISSING
    debug: bool = False
    compile: bool = True
    save_components: List[str] = field(default_factory=list)
    amp: bool = True


class Evaluator:
    def __init__(self, config: EvalConfig, config_cls, bootstrap: bool = False, n_bootstrap: int = 250, results_path: Optional[str] = None):
        self.config = prepare_config(config, config_cls, log)
        self.task = config.task
        self.dataset = get_task_dataset(self.config)
        self.bootstrap = bootstrap
        self.n_bootstrap = n_bootstrap
        self.results_path = results_path
        self.metrics = {}

    @abstractmethod
    def _predict(self, **kwargs) -> tuple:
        raise NotImplementedError
    
    @abstractmethod
    def _postprocess(self, *predictions: tuple, config) -> 'BaseModelOutput':
        raise NotImplementedError

    @abstractmethod
    def _update_metrics_with_output(self, output):
        raise NotImplementedError

    @abstractmethod
    def plot(self, **kwargs):
        raise NotImplementedError

    def optimize_inference(self, predictions: List[tuple]) -> EvalConfig:
        warnings.warn('optimize_inference is not implemented for this model')
        return self.config


    def eval_step(self, optimize_inference=False, **kwargs) -> Union['BaseModelOutput', tuple]:
        # ipdb.set_trace()
        predictions = self._predict(**kwargs)
        
        if optimize_inference:
            return predictions
        
        output = self._postprocess(*predictions, config=self.config)
        self._update_metrics_with_output(output)
        return output


    def _register_metric(self, metric, metric_name: Optional[str] = None):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name not in self.metrics
        if self.bootstrap:
            metric =  BootstrapMetricsWrapper(
                metric, 
                n_bootstrap=self.n_bootstrap,
                csv_path=f'{self.results_path}_bootstrap.csv' if self.results_path is not None else None)

        self.metrics[metric_name] = metric

    def _get_metric(self, metric_name: Optional[str] = None):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name in self.metrics
        metric = self.metrics[metric_name]
        if isinstance(metric, BootstrapMetricsWrapper):
            metric = metric.metrics
        return metric

    def _update_metric(self, metric_name: Optional[str] = None, **kwargs):
        if metric_name is None:
            metric_name = 'metrics'
        assert metric_name in self.metrics
        self.metrics[metric_name].update(**kwargs)

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def _compute_metrics(self) -> dict:
        if len(self.metrics) == 1:
            return list(self.metrics.values())[0].compute() #TrainingMetrics in training_metrics.py
        else:
            return {f'{metric_name}/{key}': value for metric_name, metric in self.metrics.items() for key, value in metric.compute().items()}

    def compute_metrics(self) -> dict:
        results = self._compute_metrics()
        if self.results_path is not None:
            json_path = f'{self.results_path}.json'
            json_results = {key: float(value) for key, value in results.items()}
            with open(json_path, 'w') as f:
                json.dump(json_results, f)

        return results
    

def build_optimizer(model: 'BaseModel', config: ExperimentConfig):
    if config.use_deepspeed:
        return DeepSpeedCPUAdam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
            # adamw_mode=True, # 可能需要设置 AdamW 模式，确认文档
        )
    else:
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                       weight_decay=config.weight_decay)


def build_scheduler(optimizer, config: ExperimentConfig,ckpt=None):
    num_steps = int(config.max_steps)
    warmup_steps = int(config.warmup_steps)
    log.info(f'num_steps:{num_steps}')
    log.info(f'warmup_steps:{warmup_steps}')
    if ckpt:
        lr_scheduler_config=ckpt['lr_scheduler']
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=lr_scheduler_config['t_initial'],
            lr_min=lr_scheduler_config['lr_min'],
            warmup_t=lr_scheduler_config['warmup_t'],
            warmup_lr_init=lr_scheduler_config['warmup_lr_init'],
            cycle_mul=lr_scheduler_config['cycle_mul'],
            cycle_decay=lr_scheduler_config['cycle_decay'],
            cycle_limit=lr_scheduler_config['cycle_limit'],
            t_in_epochs=lr_scheduler_config['t_in_epochs'],
            noise_range_t=lr_scheduler_config['noise_range_t'],
            noise_pct=lr_scheduler_config['noise_pct'],
            noise_std=lr_scheduler_config['noise_std'],
            noise_seed=lr_scheduler_config['noise_seed']
        )
        
        # 如果检查点中有步数信息，恢复学习率调度器的状态
        if 'step' in ckpt:
            for _ in range(int(ckpt['step'])):
                lr_scheduler.step_update(_)
            log.info(f"学习率调度器已恢复到步骤 {ckpt['step']}")
        return lr_scheduler
    
    else:

            # 获取当前优化器中的学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 创建调度器时，确保设置了正确的基础值
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=config.min_lr,
            warmup_lr_init=config.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
        
        # 确保调度器使用当前学习率作为基础值
        scheduler.base_values = [current_lr for _ in scheduler.base_values]
        
        return scheduler
        # cosine_scheduler = CosineLRScheduler(
    # optimizer,
    # t_initial=num_steps,
    # lr_min=config.min_lr,
    # warmup_lr_init=config.warmup_lr,
    # warmup_t=warmup_steps,
    # cycle_limit=1,
    # t_in_epochs=False,
    # )

    # # 包装为LambdaLR
    # def lr_lambda(step):
    #     cosine_scheduler.step(step)
    #     # CosineLRScheduler 会直接更新 optimizer.param_groups LR
    #     return optimizer.param_groups[0]['lr'] / config.lr

    # wrapped_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # return wrapped_scheduler
    # # warmup_scheduler = LinearLR(optimizer, start_factor= 0.01, total_iters=warmup_steps)
    # # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_steps - warmup_steps, eta_min=config.min_lr)

    # # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    # return scheduler
    # if not config.use_deepspeed:
    
    # else:
    #     return WarmupCosineLR(
    #         optimizer,
    #         num_steps,
    #         warmup_num_steps=warmup_steps,

    #     )

def build_train_dataloader(config: ExperimentConfig, use_image_paths: bool = False, pretrain: bool = False):
    assert len(config.train_dataset) >= 1, config.train_dataset
    train_datasets = list(config.train_dataset.values())
    print('='*50)
    print('train_datasets', config.train_dataset.keys())

    print('='*50)
    primary_dataset = train_datasets[0]
    train_dl = build_dataloader(
        mode='train',
        configs=train_datasets,
        pixel_mean=primary_dataset.pixel_mean,
        pixel_std=primary_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch,
        is_train=True,
        use_image_paths=use_image_paths,
        pretrain=pretrain)
    # for a in train_dl:
    #     print(a)
        
    # ipdb.set_trace()
    return train_dl

def build_val_dataloader(config: ExperimentConfig, val_task: EvalConfig, use_image_paths: bool = False, pretrain: bool = False):
    
    primary_train_dataset = list(config.train_dataset.values())[0]
    assert len(val_task.dataset) == 1, val_task.dataset
    val_dataset = list(val_task.dataset.values())[0]
    val_dl = build_dataloader(
        mode='val',
        configs=[val_dataset],
        pixel_mean=primary_train_dataset.pixel_mean,
        pixel_std=primary_train_dataset.pixel_std,
        transform=config.transform,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prefetch=config.prefetch,
        is_train=False,
        use_image_paths=use_image_paths,
        pretrain=pretrain)
    return val_dl


def build_dataloaders_for_eval(config: ExperimentConfig, eval_tasks=None, eval_mode=None, load_val=False,use_image_paths:bool=False,pretrain:bool=False):
    if eval_tasks is None:
        eval_tasks = config.eval_tasks
    if eval_mode is None:
        eval_mode = config.eval_mode
    assert len(config.train_dataset) >= 1
    parimary_train_dataset = list(config.train_dataset.values())[0]
    

    for i,(name, task) in enumerate(eval_tasks.items()):
        # ipdb.set_trace()
        # if i==0: continue
        assert len(task.dataset) == 1
        eval_dataset = list(task.dataset.values())[0]
        
        dataloader = build_dataloader(
            mode=eval_mode,
            configs=[eval_dataset],
            pixel_mean=parimary_train_dataset.pixel_mean,
            pixel_std=parimary_train_dataset.pixel_std,
            transform=config.transform,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch=config.prefetch,
            is_train=False,
            use_image_paths=use_image_paths,
            pretrain=pretrain)
        dataloader_val = None
        if load_val:
            dataloader_val = build_dataloader(
                mode='val',
                configs=[eval_dataset],
                pixel_mean=parimary_train_dataset.pixel_mean,
                pixel_std=parimary_train_dataset.pixel_std,
                transform=config.transform,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch=config.prefetch,
                is_train=False,
                use_image_paths=use_image_paths,
                pretrain=pretrain)
        yield name, task, dataloader, dataloader_val
        


def get_best_results(results, best_results, last_results, config: ExperimentConfig):
    if best_results is None:
        return results, results, True,True
    assert config.metric_mode in ('min', 'max')
    best_value = best_results['val_metric']
    value = results['val_metric']
    last_value = last_results['val_metric'] 
    # 比较当前结果与最佳结果
    is_better_than_best = (value > best_value and config.metric_mode == 'max') or \
                          (value < best_value and config.metric_mode == 'min')
    

    is_better_than_last = (value > last_value and config.metric_mode == 'max') or \
                            (value < last_value and config.metric_mode == 'min')
        
    if is_better_than_best:
        return results,results, True,is_better_than_last
    else:
        return best_results,results, False,is_better_than_last


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True

class AvgDictMeter:
    def __init__(self):
        self.values = defaultdict(float)
        self.n = 0

    def add(self, values: dict):
        for key, value in values.items():
            if value is None:
                continue
            self.values[key] += value
        self.n += 1

    def compute(self):
        return {key: value / self.n for key, value in self.values.items()}


def save_training_checkpoint(model: 'BaseModel', optimizer, lr_scheduler, results,
                             best_results, config, step, saved_components=(), is_best=False):
# def save_training_checkpoint(model: 'BaseModel', optimizer, lr_scheduler, scaler, results,
#                              best_results, config, step, saved_components=(), is_best=False):
    saved_components = () if saved_components is None else saved_components
    saved_states = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        # 'amp': scaler.state_dict(),
        'step': step,
        'results': results,
        'best_results': best_results,
        'experiment_config': OmegaConf.to_container(config)
    }

    # Save the current model
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{step:09d}.pth')

    model.save_model(checkpoint_path, **saved_states)
    for component_name in saved_components:
        os.makedirs(os.path.join('checkpoints', component_name), exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', component_name, f'checkpoint_{step:09d}.pth')
        model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Save as best model
    if is_best:
        checkpoint_path = os.path.join('checkpoints', f'checkpoint_best.pth')
        save_dir,tag = os.path.join("checkpoints"),'best'
        model.save_checkpoint(save_dir, tag=tag, client_state={'step': step})
        
        model.save_model(checkpoint_path, **saved_states)
        for component_name in saved_components:
            checkpoint_path = os.path.join('checkpoints', component_name, 'checkpoint_best.pth')
            model.save_model_component(checkpoint_path, component_name=component_name, **saved_states)

    # Remove the previous model
    if step > 0:
        for chkpt_path in glob.glob(os.path.join('checkpoints', f'checkpoint_*.pth')):
            if not chkpt_path.endswith(f'checkpoint_{step:09d}.pth') and not chkpt_path.endswith('checkpoint_best.pth'):
                os.remove(chkpt_path)

