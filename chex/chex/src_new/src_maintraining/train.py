from dataclasses import dataclass
from typing import Any, Dict, Optional
import hydra

from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass
import logging
import os
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from hydra.core.hydra_config import HydraConfig

from timeit import default_timer as timer
from deepspeed.ops.adam import DeepSpeedCPUAdam 
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import torch
from dataset.datasets import DatasetConfig
from settings import MODELS_DIR, PROJECT_DIR, WANDB_ENTITY, WANDB_PROJECT
from util.data_utils import to_device

from util.model_utils import BaseModel, BaseModelOutput, ModelRegistry, instantiate_model, load_model_by_name, load_model_from_checkpoint, load_model_and_optimizer_and_lr_scheduler_from_checkpoint
from util.train_utils import AvgDictMeter, EvalConfig, Evaluator, ExperimentConfig, build_dataloaders_for_eval, build_train_dataloader, build_optimizer, build_scheduler, build_val_dataloader, get_best_results, save_training_checkpoint, seed_everything
import ipdb
import lovely_tensors as lt
lt.monkey_patch()
import torch
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw, ImageFont
import gc
import warnings
import argparse
import deepspeed
import json
import inspect
import copy
log = logging.getLogger(__name__)


def log_gpu_memory(message):
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    log.info(f"{message}: Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB")

def print_memory_summary(message=""):
    log.info(f"\n=== Memory Summary {message} ===")
    log.info(torch.cuda.memory_summary(abbreviated=True))
    allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
    log.info(f"Allocated GPU memory: {allocated:.2f}GB")
    log.info(f"Cached GPU memory: {reserved:.2f}GB")
    log.info("="*50)

def train(config: ExperimentConfig, model: BaseModel=None, model_dir: str=None, full_model_name: str=None, use_image_paths: bool = False,pretrain: bool = False,optimizer=None,ckpt=None,lr_scheduler=None):
    # Load the datasets for training and validation
    # Check if model is DeepSpeed model
    is_deepspeed_model = '_orig_mod' in dir(model) or hasattr(model, 'module')
    log.info(f"Is model in DeepSpeed format: {is_deepspeed_model}")
    
    # If DeepSpeed model, print more information
    if is_deepspeed_model:
        if '_orig_mod' in dir(model):
            log.info("Model contains _orig_mod attribute")
        if hasattr(model, 'module'):
            log.info("Model contains module attribute")
            
        # Try to print some key information about model structure
        try:
            if hasattr(model, '_orig_mod'):
                log.info(f"_orig_mod type: {type(model._orig_mod).__name__}")
            if hasattr(model, 'module'):
                log.info(f"module type: {type(model.module).__name__}")
        except Exception as e:
            log.warning(f"Error trying to get DeepSpeed model structure info: {e}")
    if model is None:
        raise ValueError("model is None")
    train_dl = build_train_dataloader(config, use_image_paths, pretrain)
    train_data_conf = train_dl.dataset.dataset_info
    val_dls={}
    
    for i, (val_name, val_task) in enumerate(config.val_tasks.items()):
        val_dls[val_name] = build_val_dataloader(config, val_task, use_image_paths, pretrain)

    #Define variables: steps_per_epoch; max_steps; val_freq
    steps_per_epoch = len(train_dl)
    log.info(f'Note: {steps_per_epoch} steps per epoch')
    # assert config.max_steps is None or config.max_epochs is None
    #Can choose epoch or step
    if config.max_steps is None and config.max_epochs is not None: #can change to step
        config.max_steps = config.max_epochs * steps_per_epoch
    log.info(f'max_steps: {config.max_steps}')
    assert config.val_freq is not None or config.val_ep is not None
    if config.val_freq is None: 
        config.val_freq = steps_per_epoch * config.val_ep

    # Prepare optimizer, scheduler, and scaler
    if optimizer is None:
        optimizer = build_optimizer(model, config)

    model.to(config.device)
    log.info(f"Current config device: {config.device}")
    
    # ipdb.set_trace()
    use_amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if not ckpt and not optimizer:
        optimizer = build_optimizer(model, config)

        current_path = os.getcwd()
        log.info(f"当前工作路径: {current_path}")
    if not ckpt and not lr_scheduler:
        lr_scheduler = build_scheduler(optimizer, config)

    print('model type:',type(model))
    # scaler = GradScaler()
    if ckpt:
        step = int(ckpt['step'])
        log.info(f"Resuming step from checkpoint: {step}")
    else:
        # 不使用检查点时的初始化
        log.info("不使用检查点")
        step = 0
    epoch = 0
    step_metrics_meter = AvgDictMeter()

    # 在函数开始处添加记录总时间的变量
    train_start_time = timer()
    total_steps_done = 0

    """ Start training """
    log.info(f'Starting training of {full_model_name}')
    log.info(f"Using {config.device}")
    if config.debug:
        torch.autograd.set_detect_anomaly(True)
    best_results = None
    # 计算每个验证周期的步数
    val_steps = config.val_freq if config.val_freq is not None else steps_per_epoch * config.val_ep
    
    pbar = tqdm(
        total=val_steps, 
        desc='Training',
        unit='',  
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] [{postfix}]',
        dynamic_ncols=True,
        leave=False
    )
    
    def update_eta_total(pbar, step, val_steps):
        if step > 0:
            current_time = timer()
            total_elapsed = current_time - train_start_time
            # 使用总训练时间和总步数来计算
            time_per_step = total_elapsed / step
            
            total_time = time_per_step * config.max_steps
            remaining_total = time_per_step * (config.max_steps - step)
            
            # 格式化时间显示
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes}m"
            
            # 简化显示信息，减少长度
            pbar.set_postfix_str(
                f'l={step_metrics_meter.compute()["loss"]:.3f}, '
                f'GPU:{gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f}GB, '
                f'总:{format_time(total_time)}, '
                f'剩:{format_time(remaining_total)}'
            )

    # train 内部开始遍历training step
    while True:
        # print_memory_summary(f"Epoch {epoch} 开始")
        epoch_start_time = timer()
        it_start_time = timer()
        stop_training = False
        # 在训练开始前和每个epoch结束后添加
        for samples in train_dl:
            # 检查模型是否使用DeepSpeed
            # is_deepspeed = hasattr(model, 'module') or '_orig_mod' in dir(model)
            # if step<700:  # 只在第一步输出，避免重复日志
            #     if is_deepspeed:
            #         log.info(f"DeepSpeed模型类型: {type(model).__name__}")
            #         if hasattr(model, 'module'):
            #             log.info(f"DeepSpeed模块类型: {type(model.module).__name__}")
            #         if '_orig_mod' in dir(model):
            #             log.info(f"DeepSpeed原始模块类型: {type(model._orig_mod).__name__}")
            pbar.update(1)

            # Training step
            output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, scaler, epoch, step, config, train_data_conf)
            # output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, epoch, step, config, train_data_conf)
            step_metrics_meter.add(dict(output.step_metrics, loss=output.loss.detach())) #记录平均各个subloss以及总loss

            if torch.isnan(output.loss):
                log.error(f'Loss was nan: {output.step_metrics}')
                #stop_training = True
                #break
                
            # Increment step
            step += 1
            # log.info(f"当前step: {step}")
            # 每10步打印一次当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            # log.info(f"步骤 {step} - 当前学习率: {current_lr:.8f}")
            # 每40个steps清理一次GPU缓存
            if step % 25 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log.info(f"每40步清理GPU缓存: 已分配 {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, 已保留 {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")

            # Training Update progress bar and log losses
            if step % config.print_freq == 0 and not step % config.val_freq == 0:
                it_end_time = timer()
                it_time = it_end_time - it_start_time
                # it_start_time = it_end_time
                lr = optimizer.param_groups[0]['lr']
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                gc.collect()
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                
                update_eta_total(pbar, step, val_steps)
                
                
                # 记录当前GPU内存使用情况
                # log.info(f"GPU内存使用: 已分配 {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, 已保留 {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # 转换为GB
                # ipdb.set_trace()
                if not config.debug:
                    wandb.log({
                        'train_step/lr': lr,
                        'train_step/loss': output.loss.detach(),
                        'gpu/memory_allocated_gb': gpu_memory_allocated,
                        'gpu/memory_reserved_gb': gpu_memory_reserved,
                        **{'train_step/' + key: value for key, value in output.step_metrics.items()}
                    }, step=step)

            # Validate and log at validation frequency
            if config.debug or (step>int(config.warmup_steps) and (step >= config.max_steps or step % config.val_freq == 0 or (config.val_ep is not None and step % (config.val_ep * steps_per_epoch) == 0))):
                # Gather training results
                # 清理GPU缓存
                torch.cuda.empty_cache()
                gc.collect()
                train_results = step_metrics_meter.compute()
                step_metrics_meter = AvgDictMeter()

                # Validate
                results = {
                    'step': step,
                    **{'train/' + key: value for key, value in train_results.items()},
                }

                for i, (val_name, val_task) in enumerate(config.val_tasks.items()):
                    log.info(f'Validating {val_name} ({i+1}/{len(config.val_tasks)})...')
                    val_dl = val_dls[val_name]
                    log_gpu_memory(f"Before validation {val_name}")
                    val_results = validate(
                        config,
                        val_task,
                        val_dl,
                        model,
                        model_dir, 
                        step=step,
                        prefix=f'val/{val_name}',
                    )
                    log_gpu_memory(f"After validation {val_name}")
                    results.update(val_results)
                # ipdb.set_trace()
                if config.metric is None:
                    results['val_metric'] = 0.0
                    log.warning("No validation metric specified - using loss as metric")
                elif ',' in config.metric:
                    # 调和平均数计算，添加对小值的更稳健处理
                    val_metrics = [results[f'val/{m}'] for m in config.metric.split(',')]
                    vindr_metric=results['val/od_vindrcxr/AP/mAP']
                    nih_metric=results['val/od_nih8/AP/mAP']
                    mscxr_metric=results['val/od_mscxr/AP/mAP']
                    anat_cls_metric=results['val/train_metrics/l_anat/anat_cls']
                    

                    
                    # 打印各个指标的值，便于调试
                    log.info(f"vindr_metric: {vindr_metric}, nih_metric: {nih_metric}, mscxr_metric: {mscxr_metric}, anat_cls_metric: {anat_cls_metric}")
                    
                    # 计算自定义加权指标: sum(vindr*4+nih+ms-cxr)*64*2-anatcls*3
                    # custom_metric = ((vindr_metric + nih_metric + mscxr_metric*1.3) * 64 * 2) - (anat_cls_metric *0.5)
                    custom_metric = (vindr_metric  + nih_metric + mscxr_metric)/3
                    
                    # 将自定义指标添加到结果中
                    results['val_metric'] = custom_metric
                    log.info(f"自定义加权指标计算结果: {custom_metric}")
                    

                else:
                    results['val_metric'] = results[f'val/{config.metric}']
                

                if config.debug:
                    log.info('Debug mode -> Stopping training after 1 step')
                    return results

                best_results, is_best = get_best_results(results, best_results,
                                                         config)
                save_training_checkpoint(model, optimizer, lr_scheduler, scaler,
                                         results=results,
                                         best_results=best_results,
                                         config=config, step=step,
                                         saved_components=config.save_components,
                                         is_best=is_best)


                if not config.debug:
                    keys= [
                        'val/od_nih8/AP/classes/mAP_nih/Atelectasis',
                        'val/od_nih8/AP/classes/mAP_nih/Cardiomegaly',
                        'val/od_nih8/AP/classes/mAP_nih/Effusion',
                        'val/od_nih8/AP/classes/mAP_nih/Infiltration',
                        'val/od_nih8/AP/classes/mAP_nih/Mass',
                        'val/od_nih8/AP/classes/mAP_nih/Nodule',
                        'val/od_nih8/AP/classes/mAP_nih/Pneumonia',
                        'val/od_nih8/AP/classes/mAP_nih/Pneumothorax',
                        'val/od_nih8/AP/AP@0.1', 'val/od_nih8/AP/AP@0.3', 'val/od_nih8/AP/AP@0.5'
                        , 'val/od_mscxr/AP/classes/mAP_mscxr/Atelectasis', 'val/od_mscxr/AP/classes/mAP_mscxr/Cardiomegaly', 'val/od_mscxr/AP/classes/mAP_mscxr/Consolidation', 'val/od_mscxr/AP/classes/mAP_mscxr/Edema', 'val/od_mscxr/AP/classes/mAP_mscxr/Lung Opacity', 'val/od_mscxr/AP/classes/mAP_mscxr/Pleural Effusion', 'val/od_mscxr/AP/classes/mAP_mscxr/Pneumonia', 'val/od_mscxr/AP/classes/mAP_mscxr/Pneumothorax', 'val/od_mscxr/AP/AP@0.1', 'val/od_mscxr/AP/AP@0.3', 'val/od_mscxr/AP/AP@0.5'
                        , 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Aortic enlargement', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Atelectasis', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Cardiomegaly', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Calcification', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Consolidation', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/ILD', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Infiltration', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Lung Opacity', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Mediastinal shift', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Nodule/Mass', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Pulmonary fibrosis', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Pneumothorax', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Pleural thickening', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Pleural effusion', 'val/od_vindrcxr/AP/classes/mAP_vindrcxr/Other lesion', 'val/od_vindrcxr/AP/AP@0.1', 'val/od_vindrcxr/AP/AP@0.3', 'val/od_vindrcxr/AP/AP@0.5'
                    ]
                    
                    # 将这些不需要优先观察的指标后移
                    if results is not None:
                        # 创建一个新的有序字典来存储重新排序后的结果
                        reordered_results = {}
                        
                        # 首先添加不在keys列表中的指标
                        for k, v in results.items():
                            if k not in keys:
                                reordered_results[k] = v
                        
                        # 然后添加keys列表中的指标
                        for k in keys:
                            if k in results:
                                reordered_results[k] = results[k]
                        
                        # 用重新排序的结果替换原始结果
                        results = reordered_results

                    wandb.log(results, step=step)
                    wandb.run.summary.update(best_results)

                if is_best:
                    log.info(f'Step {step} (val/{config.metric}='
                             f'{results["val_metric"]}) -> best step')
                else:
                    best_step_diff = step - best_results['step']
                    log.info(f'Step {step} (val/{config.metric}='
                             f'{results["val_metric"]}) '
                             f'-> outperformed by step {best_results["step"]} '
                             f'(val/{config.metric}='
                             f'{best_results["val_metric"]})')
                    decrease_lr=False
                    if decrease_lr:
    #                     # 获取更新前的学习率
                        previous_lr = optimizer.param_groups[0]['lr']
                        log.info(f'更新前学习率: {previous_lr:.6f}')
                        
                        # 更新学习率 - 确保针对正确的优化器
                        if hasattr(model, 'optimizer'):
                            # DeepSpeed优化器
                            for param_group in model.optimizer.param_groups:
                                param_group['lr'] *= 0.9
                        else:
                            # 标准优化器
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.9
                        
                        # 获取更新后的学习率 - 确保从正确的优化器获取
                        current_lr = model.optimizer.param_groups[0]['lr'] if hasattr(model, 'optimizer') else optimizer.param_groups[0]['lr']
                        log.info(f'性能下降，学习率降低为原来的95%，更新后学习率: {current_lr:.6f}')
                        log.info(f'学习率变化比例: {current_lr/previous_lr:.2f}')
                        
    # 完全重新创建调度器
                        remaining_steps = config.max_steps - step
                        log.info(f'重新创建调度器，剩余步数: {remaining_steps}')
                        
                        # 只有当剩余步数大于0时才重新创建调度器
                        if remaining_steps > 0:
                            # 重新创建调度器，从当前降低后的学习率开始
                            new_config = copy.deepcopy(config)
                            new_config.lr = current_lr  # 使用当前降低后的学习率作为起始学习率
                            new_config.warmup_steps = 0  # 不需要再预热
                            new_config.max_steps = remaining_steps  # 使用剩余步数
                            new_config.min_lr = config.min_lr
                            # 创建新的调度器
                            lr_scheduler = build_scheduler(optimizer, new_config)
                        else:
                            log.info(f'已达到最大步数，不再重新创建调度器')

                        # 检查学习率调度器的类型和状态
                        log.info(f"学习率调度器类型: {type(lr_scheduler).__name__}")
                        
                        # 检查学习率调度器的关键参数
                        # if isinstance(lr_scheduler, CosineLRScheduler):
                        log.info(f"CosineLRScheduler参数:")
                        log.info(f"  - t_initial: {lr_scheduler.t_initial}")
                        log.info(f"  - lr_min: {lr_scheduler.lr_min}")
                        log.info(f"  - warmup_t: {lr_scheduler.warmup_t}")
                        log.info(f"  - warmup_lr_init: {lr_scheduler.warmup_lr_init}")
                        log.info(f"  - cycle_limit: {getattr(lr_scheduler, 'cycle_limit', 'N/A')}")
                        log.info(f"  - t_in_epochs: {getattr(lr_scheduler, 't_in_epochs', 'N/A')}")
                        
                        # 检查当前步骤和学习率
                        current_step = lr_scheduler.last_epoch if hasattr(lr_scheduler, 'last_epoch') else 0
                        log.info(f"  - 当前步骤: {current_step}")
                        log.info(f"  - 当前学习率: {current_lr:.8f}")
                        

                        step_lr = lr_scheduler._get_lr(i)[0]
                        log.info(f"  - 步骤 {i}: {step_lr:.8f}")
                        

                    
                    best_step_diff = step - best_results['step']

                    if not config.debug:
                        wandb.log(results, step=step)
                        wandb.run.summary.update(best_results)

                    
                    best_step_diff = step - best_results['step']
                    log.info(f'Step {step} (val/{config.metric}='
                             f'{results["val_metric"]}) '
                             f'-> outperformed by step {best_results["step"]} '
                             f'(val/{config.metric}='
                             f'{best_results["val_metric"]})')
                    

                        

                    if config.early_sopping_patience is not None:
                        if best_step_diff > config.early_sopping_patience:
                            log.info(f'Early stopping: '
                                     f'val/{config.metric} did not'
                                     f'improve for {best_step_diff} steps '
                                     f'-> stopping training')
                            stop_training = True
                            break
                        else:
                            log.info(f'Early stopping: '
                                     f'val/{config.metric} did not'
                                     f'improve for {best_step_diff} steps - '
                                     f'patience={config.early_sopping_patience}')

                # Reset progress bar
                pbar.refresh()
                pbar.reset()

            # Return if max_steps is reached
            if step >= config.max_steps:
                stop_training = True
                break
        # 每个epoch结束后清理
        torch.cuda.empty_cache()
        print_memory_summary(f"Epoch {epoch} 结束")
        epoch += 1
        epoch_end_time = timer()
        epoch_time = epoch_end_time - epoch_start_time
        log.info(f'Epoch {epoch} took {epoch_time:.2f}s')
        wandb.log({'train_epoch/epoch_time': epoch_time}, step=step)
        log.info(f'Finished epoch {epoch}')
        if stop_training:
            break

    

    log.info(f'Finished training after {step} steps / {epoch} epochs')
    if best_results is not None:
        log.info(f'Best step: {best_results["step"]} '
                    f'(val/{config.metric}='
                    f'{best_results["val_metric"]})')
    return best_results

# def train_step(model: BaseModel, samples, optimizer, lr_scheduler,  epoch, step, config: ExperimentConfig, train_data_conf: DatasetConfig) -> BaseModelOutput:
def train_step(model: BaseModel, samples, optimizer, lr_scheduler, scaler, epoch, step, config: ExperimentConfig, train_data_conf: DatasetConfig) -> BaseModelOutput:
    # Forward
    samples = to_device(samples, config.device)
    
    # 确保模型在正确的设备上
    model = model.to(config.device)
    
    # 确保优化器中的状态也在正确的设备上
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(config.device)
                
    with torch.device(config.device):
        with autocast(device_type=config.device, enabled=config.amp):
            # ipdb.set_trace()
            #chex forward
            # print(samples['has_class_bboxes'])
            output: BaseModelOutput = model.train_step(**samples, step=step, epoch=epoch, data_config=train_data_conf)
            loss = output.loss
            assert loss is not None
            loss = loss / config.accumulation_steps

    # Backward and scale if mixed precision
    scaler.scale(loss).backward()
    

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    lr_scheduler.step_update(step)

    return output


def validate(
    config: ExperimentConfig,
    val_task: EvalConfig,
    val_dl: DataLoader,
    model: BaseModel,
    model_dir,
    step,
    prefix=None,
    results_name=None,
    **kwargs,
):
    # 添加手动清理缓存
    
    if results_name is not None:
        results_dir = os.path.join(model_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, results_name)
    else:
        results_path = None

    model.eval() #'ChEX' 
    
    # build evaluator

    evaluator: Evaluator = model.build_evaluator(val_task, results_path=results_path, **kwargs) 
    #RE mscxr <model.eval.box_explainer.BoxExplainerEvaluator >
    val_task = evaluator.config
    val_data_conf = val_dl.dataset.dataset_info

    # Init progress bar
    pbar = tqdm(val_dl, 
                desc=f'验证 步骤{step}', 
                dynamic_ncols=True,
                leave=False,  # 添加这个参数
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # Iterate over batches 每个batch有4个样本，samples有四个样本
    for idx, samples in enumerate(pbar):
        # 更频繁地清理显存
        # if idx > 0 and idx % 10 == 0:  
        #     torch.cuda.empty_cache()
        #     gc.collect()
            
        samples = to_device(samples, config.device)

        with torch.inference_mode():
            with torch.device(config.device):
                with autocast(device_type=config.device, enabled=True):
                    output: BaseModelOutput = evaluator.eval_step(**samples, step=step, data_config=val_data_conf)


        try: #没经过

            
            # ipdb.set_trace()
            plot_batches = val_task.plot_val_samples // config.batch_size
            max_samples = val_task.plot_val_samples % config.batch_size if idx == plot_batches else config.batch_size
            # ipdb.set_trace()
            # if True : 
            if not config.debug and idx <= plot_batches and max_samples > 0 and (val_task.plot_wandb or val_task.plot_local): 
                # Back to CPU
                output_cpu = to_device(output, "cpu")
                pred_dir = os.path.join(model_dir, 'predictions', f'step_{step:09d}')
                os.makedirs(pred_dir, exist_ok=True)
                
                # 添加调试信息
                print(f"Plotting batch {idx} with max_samples={max_samples}")
                
                wandb_logs: dict = evaluator.plot(output_cpu, target_dir=pred_dir, max_samples=max_samples, plot_local=val_task.plot_local)

                # 检查wandb_logs中的图像
                # for key, value in wandb_logs.items():
                #     if isinstance(value, list) and len(value) > 0 and isinstance(value[0], wandb.Image):
                #         print(f"Key: {key}, Number of images: {len(value)}")

                if val_task.plot_wandb and not config.debug:
                    if prefix is not None:
                        # print(f"wandb_logs: {wandb_logs}")
                        # ipdb.set_trace()
                        wandb_logs = {f'{prefix}/{k}': v for k, v in wandb_logs.items()}
                    wandb.log(wandb_logs, step=step)
                    del wandb_logs  # 删除wandb_logs
                del output_cpu  # 删除output_cpu
            del output
            # torch.cuda.empty_cache()
        except Exception as e:
            log.error(f'Error plotting: {e}')
            # 打印更详细的错误信息
            import traceback
            log.error(traceback.format_exc())
        if config.debug:
            break  # single iteration

        # 立即删除不需要的变量
        del samples
        # if not config.debug and idx <= plot_batches:
        #     del output


    # 验证结束后清理显存    
    torch.cuda.empty_cache()
    gc.collect()
    
    # Returns
    # ipdb.set_trace()
    val_results = evaluator.compute_metrics()
    if prefix is not None:
        val_results = {f'{prefix}/{k}': v for k, v in val_results.items()}

    model.train()
    return val_results


def update_summary_with_api(results: Dict, wandb_run_api_path: str) -> None:
    if not isinstance(wandb_run_api_path, wandb.sdk.lib.disabled.RunDisabled) and len(wandb_run_api_path) > 0:
        log.info(f'Loading wandb run {wandb_run_api_path}')
        wandb_run_api = wandb.Api().run(wandb_run_api_path)
        for key, value in results.items():
            wandb_run_api.summary[key] = value
        wandb_run_api.update()


def run_training(config: ExperimentConfig, ckpt=None,ckpt_path=None):
    seed_everything(config.seed)

    # 处理传入的checkpoint
    if ckpt is not None:
        log.info("处理传入的checkpoint")
        
        if 'experiment_config' in ckpt:
            ckpt_config_dict = ckpt['experiment_config']
            log.info("成功提取检查点中的experiment_config信息")
            
            # 创建OmegaConf对象
            ckpt_config = OmegaConf.create(ckpt_config_dict)
            
            # 排除不需要合并的特定配置项
            if 'model' in ckpt_config and 'anat_tok' in ckpt_config.model:
                log.info("排除合并检查点中的model.anat_tok配置")
                ckpt_config.model.pop('anat_tok')
            
            if 'model' in ckpt_config and 'patho_tok' in ckpt_config.model:
                log.info("排除合并检查点中的model.patho_tok配置")
                ckpt_config.model.pop('patho_tok')
                
            if 'model' in ckpt_config and 'sent_tok' in ckpt_config.model:
                log.info("排除合并检查点中的model.sent_tok配置")
                ckpt_config.model.pop('sent_tok')
            
            # 合并配置
            config = OmegaConf.merge(config, ckpt_config)
            log.info("配置已与checkpoint中的配置合并（已排除指定项）")
            # 检查并记录检查点中的token信息
            if 'model' in config:
                model_config = config.model
                
                # 检查解剖学token信息
                if 'anat_tok' in model_config:
                    log.info(f"检查点中包含解剖学token信息: {len(model_config.anat_tok) if hasattr(model_config.anat_tok, '__len__') else '未知数量'}个token")
                    # 使用OmegaConf安全的方式访问配置
                    if hasattr(model_config.anat_tok, 'items') and callable(model_config.anat_tok.items):
                        sample_tokens = list(model_config.anat_tok.items())[:5]
                        for i, (key, tok) in enumerate(sample_tokens):
                            log.info(f"解剖学token样例 {key}: {tok}")
                    if len(model_config['anat_tok']) > 5:
                        log.info(f"... 共{len(model_config['anat_tok'])}个解剖学token")
                else:
                    log.info("检查点中未找到解剖学token信息")
                    
                # 检查病理学token信息
                if 'patho_tok' in model_config:
                    log.info(f"检查点中包含病理学token信息: {len(model_config.patho_tok) if hasattr(model_config.patho_tok, '__len__') else '未知数量'}个token")
                    # 使用OmegaConf安全的方式访问配置
                    if hasattr(model_config.patho_tok, 'items') and callable(model_config.patho_tok.items):
                        sample_tokens = list(model_config.patho_tok.items())[:5]
                        for i, (key, tok) in enumerate(sample_tokens):
                            log.info(f"病理学token样例 {key}: {tok}")
                    if len(model_config['patho_tok']) > 5:
                        log.info(f"... 共{len(model_config['patho_tok'])}个病理学token")
                else:
                    log.info("检查点中未找到病理学token信息")
                    
                # 检查句子token信息
                if 'sent_tok' in model_config:
                    log.info(f"检查点中包含句子token信息: {len(model_config.sent_tok) if hasattr(model_config.sent_tok, '__len__') else '未知数量'}个token")
                    # 使用OmegaConf安全的方式访问配置
                    if hasattr(model_config.sent_tok, 'items') and callable(model_config.sent_tok.items):
                        sample_tokens = list(model_config.sent_tok.items())[:5]
                        for i, (key, tok) in enumerate(sample_tokens):
                            log.info(f"句子token样例 {key}: {tok}")
                    if len(model_config['sent_tok']) > 5:
                        log.info(f"... 共{len(model_config['sent_tok'])}个句子token")
                else:
                    log.info("检查点中未找到句子token信息")
            else:
                log.info("检查点中未找到模型配置信息或token信息")
        else:
            log.info("检查点中没有找到experiment_config信息")
            
    

    try:
        # 尝试使用Hydra配置（如果在Hydra上下文中）
        hydra_config = HydraConfig.get()
        override_dirname = hydra_config.job.override_dirname
        model_dir = hydra_config.run.dir
        log.info(f'model_dir: {model_dir}')
    except ValueError:
        # 如果不在Hydra上下文中，使用替代方案
        log.info("不在Hydra上下文中，使用替代目录名")
        override_dirname = ""
        # 创建一个时间戳目录作为替代
        if ckpt and 'model_dir' in ckpt:
            model_dir = ckpt['model_dir']
        else:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_dir = os.path.join(MODELS_DIR, config.name, f"manual_run_{timestamp}") if not config.debug else None
            os.makedirs(model_dir, exist_ok=True)
            
    # 其余代码保持不变
    full_model_name = f'{config.name}/{override_dirname}' if len(override_dirname) > 0 else config.name
    print(f'full_model_name:{full_model_name}')
    if config.debug:
        log.info('Running in debug mode -> fast run to check for runtime errors')
        model_dir=None
        config.prefetch = False
        config.print_freq = 1
        config.val_freq = 1
    else:
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            os.chdir(model_dir)
            
    if not(not config.debug and not config.train and config.evaluate): #非仅eval
        # Load the model
        if config.continue_from_checkpoint is not None: #stage1 none
            model = load_model_from_checkpoint(config.continue_from_checkpoint)
        else:
            if ckpt:
                model, optimizer, lr_scheduler = load_model_and_optimizer_and_lr_scheduler_from_checkpoint(ckpt_path, config)
                step = ckpt['step']
                log.info(f"Resuming step from checkpoint: {step}")
                # 检查学习率是否正确恢复
                if optimizer:
                    current_lr = optimizer.param_groups[0]['lr']
                    log.info(f"从检查点恢复的学习率: {current_lr:.6f}")
                    
                    # 如果学习率调度器存在，确保其基础学习率与优化器一致
                    if lr_scheduler is not None:
                        if hasattr(lr_scheduler, 'base_lrs'):
                            log.info(f"学习率调度器基础学习率: {lr_scheduler.base_lrs}")
                            # 确保调度器的基础学习率与优化器当前学习率一致
                            if lr_scheduler.base_lrs[0] != current_lr:
                                lr_scheduler.base_lrs = [current_lr for _ in lr_scheduler.base_lrs]
                                log.info(f"已更新学习率调度器的base_lrs为当前学习率: {lr_scheduler.base_lrs}")
                # 更彻底地确保所有内容都在正确设备上
                model = model.to(config.device)
                
                # 直接将优化器状态移动到正确的设备上
                for param_group in optimizer.param_groups:
                    for p in param_group['params']:
                        if p in optimizer.state:
                            for k, v in optimizer.state[p].items():
                                if isinstance(v, torch.Tensor):
                                    optimizer.state[p][k] = v.to(config.device)
            else:
                model: BaseModel = instantiate_model(config.model) ##这里load总模型chex模块各自的ckpt
                optimizer = None
                lr_scheduler = None
        if model_dir is not None:
            OmegaConf.save(config=config,
                            f=os.path.join(model_dir, 'experiment_config.yaml'))
            OmegaConf.save(config=model.config,
                            f=os.path.join(model_dir, 'model_config.yaml'))
            log.info(f'Starting training of {full_model_name} ({type(model).__name__})')
        model = model.to(device=config.device)
        log.info(f'Model and training logs will be stored at: {model_dir}')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    """ Init W&B """
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name='继续训练eval',
        dir='.',
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        resume=False, #'must' if config.resume else False,
        reinit=True,  # without reinit there may be problems when running Hydra sweeps
        settings=wandb.Settings(start_method='thread', init_timeout=300, _service_wait=300),  # fork TODO: thread? -> for hydra sweeps
        mode="disabled" if config.debug else "online", 
    )
    if not(not config.debug and not config.train and config.evaluate):
        wandb.summary['n_parameters'] = n_parameters
        log.info(f'Number of params: {n_parameters}')
        wandb_api_path = str(wandb.run.path)
        log.info(f'API path: {wandb_api_path}')
    
    config.compile = False 
    if config.compile and not config.debug:
        log.info(f"编译前 - 模型类型: {type(model).__name__}")
        param_names_before = list(dict(model.named_parameters()).keys())[:3]
        log.info(f"编译前 - 模型参数名样例: {param_names_before}")

        log.info("执行torch.compile...")
        model = torch.compile(model, dynamic=True, mode='max-autotune')
        
        # 在compile后添加日志
        log.info(f"编译后 - 模型类型: {type(model).__name__}")
        param_names_after = list(dict(model.named_parameters()).keys())[:3]
        log.info(f"编译后 - 模型参数名样例: {param_names_after}")
        
        # 检查是否有_orig_mod前缀
        has_orig_mod = any('_orig_mod' in name for name in dict(model.named_parameters()).keys())
        log.info(f"编译后 - 参数是否有_orig_mod前缀: {has_orig_mod}")


    if config.model.name =='chex_stage1_raddino' or config.model.name == 'contr_train_img_txt':
        use_image_paths = True
    elif config.model.name == 'chex_stage1_clip':
        use_image_paths = False
    else:
        raise ValueError(f"未知的模型名称: {config.model.name}")
        
    if  config.model.name == 'contr_train_img_txt':
        pretrain = True
    elif config.model.name =='chex_stage1_raddino' or config.model.name == 'chex_stage1_clip':
        pretrain = False
    else:
        raise ValueError(f"未知的模型名称: {config.model.name}")
    if not(not config.debug and not config.train and config.evaluate): #train and val
        results = train(config, model=model, model_dir=model_dir, full_model_name=full_model_name, use_image_paths=use_image_paths,pretrain=pretrain,optimizer=optimizer,lr_scheduler=lr_scheduler,ckpt=ckpt)
        #  (supervisors): ModuleDict(
    # (sent_tok): SentenceTokenSupervisor(
    #   (aggregator): GlobalAvgPool()
    # )
    # (anat_tok): AnatomyTokenSupervisor()
    # (patho_tok): PathologyTokenSupervisor()
    else:
        results = {}
        
    if config.evaluate :
        # if config.train:
        # Load the best model from training
        if config.evaluate and not config.debug and not config.train:
            run_name = '/rds/general/user/lw1824/home/chex/chex/models/chex_stage1/run_2025-05-29_08-08-37/checkpoints/checkpoint_000001550.pth'
            log.info(f' evaluate run_name: {run_name}')
            if run_name=='': assert False, 'run_name is empty'
        else:
            run_name = None
        #根据config再次load模型和参数（可能是best，可能是单纯跑一个模型）
        log.info(f'model_dir: {model_dir}')
        model = load_model_by_name(config,full_model_name, load_best=True,run_name=run_name,model_dir=model_dir)
        model = model.to(device=config.device)
        log.info(f'load model from {full_model_name} (load_model_by_name)')
        for task_name, task, task_dl, _ in build_dataloaders_for_eval(config,use_image_paths=use_image_paths,pretrain=pretrain):
            log.info(f'Evaluating task {task_name} ({config.eval_mode})')
            eval_results = validate(model=model, val_task=task, val_dl=task_dl, model_dir=model_dir, config=config, step=0, prefix=f'{config.eval_mode}_{task_name}')
            if not config.debug:
                wandb.log(eval_results)
                wandb.summary.update(eval_results)
            results.update(eval_results)
    wandb.finish()
    #update_summary_with_api(results, wandb_api_path)

    return results['val_metric'] if results is not None and 'val_metric' in results else None

@hydra.main(config_path="../../conf", config_name="train_rad", version_base=None)
def do_run_training(config):
    # 打印GPU信息
    gpu_count = torch.cuda.device_count()
    log.info(f"当前可用的GPU数量: {gpu_count}")
    
    # 打印每个GPU的名称和内存信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 转换为GB
        log.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")
    
    # 直接创建ckpt并传递给run_training
    ckpt_path = '/rds/general/user/lw1824/home/chex/chex/models/chex_stage1/run_2025-05-29_08-08-37/checkpoints/checkpoint_000001550.pth'
    log.info(f"ckpt_path: {ckpt_path}")
    # ckpt_path = None
    # ckpt=None
    if ckpt_path is not None:
        try:
            map_location = 'cpu'
            ckpt = torch.load(ckpt_path, map_location=map_location)
            log.info(f"检查点加载成功: {ckpt_path}")
        except Exception as e:
            log.error(f"加载检查点时出错: {e}")
            import traceback
            log.error(traceback.format_exc())
            ckpt = None
    
    # 直接调用run_training并传递ckpt
    return run_training(config, ckpt=ckpt,ckpt_path=ckpt_path)

if __name__ == "__main__":
    # 基本设置
    logging.basicConfig(level=logging.DEBUG)
    cs = ConfigStore.instance()
    cs.store(name="ExperimentConfig", node=ExperimentConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
    OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver("ifel", lambda flag, val_true, val_false: val_true if flag else val_false)
    OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR)

    import model
    from model import img_encoder, txt_encoder, txt_decoder, detector
    ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector])
    
    # 直接调用do_run_training，它会处理配置和checkpoint
    result = do_run_training()
    
    # 如果需要，可以使用返回的结果

#     # Prepare model dir
#     override_dirname = HydraConfig.get().job.override_dirname
#     # ipdb.set_trace()
#     full_model_name = f'{config.name}/{override_dirname}' if len(override_dirname) > 0 else config.name
#     if config.debug:
#         log.info('Running in debug mode -> fast run to check for runtime errors')
#         model_dir = None
#         config.prefetch = False
#         config.print_freq = 1
#         config.val_freq = 1
#     else:
#         model_dir = HydraConfig.get().run.dir
#         os.chdir(model_dir)

#     # # Load the model
#     # if config.continue_from_checkpoint is not None: #stage1 none
#     #     model = load_model_from_checkpoint(config.continue_from_checkpoint)
#     # else:
#     #     model: BaseModel = instantiate_model(config.model) #config.model:chex

#     # if model_dir is not None:
#     #     OmegaConf.save(config=config,
#     #                     f=os.path.join(model_dir, 'experiment_config.yaml'))
#     #     OmegaConf.save(config=model.config,
#     #                     f=os.path.join(model_dir, 'model_config.yaml'))
#     #     log.info(f'Starting training of {full_model_name} ({type(model).__name__})')
#     # model = model.to(device=config.device)
#     # log.info(f'Model and training logs will be stored at: {model_dir}')
#     # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     """ Init W&B """
#     wandb.init(
#         project=WANDB_PROJECT,
#         entity=WANDB_ENTITY,
#         name='chex-stage1-raddino1',
#         # tags=[type(model).__name__],
#         dir='.',
#         config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
#         resume=False, #'must' if config.resume else False,
#         reinit=True,  # without reinit there may be problems when running Hydra sweeps
#         settings=wandb.Settings(start_method='thread', init_timeout=300, _service_wait=300),  # fork TODO: thread? -> for hydra sweeps
#         mode="disabled" if config.debug else "online", 
#     )
#     # wandb.summary['n_parameters'] = n_parameters
#     # log.info(f'Number of params: {n_parameters}')
#     wandb_api_path = str(wandb.run.path)
#     log.info(f'API path: {wandb_api_path}')

#     # if config.compile and not config.debug:
#     #     model = torch.compile(model, dynamic=True, mode='max-autotune')
#         # train = torch.compile(train, dynamic=True, mode='max-autotune')
#         # validate = torch.compile(validate, dynamic=True, mode='max-autotune')

#     if False: #train and val 修改

#         results = train(config, model=model, model_dir=model_dir, full_model_name=full_model_name)
#         #  (supervisors): ModuleDict(
#     # (sent_tok): SentenceTokenSupervisor(
#     #   (aggregator): GlobalAvgPool()
#     # )
#     # (anat_tok): AnatomyTokenSupervisor()
#     # (patho_tok): PathologyTokenSupervisor()
#     else:
#         results = {}
        
#     if config.evaluate and not config.debug:
#         if config.train:
#             # Load the best model from training
#             model = load_model_by_name(full_model_name, load_best=True,run_name='run_2025-04-09_13-42-23')
#             model = model.to(device=config.device)
#             log.info(f'load model from {full_model_name} (load_model_by_name)')
#         for task_name, task, task_dl, _ in build_dataloaders_for_eval(config):
#             log.info(f'Evaluating task {task_name} ({config.eval_mode})')
#             eval_results = validate(model=model, val_task=task, val_dl=task_dl, model_dir=model_dir, config=config, step=0, prefix=f'{config.eval_mode}_{task_name}')
#             wandb.log(eval_results)
#             wandb.summary.update(eval_results)
#             results.update(eval_results)
#     wandb.finish()
#     #update_summary_with_api(results, wandb_api_path)

#     return results['val_metric'] if results is not None and 'val_metric' in results else None


























        # Plotting

        # # 创建保存目录
        # output_dir = "annotated2_images"
        # os.makedirs(output_dir, exist_ok=True)

        # # 图像张量转换为 PIL 图像的函数
        # def tensor_to_pil_image(tensor):
        #     if tensor.dim() == 3:  # 如果是 [C, H, W]
        #         tensor = tensor.permute(1, 2, 0)
        #     tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]
        #     tensor = (tensor * 255).byte()  # 转为 uint8
        #     return Image.fromarray(tensor.cpu().numpy())

        # # 绘制目标框和简化句子的函数
        # def draw_boxes(image, boxes, sentences):
        #     # 确保图片是 RGB 模式，以便显示彩色字体
        #     if image.mode != "RGB":
        #         image = image.convert("RGB")
            
        #     draw = ImageDraw.Draw(image)
        #     font = ImageFont.load_default()
        #     for i, (box, sentence) in enumerate(zip(boxes, sentences)):
        #         # 获取坐标并转换为像素
        #         x, y, w, h, label = box
        #         x_min = int((x - w / 2) * image.width)
        #         y_min = int((y - h / 2) * image.height)
        #         x_max = int((x + w / 2) * image.width)
        #         y_max = int((y + h / 2) * image.height)

        #         # 绘制矩形框
        #         draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

        #         # 简化句子（只显示前3个单词）
        #         simplified_sentence = " ".join([sentence.split()[0],sentence.split()[-1]])

        #         # 直接绘制文字
        #         draw.text((x_min + 2, y_min +1), simplified_sentence, fill="yellow", font=font)  # 黄色文字

        #     return image

        # 保存图片和文本信息
        # for i, (id,image_tensor, boxes, gen_sentences, tgt_sentences) in enumerate(zip(samples["sample_id"],output.output_img, output.target_cls_boxes, output.generated_sentences, output.target_sentences)):
        #     # ipdb.set_trace()
        #     # 转换图像为 PIL 格式
        #     image = tensor_to_pil_image(image_tensor)

        #     # 获取边界框和生成句子
        #     boxes = boxes.cpu().numpy()  # 转为 numpy 格式
        #     gen_sentences = gen_sentences  # 生成的句子
        #     tgt_sentences = tgt_sentences  # 目标句子

        #     # 在图像上绘制边界框和句子
        #     annotated_image = draw_boxes(image, boxes, gen_sentences)

        #     # 保存标注后的图像
        #     output_path = os.path.join(output_dir, f"sample_{'-'.join(id.split('/'))}.jpg")
        #     annotated_image.save(output_path)
        #     # print(f"Saved annotated image: {output_path}")

        #     # 保存对应的文本信息（包括 generated_sentences 和 target_sentences）
        #     text_path = os.path.join(output_dir, f"sample_{'-'.join(id.split('/'))}.txt")
        #     with open(text_path, "w") as f:
        #         f.write("Generated Sentences:\n")
        #         for sentence in gen_sentences:
        #             f.write(f"- {sentence}\n")
        #         f.write("\nTarget Sentences:\n")
        #         for sentence in tgt_sentences:
        #             f.write(f"- {sentence}\n")
        #     # print(f"Saved text information: {text_path}")
