from dataclasses import dataclass
from typing import Any, Dict, Optional
import hydra

from omegaconf import MISSING, OmegaConf
from dataclasses import dataclass
import logging
import os
from torch import autocast
# from torch.cuda.amp import GradScaler
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

from util.model_utils import BaseModel, BaseModelOutput, ModelRegistry, instantiate_model, load_model_by_name, load_model_from_checkpoint
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
import copy
import torch.distributed as dist

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

def train(config: ExperimentConfig, model: BaseModel, model_dir: str, full_model_name: str,use_image_paths:bool,pretrain:bool):
    # Determine if this is the main process
    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    # Load the datasets for training and validation
    train_dl = build_train_dataloader(config, use_image_paths, pretrain)
    train_data_conf = train_dl.dataset.dataset_info
    val_dls={}
    
    for i, (val_name, val_task) in enumerate(config.val_tasks.items()):
        val_dls[val_name] = build_val_dataloader(config, val_task, use_image_paths, pretrain)

    # Define variables: steps_per_epoch; max_steps; val_freq
    steps_per_epoch = len(train_dl)
    log.info(f'Note: {steps_per_epoch} steps per epoch')
    assert config.max_steps is None or config.max_epochs is None
    # Can choose epoch or step
    if config.max_steps is None and config.max_epochs is not None: # Can change to step
        config.max_steps = config.max_epochs * steps_per_epoch
    log.info(f'max_steps: {config.max_steps}')
    assert config.val_freq is not None or config.val_ep is not None
    if config.val_freq is None: 
        config.val_freq = steps_per_epoch * config.val_ep

    # Prepare optimizer, scheduler, and scaler
    optimizer = build_optimizer(model, config)
    # lr_scheduler = build_scheduler(optimizer, config)
    # ipdb.set_trace()
    model.to(config.device)
    # Print current path
    import os
    current_path = os.getcwd()
    log.info(f"Current working path: {current_path}")
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config='/rds/general/user/lw1824/home/chex/chex/src_new/src_sec/config.json'
    )
    lr_scheduler = build_scheduler(optimizer, config)
    log.info(f"当前device: {model.device}")
    print('model type:',type(model))
    # scaler = GradScaler()

    step = 0
    epoch = 0
    step_metrics_meter = AvgDictMeter()

    # Add variable to record total time at function start
    train_start_time = timer()
    total_steps_done = 0

    """ Start training """
    log.info(f'Starting training of {full_model_name}')
    log.info(f"Using {config.device}")
    if config.debug:
        torch.autograd.set_detect_anomaly(True)
    best_results = None
    last_results = None
    # Calculate steps per validation cycle
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
            # Calculate using total training time and total steps
            time_per_step = total_elapsed / step
            
            total_time = time_per_step * config.max_steps
            remaining_total = time_per_step * (config.max_steps - step)
            
            # Format time display
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                return f"{hours}h{minutes}m"
            
            # Simplify display information, reduce length
            pbar.set_postfix_str(
                f'l={step_metrics_meter.compute()["loss"]:.3f}, '
                f'GPU:{gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f}GB, '
                f'Total:{format_time(total_time)}, '
                f'Remaining:{format_time(remaining_total)}'
            )

    # Start iterating through training steps inside train
    while True:
        # print_memory_summary(f"Epoch {epoch} start")
        epoch_start_time = timer()
        it_start_time = timer()
        stop_training = False
        for samples in train_dl:

            pbar.update(1)

            # Training step
            # output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, scaler, epoch, step, config, train_data_conf)
            output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, epoch, step, config, train_data_conf)
            step_metrics_meter.add(dict(output.step_metrics, loss=output.loss.detach())) # Record average of each subloss and total loss

            if torch.isnan(output.loss):
                log.error(f'Loss was nan: {output.step_metrics}')
                #stop_training = True
                #break
                
            # Increment step
            step += 1

            # Clear GPU cache every 40 steps
            if step % 25 == 0:
                torch.cuda.empty_cache()
                gc.collect()
                log.info(f"Clear GPU cache every 40 steps: Allocated {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, Reserved {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")

            # Training Update progress bar and log losses
            if step % config.print_freq == 0 and not step % config.val_freq == 0:
                it_end_time = timer()
                it_time = it_end_time - it_start_time
                # it_start_time = it_end_time
                
                # Modified: Get learning rate from DeepSpeed optimizer
                if hasattr(optimizer, 'param_groups'):
                    # Standard optimizer
                    log.info(f"Standard optimizer")
                    lr = optimizer.param_groups[0]['lr']
                else:
                    # DeepSpeed optimizer
                    log.info(f"DeepSpeed optimizer")
                    lr = model.optimizer.param_groups[0]['lr']
                
                # Ensure lr is a Python scalar
                lr = float(lr)
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                gc.collect()
                gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                gpu_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                
                update_eta_total(pbar, step, val_steps)
                
                
                # Record current GPU memory usage
                # log.info(f"GPU memory usage: Allocated {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, Reserved {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
                # ipdb.set_trace()
                if is_main_process:
                    # Add log to record learning rate value for debugging

                    wandb.log({
                        'train_step/lr': lr,
                        'train_step/loss': float(output.loss.detach()),
                        'gpu/memory_allocated_gb': gpu_memory_allocated,
                        'gpu/memory_reserved_gb': gpu_memory_reserved,
                        **{'train_step/' + key: value for key, value in output.step_metrics.items()}
                    }, step=step)

            # Validate and log at validation frequency
            if config.debug or (step >= config.max_steps or step % config.val_freq == 0 or (config.val_ep is not None and step % (config.val_ep * steps_per_epoch) == 0)):
                # Gather training results
                # Clear GPU cache
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
                    # Harmonic mean calculation, add more robust handling for small values
                    val_metrics = [results[f'val/{m}'] for m in config.metric.split(',')]
                    
                    # Check if there are very small values
                    min_threshold = 0.2  # Can adjust this threshold based on your specific scenario # 0.3 0.4 0.5 0.2 0.1 0.7
                    
                    # If there are very small values, consider the following solutions
                    if any(m < min_threshold for m in val_metrics):
                        # Solution 1: Scale all metrics to avoid very small values
                        # For example, if metrics range is usually 0-1, map them to 0.01-1.01
                        # scaled_metrics = [m + min_threshold for m in val_metrics]
                        # results['val_metric'] = len(scaled_metrics) / sum(1. / m for m in scaled_metrics)
                        
                        # Solution 2: Use geometric mean instead of harmonic mean
                        # import numpy as np
                        # results['val_metric'] = np.exp(sum(np.log(max(m, min_threshold)) for m in val_metrics) / len(val_metrics))
                        
                        # Solution 3: Use arithmetic mean
                        results['val_metric'] = sum(val_metrics) / len(val_metrics)
                    else:
                        # Original harmonic mean calculation
                        results['val_metric'] = len(val_metrics) / sum(1. / m for m in val_metrics)
                else:
                    results['val_metric'] = results[f'val/{config.metric}']
                

                if config.debug:
                    log.info('Debug mode -> Stopping training after 1 step')
                    return results

                best_results, last_results,is_best,is_better_than_last = get_best_results(results, best_results,last_results,
                                                         config)
                # save_training_checkpoint(model, optimizer, lr_scheduler, scaler,
                #                          results=results,
                #                          best_results=best_results,
                #                          config=config, step=step,
                #                          saved_components=config.save_components,
                #                          is_best=is_best)
                # ipdb.set_trace()
                save_training_checkpoint(model, optimizer, lr_scheduler, 
                                         results=results,
                                         best_results=best_results,
                                         config=config, step=step,
                                         saved_components=config.save_components,
                                         is_best=is_best)

                if is_main_process:
                    wandb.log(results, step=step)
                    wandb.run.summary.update(best_results)

                if is_best:
                    log.info(f'Step {step} (val/{config.metric}='
                             f'{results["val_metric"]}) -> best step')
                else:
                    # Get learning rate before update
                    previous_lr = optimizer.param_groups[0]['lr']
                    log.info(f'Learning rate before update: {previous_lr:.6f}')
                    
                    # Update learning rate - ensure targeting the correct optimizer
                    # Adjust learning rate decay based on performance changes
                    lr_decay_factor = 0.95 if is_better_than_last else 0.85
                    
                    if hasattr(model, 'optimizer'):
                        # DeepSpeed optimizer
                        for param_group in model.optimizer.param_groups:
                            param_group['lr'] *= lr_decay_factor
                    else:
                        # Standard optimizer
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_decay_factor
                    
                    # Get learning rate after update
                    current_lr = model.optimizer.param_groups[0]['lr'] if hasattr(model, 'optimizer') else optimizer.param_groups[0]['lr']
                    log.info(f'Performance declined, learning rate reduced to {lr_decay_factor} of original, updated learning rate: {current_lr:.6f}')
                    
                    # Completely recreate scheduler
                    remaining_steps = config.max_steps - step
                    log.info(f'Recreating scheduler, remaining steps: {remaining_steps}')
                    
                    # Only recreate scheduler if remaining steps > 0
                    if remaining_steps > 0:
                        # Recreate scheduler starting from current reduced learning rate
                        new_config = copy.deepcopy(config)
                        new_config.lr = current_lr  # Use current reduced learning rate as starting learning rate
                        new_config.warmup_steps = 0  # No need for warmup again
                        new_config.max_steps = remaining_steps  # Use remaining steps
                        new_config.min_lr = config.min_lr
                        # Create new scheduler
                        lr_scheduler = build_scheduler(optimizer, new_config)
                    else:
                        log.info(f'Reached maximum steps, no longer recreating scheduler')
                    # Check learning rate scheduler type and status
                    log.info(f"Learning rate scheduler type: {type(lr_scheduler).__name__}")
                    
                    # Check key parameters of learning rate scheduler
                    # if isinstance(lr_scheduler, CosineLRScheduler):
                    log.info(f"CosineLRScheduler parameters:")
                    log.info(f"  - t_initial: {lr_scheduler.t_initial}")
                    log.info(f"  - lr_min: {lr_scheduler.lr_min}")
                    log.info(f"  - warmup_t: {lr_scheduler.warmup_t}")
                    log.info(f"  - warmup_lr_init: {lr_scheduler.warmup_lr_init}")
                    log.info(f"  - cycle_limit: {getattr(lr_scheduler, 'cycle_limit', 'N/A')}")
                    log.info(f"  - t_in_epochs: {getattr(lr_scheduler, 't_in_epochs', 'N/A')}")
                    
                    # Check current step and learning rate
                    current_step = lr_scheduler.last_epoch if hasattr(lr_scheduler, 'last_epoch') else 0
                    log.info(f"  - Current step: {current_step}")
                    log.info(f"  - Current learning rate: {current_lr:.8f}")
                    
                    # Iterate through all steps of learning rate scheduler, print learning rate changes
                    log.info(f"Complete learning rate scheduler change curve:")
                    step_interval = max(1, remaining_steps // 20)  # Choose appropriate interval, print at most 20 points
                    for i in range(0, remaining_steps + 1, step_interval):
                        if hasattr(lr_scheduler, '_get_lr'):
                            try:
                                step_lr = lr_scheduler._get_lr(i)[0]
                                log.info(f"  - Step {i}: {step_lr:.8f}")
                            except Exception as e:
                                log.info(f"  - Unable to get learning rate for step {i}: {str(e)}")

                    log.info(f'Learning rate scheduler recreated, starting learning rate: {current_lr:.6f}, minimum learning rate: {config.min_lr:.6f}')
                    

                    
                    best_step_diff = step - best_results['step']
                    log.info(f'Step {step} (val/{config.metric}='
                             f'{results["val_metric"]}) '
                             f'-> outperformed by step {best_results["step"]} '
                             f'(val/{config.metric}='
                             f'{best_results["val_metric"]})')
                    

                    log.info(f'Performance declined, loading best model (step {best_results["step"]})...')
                    
                    # # 在分布式环境中确保所有进程同步
                    # if torch.distributed.is_initialized():
                    #     torch.distributed.barrier()
                    
                    # # 加载最佳模型
                    # best_checkpoint_path = os.path.join('checkpoints', 'checkpoint_best.pth')
                    # checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
                    
                    # # 加载前保存一个参数的副本用于比较
                    # if hasattr(model, 'module'):
                    #     old_param = next(iter(model.module.parameters())).clone().cpu()
                    # else:
                    #     old_param = next(iter(model.parameters())).clone().cpu()

                    # # 对于DeepSpeed模型，需要使用特定的加载方式
                    # if hasattr(model, 'module'):
                    #     # DeepSpeed模型
                    #     ckpt_dir = 'checkpoints'
                    #     # 确保目录存在
                    #     if not os.path.exists(ckpt_dir):
                    #         log.error(f"检查点目录 {ckpt_dir} 不存在")
                    #         # 尝试查找可能的检查点文件
                    #         potential_ckpts = [f for f in os.listdir(os.path.join('checkpoints','best')) if os.path.isdir(os.path.join('checkpoints','best', f))]
                    #         log.info(f"可用的检查点目录: {potential_ckpts}")
                    #         # 如果找不到检查点，则继续训练而不是崩溃
                    #         log.warning("继续使用当前模型而不加载检查点")
                    #     else:
                    #         success, client_state = model.load_checkpoint(
                    #             ckpt_dir,
                    #             tag='best',
                    #             load_optimizer_states=False,
                    #             load_lr_scheduler_states=False,
                    #             load_module_strict=True
                    #         )
                    #         if not success:
                    #             log.error(f"无法从 {ckpt_dir} 加载模型权重")
                    #             # 查看目录内容
                    #             if os.path.exists(ckpt_dir):
                    #                 files = os.listdir(ckpt_dir)
                    #                 log.info(f"{ckpt_dir} 目录包含以下文件: {files}")
                    #             # 继续训练而不是崩溃
                    #             log.warning("继续使用当前模型而不加载检查点")
                    # else:
                    #     # 普通模型
                    #     model.load_state_dict(checkpoint['state_dict'])
                        
                    # # 加载后
                    # if hasattr(model, 'module'):
                    #     new_param = next(iter(model.module.parameters())).clone().cpu()
                    # else:
                    #     new_param = next(iter(model.parameters())).clone().cpu()

                    # # 比较是否有变化
                    # param_diff = torch.sum(torch.abs(new_param - old_param))
                    # log.info(f"参数变化量: {param_diff.item()}")
                    # if param_diff.item() < 1e-6:
                    #     log.warning("模型参数基本没有变化，加载可能失败！")

                    # # 在分布式环境中再次同步所有进程
                    # if torch.distributed.is_initialized():
                    #     torch.distributed.barrier()
                        
                    log.info(f'已加载最佳模型参数')
                    
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
        # Clean up after each epoch
        torch.cuda.empty_cache()
        print_memory_summary(f"Epoch {epoch} end")
        epoch += 1
        epoch_end_time = timer()
        epoch_time = epoch_end_time - epoch_start_time
        log.info(f'Epoch {epoch} took {epoch_time:.2f}s')
        if is_main_process:
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

def train_step(model: BaseModel, samples, optimizer, lr_scheduler, epoch, step, config: ExperimentConfig, train_data_conf: DatasetConfig) -> BaseModelOutput:
    # Forward
    samples = to_device(samples, config.device)
    with torch.device(config.device):
        with autocast(device_type=config.device, enabled=config.amp):
            output: BaseModelOutput = model.train_step(**samples, step=step, epoch=epoch, data_config=train_data_conf)
            loss = output.loss
            assert loss is not None
    
    # DeepSpeed backward和step
    model.backward(loss)
    model.step()
    
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
    # Determine if this is the main process
    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
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
                desc=f'Validation step {step}', 
                dynamic_ncols=True,
                leave=False,  # Add this parameter
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # Iterate over batches, each batch has 4 samples, samples has four samples
    for idx, samples in enumerate(pbar):
        # Restore this frequent GPU memory cleanup code
        if idx > 0 and idx % 5 == 0:  # Increase frequency to clean every 5 batches
            torch.cuda.empty_cache()
            gc.collect()
            
        samples = to_device(samples, config.device)

        with torch.inference_mode():
            with torch.device(config.device):
                with autocast(device_type=config.device, enabled=True):
                    output: BaseModelOutput = evaluator.eval_step(**samples, step=step, data_config=val_data_conf)


        try: # Not passed through

            
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
                
                # Add debug information
                print(f"Plotting batch {idx} with max_samples={max_samples}")
                
                wandb_logs: dict = evaluator.plot(output_cpu, target_dir=pred_dir, max_samples=max_samples, plot_local=val_task.plot_local)

                # Check images in wandb_logs
                for key, value in wandb_logs.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], wandb.Image):
                        print(f"Key: {key}, Number of images: {len(value)}")

                if val_task.plot_wandb and is_main_process:
                    if prefix is not None:
                        wandb_logs = {f'{prefix}/{k}': v for k, v in wandb_logs.items()}
                    wandb.log(wandb_logs, step=step)
                    del wandb_logs  # Delete wandb_logs
                del output_cpu  # Delete output_cpu
            del output
            # torch.cuda.empty_cache()
        except Exception as e:
            log.error(f'Error plotting: {e}')
            # 打印更详细的错误信息
            import traceback
            log.error(traceback.format_exc())
        if config.debug:
            break  # single iteration

        # 添加更积极的显存管理
        del samples
        # if hasattr(output, 'output_img'):
        #     del output.output_img
        # if hasattr(output, 'target_cls_boxes'):
        #     del output.target_cls_boxes
        # del output
        torch.cuda.empty_cache()

    # 在验证结束后添加分布式同步点
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # 更彻底地清理显存
    torch.cuda.empty_cache()
    gc.collect()
    val_results = evaluator.compute_metrics()
    if prefix is not None:
        val_results = {f'{prefix}/{k}': v for k, v in val_results.items()}
    
    
    model.train()

    # ----------------- 新增：每次验证后销毁并重建进程组 -----------------
    if dist.is_initialized():
        # 1. 等待所有 rank 到位
        dist.barrier()
        # 2. 销毁当前默认进程组
        dist.destroy_process_group()
        # 3. 重新初始化（此处示例使用 NCCL + 环境变量方式，你可按需调整）
        dist.init_process_group(backend='nccl', init_method='env://')
    # ------------------------------------------------------------------

    return val_results


def update_summary_with_api(results: Dict, wandb_run_api_path: str) -> None:
    if not isinstance(wandb_run_api_path, wandb.sdk.lib.disabled.RunDisabled) and len(wandb_run_api_path) > 0:
        log.info(f'Loading wandb run {wandb_run_api_path}')
        wandb_run_api = wandb.Api().run(wandb_run_api_path)
        for key, value in results.items():
            wandb_run_api.summary[key] = value
        wandb_run_api.update()


def run_training(config: ExperimentConfig):
    seed_everything(config.seed)

    # 获取当前进程的rank
    is_main_process = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    
    # Prepare model dir
    override_dirname = HydraConfig.get().job.override_dirname
    full_model_name = f'{config.name}/{override_dirname}' if len(override_dirname) > 0 else config.name
    if config.debug:
        log.info('Running in debug mode -> fast run to check for runtime errors')
        model_dir = None
        config.prefetch = False
        config.print_freq = 1
        config.val_freq = 1
    else:
        model_dir = HydraConfig.get().run.dir
        os.chdir(model_dir)

    # Load the model
    if config.continue_from_checkpoint is not None:
        model = load_model_from_checkpoint(config.continue_from_checkpoint)
    else:
        model: BaseModel = instantiate_model(config.model)

    if model_dir is not None and is_main_process:
        OmegaConf.save(config=config,
                        f=os.path.join(model_dir, 'experiment_config.yaml'))
        OmegaConf.save(config=model.config,
                        f=os.path.join(model_dir, 'model_config.yaml'))
        log.info(f'Starting training of {full_model_name} ({type(model).__name__})')
    model = model.to(device=config.device)
    if is_main_process:
        log.info(f'Model and training logs will be stored at: {model_dir}')
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    """ Init W&B """
    if is_main_process:
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name='contr',
            tags=[type(model).__name__],
            dir='.',
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
            resume=False,
            reinit=True,
            settings=wandb.Settings(start_method='thread', init_timeout=300, _service_wait=300),
            mode="disabled" if config.debug else "online", 
        )
        wandb.summary['n_parameters'] = n_parameters
        log.info(f'Number of params: {n_parameters}')
        wandb_api_path = str(wandb.run.path)
        log.info(f'API path: {wandb_api_path}')
    else:
        wandb_api_path = None

    if config.compile and not config.debug:
        model = torch.compile(model, dynamic=True, mode='max-autotune')
        # train = torch.compile(train, dynamic=True, mode='max-autotune')
        # validate = torch.compile(validate, dynamic=True, mode='max-autotune')

    if config.train: #train and val
        if config.model.name =='chex_stage1_raddino' or config.model.name == 'contr_train_img_txt':
            use_image_paths = True
        elif config.model.name == 'chex_stage1_clip':
            use_image_paths = False
        else:
            warnings.warn(f"未知的模型名称: {config.model.name}")
        
        if  config.model.name == 'contr_train_img_txt':
            pretrain = True
        elif config.model.name =='chex_stage1_raddino' or config.model.name == 'chex_stage1_clip':
            pretrain = False
        else:
            warnings.warn(f"未知的模型名称: {config.model.name}")
        results = train(config, model=model, model_dir=model_dir, full_model_name=full_model_name,use_image_paths=use_image_paths,pretrain=pretrain)
        #  (supervisors): ModuleDict(
    # (sent_tok): SentenceTokenSupervisor(
    #   (aggregator): GlobalAvgPool()
    # )
    # (anat_tok): AnatomyTokenSupervisor()
    # (patho_tok): PathologyTokenSupervisor()
    else:
        results = {}
        
    # if config.evaluate and not config.debug:
    #     if config.train:
    #         # Load the best model from training
    #         model = load_model_by_name(full_model_name, load_best=True,model_dir=model_dir)
    #         model = model.to(device=config.device)
    #         log.info(f'load model from {full_model_name} (load_model_by_name)')
    #     for task_name, task, task_dl, _ in build_dataloaders_for_eval(config,use_image_paths=use_image_paths,pretrain=pretrain):
    #         log.info(f'Evaluating task {task_name} ({config.eval_mode})')
    #         eval_results = validate(model=model, val_task=task, val_dl=task_dl, model_dir=model_dir, config=config, step=0, prefix=f'{config.eval_mode}_{task_name}')
    #         wandb.log(eval_results)
    #         wandb.summary.update(eval_results)
    #         results.update(eval_results)
    # 只在主进程结束wandb
    if is_main_process and not config.debug:
        wandb.finish()
        if wandb_api_path is not None:
            update_summary_with_api(results, wandb_api_path)

    return results['val_metric'] if results is not None and 'val_metric' in results else None


@hydra.main(config_path="../../conf", config_name="train_contr", version_base=None)
def do_run_training(config):
    # print(f'os.environ.get("LOCAL_RANK"):{os.environ.get("LOCAL_RANK")}')
    # device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    # torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    # log.info(f'local_rank 598行:{os.environ.get("LOCAL_RANK")}')
    # # 探测当前可用的GPU数量
    gpu_count = torch.cuda.device_count()
    log.info(f"当前可用的GPU数量: {gpu_count}")
    
    # 打印每个GPU的名称和内存信息
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # 转换为GB
        log.info(f"GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")
    run_training(config)


if __name__ == "__main__":
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--ckpt_path", type=str, required=True, help="Path to pretrained checkpoint")
    args, _ = parser.parse_known_args()
    local_rank = args.local_rank
    # DeepSpeed 初始化分布式
    deepspeed.init_distributed()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK")))

    log.info(f'check1: 621行 Current local_rank is :{local_rank}')
    import sys
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local_rank")]

    # 原有的代码
    logging.basicConfig(level=logging.DEBUG)
    cs = ConfigStore.instance()
    cs.store(name="ExperimentConfig", node=ExperimentConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
    OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver("ifel", lambda flag, val_true, val_false: val_true if flag else val_false)
    OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR)

    import model
    from model import img_encoder, txt_encoder, txt_decoder, detector
    ModelRegistry.init_registries([model, img_encoder, txt_encoder])
    # ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector])

    do_run_training()
    
    
    


# def run_training(config: ExperimentConfig):
#     seed_everything(config.seed)

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
#             model = load_model_by_name(full_model_name, load_best=True,run_name='run_2025-04-03_14-28-20')
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
