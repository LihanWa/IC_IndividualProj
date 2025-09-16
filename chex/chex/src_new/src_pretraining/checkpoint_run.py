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
log = logging.getLogger(__name__)




def train(config: ExperimentConfig, model: BaseModel, model_dir: str, full_model_name: str):
    # Load the datasets for training and validation

    if config.model.name =='chex_stage1_raddino':
        use_image_paths = True
    elif config.model.name == 'chex_stage1_clip':
        use_image_paths = False
    else:
        warnings.warn(f"Unknown model name: {config.model.name}")
    
    train_dl = build_train_dataloader(config, use_image_paths)
    train_data_conf = train_dl.dataset.dataset_info
    val_dls={}
    
    for i, (val_name, val_task) in enumerate(config.val_tasks.items()):
        val_dls[val_name] = build_val_dataloader(config, val_task, use_image_paths)

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
    
    lr_scheduler = build_scheduler(optimizer, config)
    # ipdb.set_trace()
    scaler = GradScaler()

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
            output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, scaler, epoch, step, config, train_data_conf)
            step_metrics_meter.add(dict(output.step_metrics, loss=output.loss.detach())) # Record average of each subloss and total loss

            if torch.isnan(output.loss):
                log.error(f'Loss was nan: {output.step_metrics}')
                #stop_training = True
                #break
                
            # Increment step
            step += 1

            # Clear GPU cache every 40 steps
            # if step % 40 == 0:
            #     torch.cuda.empty_cache()
            #     gc.collect()
            #     log.info(f"Clear GPU cache every 40 steps: Allocated {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, Reserved {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")

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
                log.info(f"GPU内存使用: 已分配 {torch.cuda.memory_allocated() / (1024 ** 3):.2f}GB, 已保留 {torch.cuda.memory_reserved() / (1024 ** 3):.2f}GB")
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # 转换为GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # 转换为GB
                wandb.log({
                    'train_step/lr': lr,
                    'train_step/loss': output.loss.detach(),
                    'gpu/memory_allocated_gb': gpu_memory_allocated,
                    'gpu/memory_reserved_gb': gpu_memory_reserved,
                    **{'train_step/' + key: value for key, value in output.step_metrics.items()}
                }, step=step)

            # Validate and log at validation frequency
            if step>int(config.warmup_steps) and (step >= config.max_steps or step % config.val_freq == 0 or (config.val_ep is not None and step % (config.val_ep * steps_per_epoch) == 0)):
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
                    
                    # 检查是否有非常小的值
                    min_threshold = 0.2  # 可以根据您的具体场景调整这个阈值 # 0.3 0.4 0.5 0.2 0.1 0.7
                    
                    # 如果有非常小的值，可以考虑以下几种方案
                    if any(m < min_threshold for m in val_metrics):
                        # 方案1：对所有指标进行缩放，避免极小值
                        # 例如，如果指标范围通常是0-1，可以将它们映射到0.01-1.01
                        # scaled_metrics = [m + min_threshold for m in val_metrics]
                        # results['val_metric'] = len(scaled_metrics) / sum(1. / m for m in scaled_metrics)
                        
                        # 方案2：使用几何平均数代替调和平均数
                        # import numpy as np
                        # results['val_metric'] = np.exp(sum(np.log(max(m, min_threshold)) for m in val_metrics) / len(val_metrics))
                        
                        # 方案3：使用算术平均数
                        results['val_metric'] = sum(val_metrics) / len(val_metrics)
                    else:
                        # 原始调和平均数计算
                        results['val_metric'] = len(val_metrics) / sum(1. / m for m in val_metrics)
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

def train_step(model: BaseModel, samples, optimizer, lr_scheduler, scaler, epoch, step, config: ExperimentConfig, train_data_conf: DatasetConfig) -> BaseModelOutput:
    # Forward
    samples = to_device(samples, config.device)
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

    # Update
    if (step + 1) % config.accumulation_steps == 0:
        if config.grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config.grad_clip_norm)
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
                with autocast(device_type=config.device, enabled=False):
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
                for key, value in wandb_logs.items():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], wandb.Image):
                        print(f"Key: {key}, Number of images: {len(value)}")

                if val_task.plot_wandb:
                    if prefix is not None:
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


def run_training(config: ExperimentConfig):
    seed_everything(config.seed)

    # Prepare model dir
    override_dirname = HydraConfig.get().job.override_dirname
    # ipdb.set_trace()
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
    if config.continue_from_checkpoint is not None: #stage1 none
        model = load_model_from_checkpoint(config.continue_from_checkpoint)
    else:
        model: BaseModel = instantiate_model(config.model) #config.model:chex

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
        name='chex-stage1-raddino1',
        tags=[type(model).__name__],
        dir='.',
        config=OmegaConf.to_container(config, resolve=True, throw_on_missing=False),
        resume=False, #'must' if config.resume else False,
        reinit=True,  # without reinit there may be problems when running Hydra sweeps
        settings=wandb.Settings(start_method='thread', init_timeout=300, _service_wait=300),  # fork TODO: thread? -> for hydra sweeps
        mode="disabled" if config.debug else "online", 
    )
    wandb.summary['n_parameters'] = n_parameters
    log.info(f'Number of params: {n_parameters}')
    wandb_api_path = str(wandb.run.path)
    log.info(f'API path: {wandb_api_path}')

    if config.compile and not config.debug:
        model = torch.compile(model, dynamic=True, mode='max-autotune')
        # train = torch.compile(train, dynamic=True, mode='max-autotune')
        # validate = torch.compile(validate, dynamic=True, mode='max-autotune')

    if config.train: #train and val

        results = train(config, model=model, model_dir=model_dir, full_model_name=full_model_name)
        #  (supervisors): ModuleDict(
    # (sent_tok): SentenceTokenSupervisor(
    #   (aggregator): GlobalAvgPool()
    # )
    # (anat_tok): AnatomyTokenSupervisor()
    # (patho_tok): PathologyTokenSupervisor()
    else:
        results = {}
        
    if config.evaluate and not config.debug:
        if config.train:
            # Load the best model from training
            model = load_model_by_name(full_model_name, load_best=True,model_dir=model_dir)
            model = model.to(device=config.device)
            log.info(f'load model from {full_model_name} (load_model_by_name)')
        for task_name, task, task_dl, _ in build_dataloaders_for_eval(config):
            log.info(f'Evaluating task {task_name} ({config.eval_mode})')
            eval_results = validate(model=model, val_task=task, val_dl=task_dl, model_dir=model_dir, config=config, step=0, prefix=f'{config.eval_mode}_{task_name}')
            wandb.log(eval_results)
            wandb.summary.update(eval_results)
            results.update(eval_results)
    wandb.finish()
    #update_summary_with_api(results, wandb_api_path)

    return results['val_metric'] if results is not None and 'val_metric' in results else None


@hydra.main(config_path="../conf", config_name="train", version_base=None)
def do_run_training(config):
    run_training(config)


if __name__ == "__main__":
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

    do_run_training()
    
    