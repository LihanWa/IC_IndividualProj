
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
log = logging.getLogger(__name__)


def train(config: ExperimentConfig, model: BaseModel, model_dir: str, full_model_name: str):
    # Load the datasets for training and validation
    train_dl = build_train_dataloader(config)
    train_data_conf = train_dl.dataset.dataset_info
    val_dls={}
    
    for i, (val_name, val_task) in enumerate(config.val_tasks.items()):
        val_dls[val_name] = build_val_dataloader(config, val_task)
        # if i==0:
        #     break
    # val_dls: Dict[str, Any] = {
    #     val_name: build_val_dataloader(config, val_task)
    #     for val_name, val_task in config.val_tasks.items()
    # }

    steps_per_epoch = len(train_dl)
    log.info(f'Note: {steps_per_epoch} steps per epoch')
    assert config.max_steps is None or config.max_epochs is None
    if config.max_steps is None and config.max_epochs is not None:
        config.max_steps = config.max_epochs * steps_per_epoch
    assert config.val_freq is not None or config.val_ep is not None
    if config.val_freq is None:
        config.val_freq = steps_per_epoch * config.val_ep

    # Prepare optimizer, scheduler, and scaler
    optimizer = build_optimizer(model, config)
    lr_scheduler = build_scheduler(optimizer, config)
    # ipdb.set_trace()
    scaler = GradScaler()

    step = 0
    step_metrics_meter = AvgDictMeter()


    """ Start training """
    log.info(f'Starting training of {full_model_name}')
    log.info(f"Using {config.device}")
    if config.debug:
        torch.autograd.set_detect_anomaly(True)
    best_results = None
    pbar = tqdm(total=config.val_freq if config.val_freq is not None else steps_per_epoch * config.val_ep, desc='Training')
    epoch = 0
    while True:
        epoch_start_time = timer()
        it_start_time = timer()
        stop_training = False
        for samples in train_dl:

            pbar.update(1)

            # Training step
            output: BaseModelOutput = train_step(model, samples, optimizer, lr_scheduler, scaler, epoch, step, config, train_data_conf)
            step_metrics_meter.add(dict(output.step_metrics, loss=output.loss.detach())) #记录平均各个subloss以及总loss

            if torch.isnan(output.loss):
                log.error(f'Loss was nan: {output.step_metrics}')
                #stop_training = True
                #break

            # Increment step
            step += 1

            # Update progress bar and log losses
            if step % config.print_freq == 0 and not step % config.val_freq == 0:
                it_end_time = timer()
                it_time = it_end_time - it_start_time
                it_start_time = it_end_time
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': step_metrics_meter.compute()['loss']})

                wandb.log({'train_step/lr': lr,
                           'train_step/loss': output.loss.detach(),
                            'train_step/it_time': it_time / config.print_freq,
                           **{'train_step/' + key: value for key, value in output.step_metrics.items()}},
                          step=step)

            # Validate and log at validation frequency
            if step >= config.max_steps or step % config.val_freq == 0 or (config.val_ep is not None and step % (config.val_ep * steps_per_epoch) == 0):
                # Gather training results
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
                    val_results = validate(
                        config,
                        val_task,
                        val_dl,
                        model,
                        model_dir, 
                        step=step,
                        prefix=f'val/{val_name}',
                    )
                    results.update(val_results)
                ipdb.set_trace()
                if config.metric is None:
                    results['val_metric'] = 0.0
                elif ',' in config.metric:
                    # harmonic mean of metrics
                    val_metrics = [results[f'val/{m}'] for m in config.metric.split(',')]
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
    config: ExperimentConfig, #batchsize debug
    val_task: EvalConfig, #plot_val_samples boolean
    val_dl: DataLoader,
    model: BaseModel,
    model_dir,
    step,
    prefix=None,
    results_name=None,
    **kwargs,
):
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
    pbar = tqdm(val_dl)
    pbar.set_description(f'Validate at step {step}')

    # Iterate over batches 每个batch有4个样本，samples有四个样本
    for idx, samples in enumerate(pbar):
        # To GPU
        samples = to_device(samples, config.device)

        with torch.inference_mode():
            with torch.device(config.device):
                with autocast(device_type=config.device, enabled=False):

                    output: BaseModelOutput = evaluator.eval_step(**samples, step=step, data_config=val_data_conf)

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
                wandb_logs: dict = evaluator.plot(output_cpu, target_dir=pred_dir, max_samples=max_samples, plot_local=val_task.plot_local)

                # Log images to wandb
                # if val_task.plot_wandb:
                if val_task.plot_wandb:
                    if prefix is not None:
                        wandb_logs = {f'{prefix}/{k}': v for k, v in wandb_logs.items()}
                    wandb.log(wandb_logs, step=step)
        except Exception as e:
            log.error(f'Error plotting: {e}')
        if config.debug:
            break  # single iteration

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
    if config.continue_from_checkpoint is not None:
        # ipdb.set_trace() #config 是什么
        model = load_model_from_checkpoint(config.continue_from_checkpoint)
    else:
        model: BaseModel = instantiate_model(config.model)

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
        name=config.name,
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

    if config.train:

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
            model = load_model_by_name(full_model_name, load_best=True)
            model = model.to(device=config.device)

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
    