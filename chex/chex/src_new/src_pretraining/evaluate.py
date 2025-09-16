# import json
# import logging
# from dataclasses import dataclass, field
# from pprint import pformat, pprint,PrettyPrinter
# from typing import Any, Dict, Optional

# import hydra
# import torch
# from hydra.core.config_store import ConfigStore
# from omegaconf import MISSING, OmegaConf
# from torch import autocast
# from tqdm import tqdm
# import wandb

# import ipdb   

# from dataset.datasets import DatasetConfig
# from settings import MODELS_DIR, PROJECT_DIR, WANDB_ENTITY, WANDB_PROJECT
# from util.data_utils import to_device
# from util.model_utils import ModelRegistry, get_model_dir, get_run_dir, get_wandb_run_from_model_name, load_model_by_name
# from util.train_utils import Evaluator, ExperimentConfig, build_dataloaders_for_eval, seed_everything
# from train import validate

# log = logging.getLogger(__name__)


# @dataclass
# class EvaluationConfig: 
#     #experiment: eval_od
#     model_name: str = MISSING #chex_stage3
#     run_name: Optional[str] = None

#     eval_mode: str = 'val' #test
#     eval_tasks: Dict[str, Any] = field(default_factory=dict)
#     config_override: Dict[str, Any] = field(default_factory=dict)

#     update_wandb: bool = False
#     plot_wandb: bool = False

#     device: str = MISSING
#     num_workers: int = MISSING
#     prefetch: bool = False
#     seed: int = MISSING
#     load_best: bool = False  # loads the last by default
#     optimize_inference: bool = False #true
#     bootstrap: bool = False #True
#     n_bootstrap: int = 250

#     debug: bool = False


# def evaluate(config: EvaluationConfig):
#     seed_everything(config.seed)
#     """"""""""""""""""""""""""""""" Setup """""""""""""""""""""""""""""""
#     log.info(f'Starting Evaluation of {config.model_name}')
#     print(config.model_name)
#     model_dir = get_run_dir(get_model_dir(config.model_name), config.run_name)
#     print('model_dir: ',model_dir)
#     from transformers.utils import logging
#     logging.set_verbosity_info()
#     model, checkpoint_dict = load_model_by_name( #load model Chex 
#         config.model_name,
#         run_name=config.run_name,
#         load_best=config.load_best, 
#         return_dict=True,
#         config_override=config.config_override,
#     )
#     train_experiment_config: ExperimentConfig = OmegaConf.create(checkpoint_dict['experiment_config']) #config obtained from checkpoint
#     # with open('checkpoint.json','w') as f:
#     #     json.dump(checkpoint_dict,f,indent=4)
#     # print(f'checkpoint is loaded to checkpoint.json')
#     # ipdb.set_trace()
#     def smart_pprint(data, max_depth=2):
#         def recursive_pprint(d, depth=0):
#             if depth > max_depth:  # Only print keys when exceeding max recursion depth
#                 if isinstance(d, dict):
#                     pp.pprint({k: "..." for k in d.keys()})
#                 # else:
#                 #     pp.pprint(d)
#             else:
#                 if isinstance(d, dict):
#                     for k, v in d.items():
#                         print(f"{'  ' * depth}{k}:")
#                         recursive_pprint(v, depth + 1)
#                 # else:
#                 #     pp.pprint(d)

#         # Initialize pprint
#         pp = PrettyPrinter(indent=4)
#         # Call recursive print
#         recursive_pprint(data)
#     # smart_pprint(checkpoint_dict,max_depth=0)
#     # pprint(checkpoint_dict)
#     step = checkpoint_dict['step']
#     log.info(f'Evaluating step {step}')
#     ipdb.set_trace()
#     model = model.to(device=config.device)
#     if train_experiment_config.compile:
#         model = torch.compile(model, dynamic=True, mode='max-autotune')
#     log.info(f'Using {config.device}')
    
#     wandb_run = None
#     if not config.debug:
#         plot_wandb = any(task.plot_wandb and task.plot_val_samples > 0 for task in config.eval_tasks.values())
#         plot_wandb = plot_wandb and config.plot_wandb
#         # try:
#         #     wandb_run = get_wandb_run_from_model_name(
#         #         config.model_name,
#         #         run_name=config.run_name
#         #     )
#         #     assert wandb_run.state != 'running', 'Run is still running'
#         # except Exception as e:
#         #     log.error(f'Could not get wandb run: {e}\n'
#         #               'Evaluating without saving to wandb.')
#         #     wandb_run = None
#         if not plot_wandb:
#             try:
#                 # Try to get run from W&B
#                 wandb_run = get_wandb_run_from_model_name(
#                     config.model_name,
#                     run_name=config.run_name
#                 )
#                 assert wandb_run.state != 'running', 'Run is still running'
#             except Exception as e:
#                 # If getting run fails, log error and create new run
#                 log.error(f'Could not get wandb run: {e}\n'
#                         'Creating a new wandb run instead.')
#                 wandb_run = wandb.init(
#                     project=config.model_name,  # Use model_name as project name
#                     name=config.run_name,      # Use run_name as run name
#                 )

#         if plot_wandb:
#             if wandb.run is not None:
#                 wandb.finish()
#             wandb.init(
#                 project=WANDB_PROJECT,
#                 entity=WANDB_ENTITY,
#                 name=f'plot_{config.model_name}',
#                 tags=[type(model).__name__],
#                 dir='.',
#                 resume=False, #'must' if config.resume else False,
#                 settings=wandb.Settings(start_method='thread'), 
#             )
    
#     all_results = {}
#     train_experiment_config.prefetch = config.prefetch
#     train_experiment_config.num_workers = config.num_workers
#     train_experiment_config.device = config.device
#     train_experiment_config.debug = config.debug
#     train_experiment_config.batch_size=1 #modified
    
#     # ipdb.set_trace()
#     x=build_dataloaders_for_eval(train_experiment_config, eval_tasks=config.eval_tasks, eval_mode=config.eval_mode, load_val=config.optimize_inference)
#                                 #training params:model,para,act,scale  task:task prompt sentence     test                      False
    
#     #use yield to load different data step by step
#     # config.bootstrap=False#modified
#     # config.optimize_inference=False#modified can improve statistical reliability of evaluation results, but increases computational overhead.
#     for task_name, task_config, task_dl, task_dl_val in x: #at this point the first buildloader has formed samples in task_dl
#         # ipdb.set_trace()
#         #task_dl contains data
#         eval_prefix = f'{config.eval_mode}_{task_name}'
#         log.info(f'Evaluating task {task_name} ({config.eval_mode})')
#         # ipdb.set_trace()
#         if config.optimize_inference:
#             task_config = optimize_inference(model, task_config, task_dl_val, config, step) #优化参数 第一次评估可能需要较长时间，因为需要进行超参数的搜索和调整。后续评估可以直接复用优化结果，从而大幅提升评估速度。
#             #task_config在validate用到 
#             #与config的 task config比较，如果没变，说明更新了
#         kwargs = {}
#         # if config.bootstrap:
#         #     kwargs['bootstrap'] = True
#         #     kwargs['n_bootstrap'] = config.n_bootstrap
#         if not config.plot_wandb:
#             task_config.plot_wandb = False
#         eval_results = validate(model=model, val_task=task_config, val_dl=task_dl, 
#                                 model_dir=model_dir, config=train_experiment_config,
#                                 results_name=f'{config.eval_mode}_{task_name}',
#                                 step=step, prefix=f'{config.eval_mode}_{task_name}', **kwargs)

#         all_results.update(eval_results)
#         if wandb_run is not None and config.update_wandb:
#             wandb_run.summary.update(eval_results)
#             wandb_run.config.update({eval_prefix: {
#                 'task': task_config,
#                 'config_override': config.config_override
#             }})
#         break
#     # ipdb.set_trace()
#     log.info('Finished evaluating')
#     log.info(f'Results: {pformat(all_results)}')
    
# def optimize_inference(model, val_task, val_dl, config, step):
#     model.eval()
#     evaluator: Evaluator = model.build_evaluator(val_task)
#     if not hasattr(evaluator, 'optimize_inference'):
#         log.warning(f'Model {type(evaluator).__name__} does not support optimize_inference')
#         return None
#     val_task = evaluator.config
#     val_data_conf = val_dl.dataset.dataset_info

#     # Init progress bar
#     pbar = tqdm(val_dl)
#     pbar.set_description(f'Validate at step {step}')

#     # Iterate over batches
#     predictions = []
#     for idx, samples in enumerate(pbar):
#         # To GPU
#         samples = to_device(samples, config.device)
#         with torch.inference_mode():
#             with torch.device(config.device):
#                 with autocast(device_type=config.device, enabled=False):
#                     predictions.append(evaluator.eval_step(**samples, step=step, data_config=val_data_conf, optimize_inference=True))
#     best_eval_config = evaluator.optimize_inference(predictions)

#     model.train()
#     return best_eval_config

# @hydra.main(config_path="../conf", config_name="evaluate", version_base=None) #config的来源
# def run_evaluate(config):
#     evaluate(config)


# if __name__ == "__main__":
#     cs = ConfigStore.instance()
#     cs.store(name="EvaluationConfig", node=EvaluationConfig)
#     cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
#     OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR) #添加变量 路径

#     import model
#     from model import img_encoder, txt_encoder, txt_decoder, detector
#     ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector]) #注册多个ModelRegistry
#     OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
#     OmegaConf.register_new_resolver(
#         "ifel",
#         lambda flag, val_true, val_false: val_true if flag else val_false
#     )
#     run_evaluate()



import json
import logging
from dataclasses import dataclass, field
from pprint import pformat, pprint,PrettyPrinter
from typing import Any, Dict, Optional

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from torch import autocast
from tqdm import tqdm
import wandb

import ipdb   

from dataset.datasets import DatasetConfig
from settings import MODELS_DIR, PROJECT_DIR, WANDB_ENTITY, WANDB_PROJECT
from util.data_utils import to_device
from util.model_utils import ModelRegistry, get_model_dir, get_run_dir, get_wandb_run_from_model_name, load_model_by_name
from util.train_utils import Evaluator, ExperimentConfig, build_dataloaders_for_eval, seed_everything
from train import validate

log = logging.getLogger(__name__)


@dataclass
class EvaluationConfig: 
    #experiment: eval_od
    model_name: str = MISSING #chex_stage3
    run_name: Optional[str] = None

    eval_mode: str = 'val' #test
    eval_tasks: Dict[str, Any] = field(default_factory=dict)
    config_override: Dict[str, Any] = field(default_factory=dict)

    update_wandb: bool = False
    plot_wandb: bool = False

    device: str = MISSING
    num_workers: int = MISSING
    prefetch: bool = False
    seed: int = MISSING
    load_best: bool = False  # loads the last by default
    optimize_inference: bool = False #true
    bootstrap: bool = False #True
    n_bootstrap: int = 250

    debug: bool = False


def evaluate(config: EvaluationConfig):
    seed_everything(config.seed)
    """"""""""""""""""""""""""""""" Setup """""""""""""""""""""""""""""""
    log.info(f'Starting Evaluation of {config.model_name}')
    print(config.model_name)
    model_dir = get_run_dir(get_model_dir(config.model_name), config.run_name)
    print('model_dir: ',model_dir)
    from transformers.utils import logging
    logging.set_verbosity_info()
    model, checkpoint_dict = load_model_by_name( # Load model Chex 
        config.model_name,
        run_name=config.run_name,
        load_best=config.load_best, 
        return_dict=True,
        config_override=config.config_override,
    )
    train_experiment_config: ExperimentConfig = OmegaConf.create(checkpoint_dict['experiment_config']) # Config obtained from checkpoint
    step = checkpoint_dict['step']
    log.info(f'Evaluating step {step}')

    

    model = model.to(device=config.device)
    if train_experiment_config.compile:
        model = torch.compile(model, dynamic=True, mode='max-autotune')
    log.info(f'Using {config.device}')
    
    wandb_run = None
    if not config.debug:
        plot_wandb = any(task.plot_wandb and task.plot_val_samples > 0 for task in config.eval_tasks.values())
        plot_wandb = plot_wandb and config.plot_wandb
        try:
            wandb_run = get_wandb_run_from_model_name(
                config.model_name,
                run_name=config.run_name
            )
            assert wandb_run.state != 'running', 'Run is still running'
        except Exception as e:
            log.error(f'Could not get wandb run: {e}\n'
                      'Evaluating without saving to wandb.')
            wandb_run = None

        if plot_wandb:
            wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f'plot_{config.model_name}',
                tags=[type(model).__name__],
                dir='.',
                resume=False, #'must' if config.resume else False,
                settings=wandb.Settings(start_method='thread'), 
            )
    
    all_results = {}
    train_experiment_config.prefetch = config.prefetch
    train_experiment_config.num_workers = config.num_workers
    train_experiment_config.device = config.device
    train_experiment_config.debug = config.debug

    
    # ipdb.set_trace()
    x=build_dataloaders_for_eval(train_experiment_config, eval_tasks=config.eval_tasks, eval_mode=config.eval_mode, load_val=config.optimize_inference)
                                # Training parameters: model,para,act,scale  Task: task prompt sentence     test                      False
    
    # Use yield to load different data step by step
    # config.bootstrap=False# Modified
    # config.optimize_inference=False# Modified can improve statistical reliability of evaluation results, but increases computational overhead.
    for task_name, task_config, task_dl, task_dl_val in x: # At this point the first buildloader has formed samples in task_dl
        # ipdb.set_trace()
        # task_dl contains data
        eval_prefix = f'{config.eval_mode}_{task_name}'
        log.info(f'Evaluating task {task_name} ({config.eval_mode})')
        # ipdb.set_trace()
        if config.optimize_inference:
            task_config = optimize_inference(model, task_config, task_dl_val, config, step) # Optimize parameters. First evaluation may take longer as it requires hyperparameter search and adjustment. Subsequent evaluations can directly reuse optimization results, greatly improving evaluation speed.
            # task_config is used in validate 
            # Compare with config's task config, if unchanged, it means updated
        kwargs = {}
        if config.bootstrap:
            kwargs['bootstrap'] = True
            kwargs['n_bootstrap'] = config.n_bootstrap
        if not config.plot_wandb:
            task_config.plot_wandb = False
        eval_results = validate(model=model, val_task=task_config, val_dl=task_dl, 
                                model_dir=model_dir, config=train_experiment_config,
                                results_name=f'{config.eval_mode}_{task_name}',
                                step=step, prefix=f'{config.eval_mode}_{task_name}', **kwargs)

        all_results.update(eval_results)
        if wandb_run is not None and config.update_wandb:
            wandb_run.summary.update(eval_results)
            wandb_run.config.update({eval_prefix: {
                'task': task_config,
                'config_override': config.config_override
            }})
        break
    # ipdb.set_trace()
    log.info('Finished evaluating')
    log.info(f'Results: {pformat(all_results)}')
    
def optimize_inference(model, val_task, val_dl, config, step):
    model.eval()
    evaluator: Evaluator = model.build_evaluator(val_task)
    if not hasattr(evaluator, 'optimize_inference'):
        log.warning(f'Model {type(evaluator).__name__} does not support optimize_inference')
        return None
    val_task = evaluator.config
    val_data_conf = val_dl.dataset.dataset_info

    # Init progress bar
    pbar = tqdm(val_dl)
    pbar.set_description(f'Validate at step {step}')

    # Iterate over batches
    predictions = []
    for idx, samples in enumerate(pbar):
        # To GPU
        samples = to_device(samples, config.device)
        with torch.inference_mode():
            with torch.device(config.device):
                with autocast(device_type=config.device, enabled=False):
                    predictions.append(evaluator.eval_step(**samples, step=step, data_config=val_data_conf, optimize_inference=True))
    best_eval_config = evaluator.optimize_inference(predictions)
    model.train()
    return best_eval_config

@hydra.main(config_path="../conf", config_name="evaluate", version_base=None) # Source of config
def run_evaluate(config):
    evaluate(config)


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="EvaluationConfig", node=EvaluationConfig)
    cs.store(name="DatasetConfig", group="dataset", node=DatasetConfig)
    OmegaConf.register_new_resolver("project_dir", lambda: PROJECT_DIR) # Add variable path

    import model
    from model import img_encoder, txt_encoder, txt_decoder, detector
    ModelRegistry.init_registries([model, img_encoder, txt_encoder, txt_decoder, detector]) # Register multiple ModelRegistry
    OmegaConf.register_new_resolver("models_dir", lambda: MODELS_DIR)
    OmegaConf.register_new_resolver(
        "ifel",
        lambda flag, val_true, val_false: val_true if flag else val_false
    )
    run_evaluate()
