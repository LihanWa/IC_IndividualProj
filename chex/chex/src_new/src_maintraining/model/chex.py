from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import einops
from omegaconf import MISSING
from model.img_encoder.chexzero_img_encoder import ImageEncoder, ImageEncoderConfig
from model.txt_decoder.ptuned_decoder import PTunedDecoderConfig, PTunedDecoderModel
from model.txt_encoder.chexzero_txt_encoder import ChexzeroTextEncoder, ChexzeroTextEncoderConfig

import torch.nn.functional as F
import torch
import numpy as np
from deepdiff import DeepDiff

from metrics.training_metrics import TrainingMetrics
from model.components.bbox_prediction import clip_bboxes
from model.components.classification_losses import classify_features
from model.detector.token_decoder_detector import PostDecoder, TokenDetectorOutput, TokenDetector, TokenDetectorConfig
from model.eval.anat_explainer import AnatomyExplainerEvaluator
from model.eval.box_explainer import BoxExplainerEvaluator
from model.eval.pathology_detection import PathologyDetectionEvaluator
from model.eval.report_generation import ReportEvaluator
from model.eval.sentence_grounding import SentenceGroundingEvaluator
from model.img_encoder import ImageEncoderOutput
from model.supervisors.anatomy_token_supervisor import AnatomyTokenConfig, AnatomyTokenSupervisor
from model.supervisors.patho_token_supervisor import PathologyTokenConfig, PathologyTokenSupervisor
from model.supervisors.sentence_token_supervisor import SentenceTokenConfig, SentenceTokenSupervisor
from model.txt_encoder import TextEncoderOutput
from torch import BoolTensor, FloatTensor, LongTensor, Tensor, nn
from util.plot_grounding import plot_grounding

from util.model_utils import BaseModel, BaseModelOutput, MainModelConfig, load_model_by_name
from util.plot_utils import wandb_plot_text
from util.prompt_utils import fill_prompt_templates, flatten_prompts, flatten_prompts_2, localized_prompt_templates
from util.train_utils import EvalConfig, Evaluator
import ipdb
import tracemalloc

log = logging.getLogger(__name__)

@dataclass
class ImgTextTrainOutput(BaseModelOutput):
    sample_id: List[str] = field(default_factory=list)
    x: Tensor = MISSING
    sentences: List[List[str]] = MISSING

    encoded_img: ImageEncoderOutput = MISSING
    encoded_sentences: TextEncoderOutput = MISSING

    grounding: TokenDetectorOutput = MISSING

    generated_sentences: Optional[List[List[str]]] = None

    @property
    def device(self):
        return self.x.device
    
    @property
    def N(self):
        return len(self.sample_id)


@dataclass
class ChEXConfig(MainModelConfig):
    # --- Model ---
    name: str = MISSING
    img_encoder: ImageEncoderConfig = MISSING
    txt_encoder: ChexzeroTextEncoderConfig = MISSING
    detector: TokenDetectorConfig = MISSING
    txt_decoder: Optional[PTunedDecoderConfig] = None

    load_components_from: Dict[str, Optional[str]] = field(default_factory=dict)
    freeze_loaded_components: List[str] = field(default_factory=list)

    # --- Post-decoder layers ---
    n_post_decoder_layers: int = 0
    post_decoder_patches: bool = False
    post_decoder_cls: bool = False
    post_decoder_droppath: bool = False
    post_decoder_gate: bool = False
    post_decoder_multibox_drop: float = 0.0

    # --- Prompts ---
    anatomy_prompts: Dict[str, List[str]] = field(default_factory=dict)
    pathology_pos_prompts: Dict[str, List[str]] = field(default_factory=dict)
    pathology_neg_prompts: Dict[str, List[str]] = field(default_factory=dict)
    no_finding_prompts: List[str] = field(default_factory=list)
    pathology_neg_prompt_mode: str = 'neg'  # neg, pos_centroid, no_finding
    randomize_prompts: bool = False

    # --- Supervisors (for sentence / anatomy / pathology tokens) ---
    sent_tok: Optional[SentenceTokenConfig] = None
    anat_tok: Optional[AnatomyTokenConfig] = None
    patho_tok: Optional[PathologyTokenConfig] = None

    max_generated_sentences: Optional[int] = 128
    max_sentences_full_sample: bool = False

    cache_encoded_sentences: bool = True    
    drop_sentence_prob: float = 0.0


def memory_monitor(func):
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats()
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB")
        return result
    return wrapper

class ChEX(BaseModel):
    CONFIG_CLS = ChEXConfig

    def __init__(self, config: ChEXConfig, from_checkpoint=False) -> None:
        self.config: ChEXConfig
        super().__init__(config)

        # --- Components ---
        self.img_encoder = ImageEncoder(self.config.img_encoder, main_config=self.config)
        self.txt_encoder = ChexzeroTextEncoder(self.config.txt_encoder, main_config=self.config)
        self.detector = TokenDetector(self.config.detector, main_config=self.config)
        # ipdb.set_trace()
        if self.config.txt_decoder is not None:
            self.txt_decoder = PTunedDecoderModel(self.config.txt_decoder, main_config=self.config)
            self.has_txt_decoder = True
        else:
            self.has_txt_decoder = False
    
        if self.config.n_post_decoder_layers > 0:
            self.post_decoder = PostDecoder(
                n_post_decoder_layers=self.config.n_post_decoder_layers,
                attend_to_patches=self.config.post_decoder_patches,
                attend_to_cls=self.config.post_decoder_cls,
                enc_dec_droppath=self.config.post_decoder_droppath,
                gate=self.config.post_decoder_gate,
                multibox_drop_prob=self.config.post_decoder_multibox_drop,
                main_config=config
            )
        else:
            self.post_decoder = None

        # --- Supervisors (for sentence / anatomy / pathology tokens) ---
        supervisors = {}
        if self.config.sent_tok is not None:
            from .supervisors.sentence_token_supervisor import SentenceTokenSupervisor
            supervisors['sent_tok'] = SentenceTokenSupervisor(self.config.sent_tok, self.config)
        # if self.config.anat_tok is not None:
        #     from .supervisors.anatomy_token_supervisor import AnatomyTokenSupervisor
        #     supervisors['anat_tok'] = AnatomyTokenSupervisor(self.config.anat_tok, self.config)
        # if self.config.patho_tok is not None:
        #     from .supervisors.patho_token_supervisor import PathologyTokenSupervisor
        #     supervisors['patho_tok'] = PathologyTokenSupervisor(self.config.patho_tok, self.config)
        self.requires_sentence_tokens = any(sup.requires_sentence_tokens for sup in supervisors.values())
        self.requires_anatomy_tokens = any(sup.requires_anatomy_tokens for sup in supervisors.values())
        self.requires_pathology_tokens = any(sup.requires_pathology_tokens for sup in supervisors.values())
        self.requires_region_pathology_tokens = any(sup.requires_region_pathology_tokens for sup in supervisors.values())

        self.supervisors = nn.ModuleDict(supervisors)
            
       # ----- Load components from other models -----
        # ipdb.set_trace()# whether passed through
        if not from_checkpoint:  # ignore loading if this model is loaded from a checkpoint; skip for eval
            for component_name, model_name in self.config.load_components_from.items():
                if model_name is None:
                    continue
                assert hasattr(self, component_name), f'Component {component_name} does not exist'
                component = getattr(self, component_name)
                model = load_model_by_name(model_name, return_dict=False)
                assert hasattr(model, component_name), f'Component {component_name} does not exist in model {model_name}'
                loaded_component = getattr(model, component_name)
                assert type(component) == type(loaded_component), f'Component {component_name} has type {type(component)} but loaded component has type {type(loaded_component)}'
                if hasattr(loaded_component, 'config') and not component.config == loaded_component.config:
                    diff = DeepDiff(loaded_component.config, component.config, verbose_level=2)
                    log.warning(f'Component {component_name} has different config than loaded component: {diff}')

                component.load_state_dict(loaded_component.state_dict())
                if component_name in self.config.freeze_loaded_components:
                    log.info(f'Freezing component {component_name}')
                    for param in component.parameters():
                        param.requires_grad = False
    
    # @memory_monitor
    def forward(self, 
                x: FloatTensor, 
                sample_id: List[str],
                epoch=None,
                data_config=None,
                no_patho_token_supervision=False,
                no_anat_token_supervision=False,
                no_sent_token_supervision=False,
                **kwargs): #samples
        
        # Add memory tracking at main entry point
        # tracemalloc.start()

        # ---------- Encode the image features (i.e. patches) and all required prompts ----------
        encoded_image: ImageEncoderOutput = self.img_encoder(x, epoch=epoch,sample_id=sample_id)
        
        assert data_config is not None
        if self.requires_sentence_tokens and not no_sent_token_supervision:
            assert 'sentences' in kwargs, 'Sentences must be provided in the dataset if sentence tokens are required'
            sentences = kwargs['sentences']
            if self.config.drop_sentence_prob > 0.0 and self.training:
                tmp=sentences.copy()
                sentences = [[s for s in sent if np.random.rand() > self.config.drop_sentence_prob] if sent is not None else sent for sent in sentences]
                # Check each sentence list, if length is 0 after drop, keep original sentence list
                sentences = [sent_list if sent_list is None or len(sent_list) > 0 else tmp[i] for i, sent_list in enumerate(sentences)]
            # Check if sentences is empty list
            if not sentences or all(sent is None or len(sent) == 0 for sent in sentences):
                print("Warning: sentences is empty list or contains only empty elements")
                print("Original sentences content:", kwargs['sentences'])
                # ipdb.set_trace()
                encoded_sentences = None
            else:
                encoded_sentences: TextEncoderOutput = self.encode_sentences(sentences, device=encoded_image.device, epoch=epoch)
        else:
            encoded_sentences = None
        # if 'validate' in kwargs and kwargs['validate'] and list(kwargs['task'].dataset.values())[0].name.startswith('mimic_cxr-'):
        #     if encoded_sentences is not None:
        #         sentence_features = encoded_sentences.sentence_features
        #         # (N x S)
        #         sentence_mask = encoded_sentences.sentence_mask
        #         sentences=encoded_sentences.sentences
        #         flattened_sentences=encoded_sentences.flattened_sentences

        #         # has_sentences=kwargs['has_sentences']
        #         # if has_sentences is None:
        #         has_sentences = encoded_sentences.sentence_mask.any(dim=-1)
        #         # if has_sentences is not None:
        #         encoded_image = encoded_image[has_sentences]
        #         sentence_features = sentence_features[has_sentences]
        #         sentence_mask = sentence_mask[has_sentences]
        #         # Fix: convert boolean tensor to list index
        #         has_sentences_indices = has_sentences.nonzero().squeeze(-1).cpu().tolist()
        #         sample_id = [sample_id[i] for i in has_sentences_indices]
        #         sentences=[sentences[i] for i in has_sentences_indices]
        #         flattened_sentences=[flattened_sentences[i] for i in has_sentences_indices]
        #         encoded_sentences=TextEncoderOutput(
        #             sentence_features=sentence_features,
        #             sentence_mask=sentence_mask,
        #             sentences=sentences,
        #             flattened_sentences=flattened_sentences
        #         )
        #         # ---------- Prompt detection ----------
        #         sentence_grounding_output: TokenDetectorOutput = \
        #                 self.detect_prompts(
        #                     encoded_image,
        #                     box_prompts_emb=sentence_features, # (N x S x d)
        #                     box_prompt_mask=sentence_mask,
        #                     skip_roi_pool=True) # (N x S)

                
        #     else: 
        #         sentence_grounding_output=None
            
        #     output = ImgTextTrainOutput( #dict
        #         sample_id=sample_id,
        #         x=x, 
        #         encoded_img=encoded_image,    #img encoder
        #         encoded_sentences=encoded_sentences,
        #         grounding= sentence_grounding_output
        #     )

                
                
                
        if self.requires_pathology_tokens:
            assert no_patho_token_supervision or (data_config.class_names is not None and len(data_config.class_names) > 0) , 'Pathology names must be provided in the dataset if pathology tokens are required'
            assert no_patho_token_supervision or len(self.config.pathology_pos_prompts) > 0, 'Pathology prompts must be provided in the model config if pathology tokens are required'
            class_names = data_config.class_names if data_config.class_names is not None else []
            # ipdb.set_trace() #how large is data_config classnames
            # (C x d), (C x d)
            patho_pos_prompt_emb, patho_neg_prompt_emb = self.encode_pos_neg_prompts(
                self.config.pathology_neg_prompt_mode, class_names,
                self.config.pathology_pos_prompts, self.config.pathology_neg_prompts, self.config.no_finding_prompts,
                epoch=epoch, random_synonym=self.config.randomize_prompts)
        else:
            patho_pos_prompt_emb = patho_neg_prompt_emb = None

        if self.requires_anatomy_tokens:
            assert no_anat_token_supervision or (data_config.anatomy_names is not None and len(data_config.anatomy_names) > 0), 'Anatomy names must be provided in the dataset if anatomy tokens are required'
            assert no_anat_token_supervision or len(self.config.anatomy_prompts) > 0, 'Anatomy prompts must be provided in the model config if anatomy tokens are required'
            anatomy_names = data_config.anatomy_names if data_config.anatomy_names is not None else []
            if data_config.multi_anatomy_names is not None and len(data_config.multi_anatomy_names) > 0:
                anatomy_names = anatomy_names + data_config.multi_anatomy_names
            anatomy_token_emb, _ = self.encode_prompts(self.config.anatomy_prompts, anatomy_names, epoch=epoch, random_synonym=self.config.randomize_prompts)
        else:
            anatomy_token_emb = None

        if self.requires_region_pathology_tokens:
            anatomy_names = data_config.anatomy_names if data_config.anatomy_names is not None else []
            A = len(anatomy_names)
            class_names = data_config.class_names if data_config.class_names is not None else []
            C = len(class_names)
            pos_prompts = {
                f'{anat_name}_{patho_name}': [f'{patho_prompt} in {anat_prompt}' for patho_prompt in self.config.pathology_pos_prompts[patho_name] for anat_prompt in self.config.anatomy_prompts[anat_name]]
                for anat_name in anatomy_names for patho_name in class_names
            }
            neg_prompts = {
                f'{anat_name}_{patho_name}': [f'{patho_prompt} in {anat_prompt}' for patho_prompt in self.config.pathology_neg_prompts[patho_name] for anat_prompt in self.config.anatomy_prompts[anat_name]]
                for anat_name in anatomy_names for patho_name in class_names
            }
            names = [f'{anat_name}_{patho_name}' for anat_name in anatomy_names for patho_name in class_names]
            # (A*C x d), (A*C x d)
            region_pos_prompt_emb, region_neg_prompt_emb = self.encode_pos_neg_prompts(
                self.config.pathology_neg_prompt_mode, names, pos_prompts, neg_prompts, self.config.no_finding_prompts,
                epoch=epoch, random_synonym=self.config.randomize_prompts)
            region_pos_prompt_emb = region_pos_prompt_emb.view(A, C, -1)
            region_neg_prompt_emb = region_neg_prompt_emb.view(A, C, -1)
        else:
            region_pos_prompt_emb = region_neg_prompt_emb = None


        # ---------- Supervisors ----------
        loss = 0.
        step_metrics = {}
        # if not ('validate' in kwargs and kwargs['validate'] and list(kwargs['task'].dataset.values())[0].name.startswith('mimic_cxr-')):
        output = ImgTextTrainOutput( #dict
            sample_id=sample_id,
            x=x, 
            encoded_img=encoded_image,    #img encoder
            encoded_sentences=encoded_sentences,
        )
        for supervisor in self.supervisors.values():
            if (no_patho_token_supervision and isinstance(supervisor, PathologyTokenSupervisor)) or \
                (no_anat_token_supervision and isinstance(supervisor, AnatomyTokenSupervisor)) or \
                (no_sent_token_supervision and isinstance(supervisor, SentenceTokenSupervisor)):
                continue
            
            # all vindr dataset
            # Check special cases for SentenceTokenSupervisor
            # if isinstance(supervisor, SentenceTokenSupervisor) and encoded_sentences is None:
            #     print(kwargs['dataset_name'][0])
            #     log.warning("SentenceTokenSupervisor requires encoded_sentences, but current dataset does not provide sentences. Skipping this supervisor.")
            #     continue
            # if isinstance(supervisor, AnatomyTokenSupervisor) and encoded_sentences is None:
            #     print(x)
            #     log.warning("Vindr does not have anatomy information. AnatomyTokenSupervisor ")
            #     continue
            
            #all vindr dataset
            
            if 'dataset_name' in kwargs: #train
                all_vindr=True
                for dataname in kwargs['dataset_name']: 
                    if dataname!='vindrcxr': all_vindr=False
                if (isinstance(supervisor, SentenceTokenSupervisor) or isinstance(supervisor, AnatomyTokenSupervisor)) and all_vindr:
                    # print(x)
                    ipdb.set_trace()
                    log.warning("Vindr does not have sentence or anatomy information. Skipping this supervisor.")
                    continue
            #all mimic dataset
            if 'dataset_name' in kwargs:
                all_mimic=True
                for dataname in kwargs['dataset_name']: 
                    if dataname!='mimic_cxr-frontal-report-cig_anatboxes-cig_anatlabels-cig_anatphrases': all_mimic=False
                if isinstance(supervisor, PathologyTokenSupervisor) and all_mimic:
                    # print(x)
                    log.warning("Mimic does not have pathology information. Skipping this supervisor.")
                    continue


            sup_loss, sup_metrics, sup_outputs = supervisor(
                self,
                encoded_image=encoded_image, encoded_sentences=encoded_sentences,
                patho_pos_prompt_emb=patho_pos_prompt_emb, patho_neg_prompt_emb=patho_neg_prompt_emb,
                region_pos_prompt_emb=region_pos_prompt_emb, region_neg_prompt_emb=region_neg_prompt_emb,
                anatomy_token_emb=anatomy_token_emb,
                epoch=epoch, **kwargs)
            

                
            loss += sup_loss
            step_metrics.update({key: value.detach() for key, value in sup_metrics.items()})
            for k, v in sup_outputs.items():
                setattr(output, k, v)
            #     model: 'ChEX',
            #     encoded_image: ImageEncoderOutput, 
            #     patho_pos_prompt_emb: Optional[torch.FloatTensor] = None,
            #     patho_neg_prompt_emb: Optional[torch.FloatTensor] = None,
            #     has_class_bboxes: Optional[torch.BoolTensor] = None,
            #     target_cls_boxes_padded: Optional[torch.FloatTensor] = None,
            #     target_cls_boxes_mask: Optional[torch.BoolTensor] = None,
            #     has_class_labels: Optional[torch.BoolTensor] = None,
            #     target_cls_labels: Optional[torch.FloatTensor] = None,
            #     **kwargs
        
        # Ensure loss is tensor type
        # print(f'The loss type is {type(loss)}')
        # if isinstance(loss, float):
        #     loss = torch.tensor(loss, device=encoded_image.device, requires_grad=True)
        output.loss = loss
        output.step_metrics = step_metrics

        # Add memory tracking at main entry point
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("[ Top 10 memory users ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        return output


    def encode_sentences(self, sentences: List[List[str]], device, epoch=None) -> TextEncoderOutput:
        flattened_sentences, sentence_mask = flatten_prompts(sentences, device=device)
        flat_sentence_features = self.txt_encoder.encode_sentences(flattened_sentences, cache=self.config.cache_encoded_sentences, epoch=epoch)
        N, S = sentence_mask.shape
        d = flat_sentence_features.shape[-1]
        # (N x S x d)
        sentence_features = flat_sentence_features.new_zeros((N, S, d))
        sentence_features[sentence_mask] = flat_sentence_features
        return TextEncoderOutput(
            sentence_features=sentence_features, # (N x S x d)
            sentence_mask=sentence_mask, # (N x S)
            sentences=sentences,
            flattened_sentences=flattened_sentences
        )
    
    def encode_region_sentences(self, region_sentences: List[List[List[str]]], device, epoch=None) -> TextEncoderOutput:
        flattened_sentences, sentence_mask = flatten_prompts_2(region_sentences, device=device)
        flat_sentence_features = self.txt_encoder.encode_sentences(flattened_sentences, cache=self.config.cache_encoded_sentences, epoch=epoch)
        N, R, S = sentence_mask.shape
        d = flat_sentence_features.shape[-1]
        # (N x S x d)
        sentence_features = flat_sentence_features.new_zeros((N, R, S, d))
        sentence_features[sentence_mask] = flat_sentence_features
        return TextEncoderOutput(
            sentence_features=sentence_features, # (N x S x d)
            sentence_mask=sentence_mask, # (N x S)
            sentences=region_sentences,
            flattened_sentences=flattened_sentences
        )
    
    def train_sentence_decoder(self, flattened_features, sentences: List[List[str]], sentence_mask, epoch):
        """
        :param flattened_features: (N_sent x d)
        :param sentences: List[List[str]] of length N with lists of lengths S_i <= S
        :param sentence_mask: (N x S) with N_sent true values
        :param 
        """
        N, S = sentence_mask.shape
        sentence_mask = sentence_mask.clone()
        
        # (N_sent_flat)
        flattened_sentences: List[str] = [s for sents in sentences for s in sents]
        M_is: List[int] = [len(sents) for sents in sentences]
        # (N x S)
        flattened_sent_mask = torch.zeros(N, S, dtype=torch.bool, device=flattened_features.device)
        for i, M_i in enumerate(M_is):
            flattened_sent_mask[i, :M_i] = True

        # (N_sent)
        if self.config.max_generated_sentences is not None and flattened_features.shape[0] > self.config.max_generated_sentences:
            N_generated = self.config.max_generated_sentences
            
            if self.config.max_sentences_full_sample:
                # (N)
                sample_sentence_counts = sentence_mask.sum(dim=1)
                # (N)
                sample_mask = sample_sentence_counts.cumsum(dim=0) <= N_generated
                # (N x S)
                gen_sent_mask = sample_mask[:, None] & sentence_mask
                gen_count = gen_sent_mask.sum()
                # (N_sent)
                gen_mask = gen_sent_mask.new_zeros(flattened_features.shape[0], dtype=torch.bool, device=flattened_features.device)
                gen_mask[:gen_count] = True
            else:
                # (N_sent)
                gen_mask = torch.zeros(flattened_features.shape[0], dtype=torch.bool, device=flattened_features.device)
                gen_mask.scatter_(0, torch.multinomial(torch.ones(flattened_features.shape[0], device=flattened_features.device), N_generated, replacement=False), True)
            flattened_features = flattened_features[gen_mask]
            
            sentence_mask[sentence_mask.clone()] = gen_mask

        # (sum(M_i))
        target_sentences_mask = sentence_mask[flattened_sent_mask]
        sentence_gen_losses = self.txt_decoder.train_step(flattened_features, flattened_sentences, target_sentences_mask=target_sentences_mask, epoch=epoch)
        return sentence_gen_losses
    
    # @memory_monitor
    def generate(self, features: Tensor, feature_mask, **generation_kwargs) -> List[List[str]]:
        """
        :param features: (N x S x d) or (N x d)
        :param feature_mask: (N x S) or None
        :return: List[List[str]] of length N with lists of lengths S_i <= S or List[str] of length N
        """
        # ipdb.set_trace() #check how many N, who is prefix
        assert features.ndim in (2, 3)
        N = features.shape[0]
        sent_per_sample = features.shape[1] if features.ndim == 3 else 1
        #self.config.max_generated_sentences = 32
        N_batch = self.config.max_generated_sentences // sent_per_sample if self.config.max_generated_sentences is not None else N # 32//3=10
        N_batch = max(N_batch, 1) # 10
        batched_features: List[torch.Tensor] = features.split(N_batch, dim=0) # features too few 4 items ([4, 3, 512],)
        batched_feature_mask: List[torch.Tensor] = feature_mask.split(N_batch, dim=0) if feature_mask is not None else [None] * len(batched_features)
        all_generated_sentences = []
        # ipdb.set_trace() #what is S, how is prefix formed?
        for features_batch, feature_mask_batch in zip(batched_features, batched_feature_mask): #mask because some Q quantities don't reach M=3
            flattened_features = features_batch[feature_mask_batch] if feature_mask_batch is not None else features_batch.view(-1, features_batch.shape[-1])
            #[4, 3, 512] becomes [9, 512] because some masks are False, 12 becomes 9
            if len(flattened_features) == 0:
                generated_sentences = []
            else:
                generated_sentences: List[str] = self.txt_decoder.generate(flattened_features, **generation_kwargs)
            # ipdb.set_trace()
            if features_batch.ndim == 2 and feature_mask_batch is None:
                all_generated_sentences.extend(generated_sentences)

            if feature_mask_batch is not None:
                assert feature_mask_batch.ndim == 2
                N_b, S = feature_mask_batch.shape
                # (N)
                sentence_end_indixes = torch.cumsum(feature_mask_batch.sum(dim=1), dim=0).cpu()
                sentence_start_indixes = torch.cat([torch.zeros(1, dtype=torch.long, device='cpu'), sentence_end_indixes[:-1]])
            elif features_batch.ndim == 3:
                N_b, S, d = features_batch.shape
                sentence_start_indixes = torch.arange(0, N * S, S, device='cpu')
                sentence_end_indixes = torch.arange(S, N * S + 1, S, device='cpu')
            assert len(sentence_start_indixes) == len(sentence_end_indixes) == N_b, f"{len(sentence_start_indixes)} != {len(sentence_end_indixes)} != {N_b}"

            all_generated_sentences.extend([generated_sentences[start:end] for start, end in zip(sentence_start_indixes.numpy(), sentence_end_indixes.numpy())])

        return all_generated_sentences

    def encode_pos_neg_prompts(
            self, 
            neg_prompt_mode: str,  # neg, pos_centroid, neg_centroid, posneg_centroid, no_finding, posnofind_centroid, all_centroid
            class_names: List[str], 
            pos_prompts_by_class: Dict[str, List[str]]=None, 
            neg_prompts_by_class: Optional[Dict[str, List[str]]]=None,
            no_finding_prompts: Optional[List[str]]=None, 
            region_templates: Optional[Dict[str, List[str]]]=None,
            epoch=None,
            random_synonym=False):
        # (C x d) or (C x R x d)
        pos_prompt_emb, _ = self.encode_prompts(pos_prompts_by_class, class_names, region_templates=region_templates, epoch=epoch, random_synonym=random_synonym)

        assert neg_prompt_mode in ['neg', 'pos_centroid', 'neg_centroid', 'no_finding', 'posneg_centroid', 'posnofind_centroid', 'negnofind_centroid', 'all_centroid']
        if neg_prompt_mode in ['neg', 'neg_centroid', 'posneg_centroid', 'negnofind_centroid', 'all_centroid']:
            assert neg_prompts_by_class is not None
            # (C x d) or (C x R x d)
            neg_prompt_emb, _ = self.encode_prompts(neg_prompts_by_class, class_names, region_templates=region_templates, epoch=epoch, random_synonym=random_synonym)
        if neg_prompt_mode in ['no_finding', 'posnofind_centroid', 'negnofind_centroid', 'all_centroid']:
            assert no_finding_prompts is not None
            no_finding_prompts = {'no_finding': no_finding_prompts}
            no_finding_region_templates = {'no_finding': ['{}']} if region_templates is not None else None
            # (1 x d) or (1 x 1 x d)
            nofind_prompt_emb, _ = self.encode_prompts(no_finding_prompts, ['no_finding'], region_templates=no_finding_region_templates, epoch=epoch, random_synonym=random_synonym)
            if region_templates is not None:
                C, R, d = pos_prompt_emb.shape
                # (1 x R x d)
                nofind_prompt_emb = nofind_prompt_emb.expand(1, R, d)
        
        if neg_prompt_mode == 'pos_centroid':
            pos_centroid = pos_prompt_emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = pos_centroid.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'neg_centroid':
            neg_centroid = neg_prompt_emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = neg_centroid.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'no_finding':
            neg_prompt_emb = nofind_prompt_emb.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'posneg_centroid':
            emb = torch.cat([pos_prompt_emb, neg_prompt_emb], dim=0)
            centroid = emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = centroid.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'posnofind_centroid':
            emb = torch.cat([pos_prompt_emb, nofind_prompt_emb], dim=0)
            centroid = emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = centroid.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'negnofind_centroid':
            emb = torch.cat([neg_prompt_emb, nofind_prompt_emb], dim=0)
            centroid = emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = centroid.expand(pos_prompt_emb.shape).contiguous()
        elif neg_prompt_mode == 'all_centroid':
            emb = torch.cat([pos_prompt_emb, neg_prompt_emb, nofind_prompt_emb], dim=0)
            centroid = emb.mean(dim=0, keepdim=True)
            neg_prompt_emb = centroid.expand(pos_prompt_emb.shape).contiguous()
        else:
            assert neg_prompt_mode == 'neg', neg_prompt_mode

        return pos_prompt_emb, neg_prompt_emb

    def encode_prompts(self, prompts_by_class: Dict[str, List[str]], class_names: List[str], region_templates: Optional[Dict[str, List[str]]]=None, epoch=None, random_synonym: bool = False):
        if len(class_names) == 0:
            return None, None

        # outer list: classes, inner list: synonym prompts
        prompts: List[List[str]] = [fill_prompt_templates(prompts_by_class[name]) for name in class_names]
        if random_synonym:
            # for each class (outer list), randomly select one of the synonym prompts (inner list)
            synonym_indices: List[int] = [np.random.randint(len(p)) for p in prompts]
            prompts = [[p[i]] for p, i in zip(prompts, synonym_indices)]

        if region_templates is None:
            flattened_prompts, prompt_mask = flatten_prompts(prompts, device=None) #List[str]
        else:
            # outer list: classes, inner list: regions
            region_templates: List[List[str]] = [region_templates[name]
                                                 for name in class_names]
            # outer list: classes, inner list: regions, inner inner list: synonym prompts
            prompts: List[List[List[str]]] = [localized_prompt_templates(cls_prompts, cls_reg_templates)
                                              for cls_prompts, cls_reg_templates in zip(prompts, region_templates)]
            flattened_prompts, prompt_mask = flatten_prompts_2(prompts, device=None)

        # (C*P x d) or (C*R*P x d)
        flat_prompt_embeddings = self.txt_encoder.encode_sentences(flattened_prompts, cache=True, epoch=epoch)
        prompt_mask = prompt_mask.to(device=flat_prompt_embeddings.device)
        d = flat_prompt_embeddings.shape[-1]
        # (C x P x d) or (C x R x P x d)
        prompt_embeddings = flat_prompt_embeddings.new_zeros((*prompt_mask.shape, d))
        prompt_embeddings[prompt_mask] = flat_prompt_embeddings
        # (C x d) or (C x R x d)
        prompt_embeddings = (prompt_mask[..., None] * prompt_embeddings).sum(dim=-2) / prompt_mask[..., None].sum(dim=-2).clamp(min=1)
        
        return prompt_embeddings, prompt_mask.any(dim=-1)


    def detect_prompts(self,  # img->post decoder
        x: Union[FloatTensor, ImageEncoderOutput], 
        box_prompts_emb: FloatTensor, box_prompt_mask: Optional[FloatTensor] = None, 
        clip_boxes: bool = False, skip_roi_pool: Optional[bool] = None, use_post_decoder: bool =True) -> TokenDetectorOutput:
        """
        :param x: images (N x H x W)
        :param box_prompts_emb: (Q x d) or (N x Q x d)
        :param box_prompt_mask: (N x Q x d)
        """
        # multi_region: bool = False -> if false and model uses multi-region, then only use highest scoring region per prompt
        if isinstance(x, ImageEncoderOutput):
            # raise ValueError('ImageEncoderOutput is not supported for region encoding')
            encoded_image = x
            
        else:
            encoded_image: ImageEncoderOutput = self.img_encoder(x)
        N = encoded_image.N
        if box_prompt_mask is not None and box_prompt_mask.ndim == 1:
            box_prompt_mask = einops.repeat(box_prompt_mask, 'q -> n q', n=N)
        
        detection_results: TokenDetectorOutput = self.detector(
            encoded_image=encoded_image,
            query_tokens=box_prompts_emb,
            query_mask=box_prompt_mask,
            skip_roi_pool=skip_roi_pool)
        # ipdb.set_trace() #meis
        if clip_boxes:
            detection_results.boxes = clip_bboxes(detection_results.boxes)
            if detection_results.multiboxes is not None:
                detection_results.multiboxes = clip_bboxes(detection_results.multiboxes)

        if self.post_decoder is not None and use_post_decoder:
            detection_results.box_features, detection_results.multiboxes_features = self.post_decoder(encoded_image, detection_results.box_features, detection_results.multiboxes_features, box_prompt_mask)

        return detection_results
        
    def encode_regions(self, #image, box -> postdecoder output
        x: FloatTensor, 
        region_boxes: Union[FloatTensor, List[FloatTensor]], region_mask: Optional[BoolTensor] = None, region_prompt_emb=None, 
        use_post_decoder: bool = True) -> FloatTensor:
        """
        x: images (N x H x W)
        region_boxes: (N x M x 4) or list of length (M_i x 4)
        region_mask: (N x M), optional, not possible if region_boxes is a list
        """
        if isinstance(x, ImageEncoderOutput):
            raise ValueError('ImageEncoderOutput is not supported for region encoding')
            encoded_image = x
            
        else:
            encoded_image: ImageEncoderOutput = self.img_encoder(x)
        N = encoded_image.N

        if torch.is_tensor(region_boxes):
            assert region_boxes.ndim == 3
            assert region_boxes.shape[0] == N
            assert region_mask is None or region_mask.shape == region_boxes.shape[:2]
        else:
            assert isinstance(region_boxes, (list, tuple))
            assert len(region_boxes) == N
            assert region_mask is None
            max_M = max([b.shape[0] for b in region_boxes])
            # (N x M_max)
            region_mask = torch.stack([torch.cat([b.new_ones(b.shape[0]), b.new_zeros(max_M - b.shape[0])], dim=0) for b in region_boxes])
            # (N x M_max x 4)
            region_boxes = torch.stack([torch.cat([b, b.new_zeros(max_M - b.shape[0], 4)], dim=0) for b in region_boxes])

        if region_prompt_emb is not None:
            assert region_prompt_emb.shape[:2] == region_boxes.shape[:2]

        box_features = self.detector.encode_regions(encoded_image, region_boxes, region_mask, region_prompt_emb=region_prompt_emb)
    
        if self.post_decoder is not None and use_post_decoder:
            box_features, _ = self.post_decoder(encoded_image, box_features, multiboxes_features=None, query_mask=region_mask)

        return box_features

    def classify_features(self, 
            features: FloatTensor, pos_prompt_emb: FloatTensor, neg_prompt_emb: FloatTensor, 
            normalized: bool = True, temp=1.0, softmax=False, 
            threshold: Optional[float] = 0.5, return_logits=False) -> Tuple[FloatTensor, BoolTensor]:
        return classify_features(features, pos_prompt_emb, neg_prompt_emb,
            normalized=normalized, temp=temp, softmax=softmax, threshold=threshold, return_logits=return_logits)

    def build_evaluator(self, task: EvalConfig, **kwargs) -> Evaluator:
        if task.task is None:
            return TrainEvaluator(task, self, **kwargs)
        elif task.task == 'pathology_detection':
            return PathologyDetectionEvaluator(task, self, **kwargs)
        elif task.task == 'anatomy_explanation':
            return AnatomyExplainerEvaluator(task, self, **kwargs)
        elif task.task == 'box_explanation':
            return BoxExplainerEvaluator(task, self, **kwargs)
        elif task.task == 'sentence_grounding':
            return SentenceGroundingEvaluator(task, self, **kwargs)
        elif task.task == 'report_generation':
            return ReportEvaluator(task, self, **kwargs)
        else:
            raise ValueError(f'Unknown task {task.task}')


@dataclass
class TrainEvalConfig(EvalConfig):
    no_patho_token_supervision: bool = False
    no_anat_token_supervision: bool = False
    no_sent_token_supervision: bool = False

class TrainEvaluator(Evaluator):
    def __init__(self, task: TrainEvalConfig, model: ChEX, **kwargs):
        super().__init__(task, TrainEvalConfig, **kwargs)
        self.model = model
        self.task=task
        self._register_metric(TrainingMetrics(eval_generation=model.has_txt_decoder))
        self.no_patho_token_supervision = self.config.no_patho_token_supervision
        self.no_anat_token_supervision = self.config.no_anat_token_supervision
        self.no_sent_token_supervision = self.config.no_sent_token_supervision

    def eval_step(self, **kwargs) -> 'ImgTextTrainOutput':
        output: ImgTextTrainOutput = self.model(**kwargs, generate=self.model.has_txt_decoder, 
                                                no_patho_token_supervision=self.no_patho_token_supervision,
                                                no_anat_token_supervision=self.no_anat_token_supervision,
                                                no_sent_token_supervision=self.no_sent_token_supervision,
                                                validate=True,
                                                task=self.task)
        self._update_metric(model_output=output)
        return output
 
    
    def plot(self, output: ImgTextTrainOutput, max_samples: int, target_dir: str, plot_local):
        plots = {}
        if output.encoded_sentences is not None:
            plots['grounding'] = plot_grounding(output, max_samples=max_samples),

            if self.model.has_txt_decoder:
                plots['gen_text_sent'] = wandb_plot_text(output.generated_sentences, output.encoded_sentences.sentences, max_samples=max_samples)
            
        return plots
