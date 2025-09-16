
import einops
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics import PearsonCorrCoef
from metrics.detection_metrics import BoxStatistics, batched_box_iou
from metrics.textgen_metrics import SentenceMetrics
from util.data_utils import to_device

from util.train_utils import AvgDictMeter
import ipdb
import logging
log = logging.getLogger(__name__)

class ContrastTrainingMetrics(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.step_metrics_meter = AvgDictMeter()


        self.device = device
        self.to(device)

    def reset(self):
        self.step_metrics_meter.reset()



    @torch.inference_mode()
    def update(self, model_output: "ImgTextTrainOutput"):

        step_metrics = dict( loss=model_output.loss.detach())
        self.step_metrics_meter.add(to_device(step_metrics, 'cpu'))
    
    @torch.inference_mode()
    def compute(self):
        metrics = {
            **{k: v.cpu() for k, v in self.step_metrics_meter.compute().items()},
        }
        return metrics
