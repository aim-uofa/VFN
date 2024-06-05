import math
import torch
import torch.nn as nn
from typing import Tuple

from .inverse_module_utils import AFInverseStructureModule, AFInverseGraghInitModule

from unifold.modules.featurization import atom14_to_atom37

class InverseModule(nn.Module):
    def __init__(self, config, type):
        super(InverseModule, self).__init__()
        self.refine_module_type = type
        if self.refine_module_type== 'default':
            self.refine_module = AFInverseStructureModule(**config["baseline_module"])
        elif self.refine_module_type== 'gragh_init':
            self.refine_module = AFInverseGraghInitModule(**config["baseline_module"])
        else:
            raise

    def forward(self, s,z,gt_frame,mask):
        outputs = self.refine_module(s,z,gt_frame,mask)

        return outputs