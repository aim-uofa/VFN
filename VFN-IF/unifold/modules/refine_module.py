import math
import torch
import torch.nn as nn
from typing import Tuple

from .refine_module_utils import AFRefineStructureModule

from unifold.modules.featurization import atom14_to_atom37

class RefineModule(nn.Module):
    def __init__(self, config):
        super(RefineModule, self).__init__()
        self.refine_module_type = config['type']
        if self.refine_module_type== 'baseline':
            self.refine_module = AFRefineStructureModule(**config["baseline_module"])
        else:
            raise
        print('a')

    def forward(self, outputs,feats):
        if self.refine_module_type== 'baseline':
            sm_output = outputs['sm']

            s = outputs['single']
            z = outputs["pair"]
            aatype = sm_output["aatype"]
            mask = sm_output["mask"]
            quat_encoder = sm_output["quat_encoder"]

            outputs['rm'] = self.refine_module(s,z,aatype,quat_encoder,mask)
        else:
            raise

        outputs["final_atom_positions_rm"] = atom14_to_atom37(
            outputs["rm"]["positions"], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["pred_frame_tensor"] = outputs["rm"]["frames"][-1]

        return outputs