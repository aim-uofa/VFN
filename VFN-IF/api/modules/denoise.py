import torch
import torch.nn as nn

from unicore.utils import tensor_tree_map

from .frame import Frame
from .inverse_module_utils import Pifold_featurizer
from ..data import residue_constants as rc
from .VFN import VFN

class Denosie(nn.Module):
    def __init__(self, config):
        super(Denosie, self).__init__()

        self.globals = config.globals
        config = config.model

        self.featurizer = Pifold_featurizer(config["featurizer"])
        self.VFN = VFN(config["VFN"],config["featurizer"])
 
        self.config = config
        self.dtype = torch.float
        self.inf = self.globals.inf
        if self.globals.alphafold_original_mode:
            self.alphafold_original_mode()

    def __make_input_float__(self):
        self.featurizer = self.featurizer.float()

    def half(self):
        super().half()
        if (not getattr(self, "inference", False)):
            self.__make_input_float__()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        if (not getattr(self, "inference", False)):
            self.__make_input_float__()
        self.dtype = torch.bfloat16
        return self

    def alphafold_original_mode(self):
        def set_alphafold_original_mode(module):
            if hasattr(module, "apply_alphafold_original_mode"):
                module.apply_alphafold_original_mode()
            if hasattr(module, "act"):
                module.act = nn.ReLU()

        self.apply(set_alphafold_original_mode)

    def inference_mode(self):
        def set_inference_mode(module):
            setattr(module, "inference", True)
        self.apply(set_inference_mode)

    def __convert_input_dtype__(self, batch):
        for key in batch:
            # only convert features with mask
            if batch[key].dtype != self.dtype and "mask" in key:
                batch[key] = batch[key].type(self.dtype)
        return batch


    def forward(self, batch):
        with torch.no_grad():
            batch = self.__convert_input_dtype__(batch)
            fetch_cur_batch = lambda t: t[0, ...]
            feats = tensor_tree_map(fetch_cur_batch, batch)

        true_frame_tensor = feats["true_frame_tensor"]
        gt_frame = Frame.from_tensor_4x4(true_frame_tensor)
        gt_frame = gt_frame.type(self.dtype)
        atoms37  = feats['all_atom_positions'] # [B, N, 37, 3]
        id_map = rc.restype_atom37_to_atom14 # 21x37 , can map 37 atom to 14 atom according to residue type
        atom_mask = feats['all_atom_mask'] # [B, N, 37]
        atoms14 = feats['atom14_gt_positions'] # [B, N, 14, 3]
        atoms14_mask = feats['atom14_gt_exists'] # [B, N, 14]
        
        atoms14_alt = feats['atom14_alt_gt_positions'] # [B, N, 14, 3]
        atoms14_alt_mask = feats['atom14_alt_gt_exists'] # [B, N, 14]
        atoms14_mask_diff = atoms14_mask-atoms14_alt_mask
        atoms14_diff = atoms14-atoms14_alt
        atoms14_mask = atoms14_mask*atoms14_alt_mask
        atoms14_mask_ori = atoms14_mask.clone()
        atoms14_ambiguous_mask = feats['atom14_atom_is_ambiguous'] # [B, N, 14]
        h_V, h_P, P_idx, batch_id, mask_edge, E_idx_w_gap, Q_neighbors,V_xyz,atoms14,\
            atoms14_mask ,atoms14_alt, atoms14_alt_mask,atoms14_ambiguous_mask = self.featurizer(
            S = feats['aatype'],
            X = feats['all_atom_positions'][:,:,rc.pifold_atom37_index].float(),
            mask = feats['seq_mask'].float(),
            gt_frame = gt_frame,
            atoms14 = atoms14,
            atoms14_mask = atoms14_mask,
            atoms14_alt = atoms14_alt,
            atoms14_alt_mask = atoms14_alt_mask,
            atoms14_ambiguous_mask = atoms14_ambiguous_mask,
        )

        h_V = h_V.type(self.dtype)
        h_P = h_P.type(self.dtype)
        logits = self.VFN(h_V, h_P, P_idx, batch_id, mask_edge, Q_neighbors,V_xyz)
        if isinstance(logits, dict):
            sidechain_location = logits['sidechain_location']
            logits = logits['logits']
        else:
            sidechain_location = None
        return_dict = {'res_score':logits, 'side_chain_pred':sidechain_location, 'atoms14':atoms14, 'atoms14_mask':atoms14_mask }
        if self.config.VFN.consider_alt:
            return_dict['atoms14_alt'] = atoms14_alt
            return_dict['atoms14_alt_mask'] = atoms14_alt_mask
        return  return_dict
