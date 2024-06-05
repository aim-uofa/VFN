import torch
import torch.nn as nn
from typing import Optional, Tuple

from unicore.utils import one_hot

from .common import Linear, residual, Transition
from .common import SimpleModuleList
from .frame import Frame
from einops import rearrange
import copy

from ..utils.layer_norm import LayerNorm


class InputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        d_pair: int,
        d_msa: int,
        relpos_k: int,
        use_chain_relative: bool = False,
        max_relative_chain: Optional[int] = None,
        **kwargs,
    ):
        super(InputEmbedder, self).__init__()

        self.tf_dim = tf_dim
        self.msa_dim = msa_dim

        self.d_pair = d_pair
        self.d_msa = d_msa

        self.linear_tf_z_i = Linear(tf_dim, d_pair)
        self.linear_tf_z_j = Linear(tf_dim, d_pair)
        self.linear_tf_m = Linear(tf_dim, d_msa)
        self.linear_msa_m = Linear(msa_dim, d_msa)

        # RPE stuff
        self.relpos_k = relpos_k
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if not self.use_chain_relative:
            self.num_bins = 2 * self.relpos_k + 1
        else:
            self.num_bins = 2 * self.relpos_k + 2
            self.num_bins += 1  # entity id
            self.num_bins += 2 * max_relative_chain + 2

        self.linear_relpos = Linear(self.num_bins, d_pair)

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ):

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
        num_sym: Optional[torch.Tensor] = None,
    ):

        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(
                one_hot(rp, num_classes=self.num_bins, dtype=dtype)
            )
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(
                rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype
            )
            return self.linear_relpos(
                torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1)
            )

    def forward(
        self,
        tf: torch.Tensor,
        msa: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, d_pair]
        if self.tf_dim == 21:
            # multimer use 21 target dim
            tf = tf[..., 1:]
        # convert type if necessary
        tf = tf.type(self.linear_tf_z_i.weight.dtype)
        msa = msa.type(self.linear_tf_z_i.weight.dtype)
        n_clust = msa.shape[-3]

        msa_emb = self.linear_msa_m(msa)
        # target_feat (aatype) into msa representation
        tf_m = (
            self.linear_tf_m(tf)
            .unsqueeze(-3)
            .expand(((-1,) * len(tf.shape[:-2]) + (n_clust, -1, -1)))  # expand -3 dim
        )
        msa_emb += tf_m

        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        return msa_emb, pair_emb

class InverseFoldingInputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        d_pair: int,
        d_msa: int,
        relpos_k: int,
        use_chain_relative: bool = False,
        max_relative_chain: Optional[int] = None,
        **kwargs,
    ):
        super(InverseFoldingInputEmbedder, self).__init__()

        tf_dim = 1
        self.tf_dim = tf_dim

        self.d_pair = d_pair
        self.d_msa = d_msa

        self.linear_tf_z_i = Linear(tf_dim, d_pair)
        self.linear_tf_z_j = Linear(tf_dim, d_pair)
        self.linear_tf_m = Linear(tf_dim, d_msa)


        # RPE stuff
        self.relpos_k = relpos_k
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if not self.use_chain_relative:
            self.num_bins = 2 * self.relpos_k + 1
        else:
            self.num_bins = 2 * self.relpos_k + 2
            self.num_bins += 1  # entity id
            self.num_bins += 2 * max_relative_chain + 2

        self.linear_relpos = Linear(self.num_bins, d_pair)

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ):

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
        num_sym: Optional[torch.Tensor] = None,
    ):

        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(
                one_hot(rp, num_classes=self.num_bins, dtype=dtype)
            )
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(
                rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype
            )
            return self.linear_relpos(
                torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1)
            )

    def forward(
        self,
        tf: torch.Tensor,
        residue_index: torch.Tensor,
        *arg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [*, N_res, d_pair]
        tf = tf.new_ones(tf[...,:1].shape)
        tf = tf.type(self.linear_tf_z_i.weight.dtype)

        s = self.linear_tf_m(tf)

        tf_emb_i = self.linear_tf_z_i(tf)
        tf_emb_j = self.linear_tf_z_j(tf)
        pair_emb = tf_emb_i[..., None, :] + tf_emb_j[..., None, :, :]

        relpos_emb = self.relpos_emb(residue_index)
        pair_emb +=relpos_emb

        return s, pair_emb
    
class InverseFoldingPairInputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        d_pair: int,
        d_msa: int,
        relpos_k: int,
        use_chain_relative: bool = False,
        max_relative_chain: Optional[int] = None,
        **kwargs,
    ):
        super(InverseFoldingPairInputEmbedder, self).__init__()

        tf_dim = 1
        self.tf_dim = tf_dim

        self.d_pair = d_pair
        self.d_msa = d_msa

        # self.linear_tf_z_i = Linear(tf_dim, d_pair)
        # self.linear_tf_z_j = Linear(tf_dim, d_pair)
        self.linear_tf_m = Linear(tf_dim, d_msa)
        self.linear_pair = Linear(9, d_pair, init="relu")
        self.linear_pair_ffn = Transition(d_pair,4)


        # RPE stuff
        self.relpos_k = relpos_k
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if not self.use_chain_relative:
            self.num_bins = 2 * self.relpos_k + 1
        else:
            self.num_bins = 2 * self.relpos_k + 2
            self.num_bins += 1  # entity id
            self.num_bins += 2 * max_relative_chain + 2

        self.linear_relpos = Linear(self.num_bins, d_pair)

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ):

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
        num_sym: Optional[torch.Tensor] = None,
    ):

        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(
                one_hot(rp, num_classes=self.num_bins, dtype=dtype)
            )
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(
                rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype
            )
            return self.linear_relpos(
                torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1)
            )
        
    def get_pair_info( # TODO weian: try x**2 or x**-2 or x or log(x) and so on
        self,
        gt_aff,
        residue_index,
    ):
        loc = gt_aff._t
        xyz_loc = loc[:,:,None] - loc[:,None]
        squ_dis = (xyz_loc**2).sum(-1,keepdim=True) # x**2 + y**2 + z**2

        pair_distance = torch.sqrt_(squ_dis)# dis = sqrt (x**2 + y**2 + z**2)
        eye_mask = torch.eye(pair_distance.shape[1],dtype=bool,device=pair_distance.device).repeat(pair_distance.shape[0],1,1)  # eye is zero #TODO weian big issue

        direct_angle = xyz_loc / (pair_distance + 1e-10)
        direct_angle_inverse = rearrange(direct_angle,'b q k c -> b k q c')

        rel_res_idx = residue_index[:,:,None] - residue_index[:,None]
        connect_mask = rel_res_idx.abs() == 1

        pair_info = {
            'pair_distance':pair_distance,
            'eye_mask':eye_mask,
            'direct_angle':direct_angle,
            'direct_angle_inverse':direct_angle_inverse,
            'connect_mask':connect_mask,
        }
        return pair_info
        # print('a')# [b,q,k,c]

    def get_pair_rep(
        self,
        pair_info,
    ):
        eye_mask = pair_info['eye_mask']
        pair_distance = pair_info['pair_distance'].clamp(min=1e-10)
        pair_dist_rep = 1/(pair_distance**2) #1/x**2
        pair_dist_rep[eye_mask] = 0
        pair_dist_rep /= 0.08 #TODO weian: chack the value
        # TODO weian: gnorm warning

        direct_angle = pair_info['direct_angle']
        direct_angle_inverse = pair_info['direct_angle_inverse']
        direct_angle[eye_mask] = 0
        direct_angle_inverse[eye_mask] = 0

        connect_mask = pair_info['connect_mask'].to(pair_dist_rep.dtype)[...,None]
        eye_mask = eye_mask.to(pair_dist_rep.dtype)[...,None]

        pair_rep = torch.cat([eye_mask,connect_mask,direct_angle_inverse,direct_angle,pair_dist_rep],dim=-1)
        pair_rep = self.linear_pair(pair_rep)

        pair_rep = residual(pair_rep,self.linear_pair_ffn(pair_rep),self.training)


        return pair_rep

    def forward(
        self,
        tf: torch.Tensor,
        residue_index: torch.Tensor,
        true_frame_tensor: torch.Tensor,
        *arg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            gt_aff = Frame.from_tensor_4x4(true_frame_tensor)
            pair_info = self.get_pair_info(gt_aff,residue_index) # [b,q,k,c]

        # [*, N_res, d_pair]
        tf = tf.new_ones(tf[...,:1].shape)
        tf = tf.type(self.linear_tf_m.weight.dtype)

        s = self.linear_tf_m(tf)

        pair_emb = self.get_pair_rep(pair_info)

        relpos_emb = self.relpos_emb(residue_index)
        pair_emb +=relpos_emb

        return s, pair_emb
    
class InverseFoldingGraghInputEmbedder(nn.Module):
    def __init__(
        self,
        tf_dim: int,
        msa_dim: int,
        d_pair: int,
        d_msa: int,
        relpos_k: int,
        use_chain_relative: bool = False,
        max_relative_chain: Optional[int] = None,
        topk = 30,
        **kwargs,
    ):
        super(InverseFoldingGraghInputEmbedder, self).__init__()

        self.topk = topk
        tf_dim = 1
        self.tf_dim = tf_dim

        self.d_pair = d_pair
        self.d_msa = d_msa

        # self.linear_tf_z_i = Linear(tf_dim, d_pair)
        # self.linear_tf_z_j = Linear(tf_dim, d_pair)
        self.linear_tf_m = Linear(tf_dim, d_msa)
        # self.linear_pair = Linear(9, d_pair, init="relu")
        # self.linear_pair_ffn = Transition(d_pair,4)
        self.linear_pair_angle = TemplateAngleEmbedder(5,d_pair)
        self.linear_pair_dist = TemplateAngleEmbedder(3,d_pair)

        # RPE stuff
        self.relpos_k = relpos_k
        self.use_chain_relative = use_chain_relative
        self.max_relative_chain = max_relative_chain
        if not self.use_chain_relative:
            self.num_bins = 2 * self.relpos_k + 1
        else:
            self.num_bins = 2 * self.relpos_k + 2
            self.num_bins += 1  # entity id
            self.num_bins += 2 * max_relative_chain + 2

        self.linear_relpos = Linear(self.num_bins, d_pair)

        self.inf = 1e10

    def _relpos_indices(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
    ):

        max_rel_res = self.relpos_k
        rp = res_id[..., None] - res_id[..., None, :]
        rp = rp.clip(-max_rel_res, max_rel_res) + max_rel_res
        if not self.use_chain_relative:
            return rp
        else:
            asym_id_same = asym_id[..., :, None] == asym_id[..., None, :]
            rp[~asym_id_same] = 2 * max_rel_res + 1
            entity_id_same = entity_id[..., :, None] == entity_id[..., None, :]
            rp_entity_id = entity_id_same.type(rp.dtype)[..., None]

            rel_sym_id = sym_id[..., :, None] - sym_id[..., None, :]

            max_rel_chain = self.max_relative_chain

            clipped_rel_chain = torch.clamp(
                rel_sym_id + max_rel_chain, min=0, max=2 * max_rel_chain
            )

            clipped_rel_chain[~entity_id_same] = 2 * max_rel_chain + 1
            return rp, rp_entity_id, clipped_rel_chain

    def relpos_emb(
        self,
        res_id: torch.Tensor,
        sym_id: Optional[torch.Tensor] = None,
        asym_id: Optional[torch.Tensor] = None,
        entity_id: Optional[torch.Tensor] = None,
        num_sym: Optional[torch.Tensor] = None,
    ):

        dtype = self.linear_relpos.weight.dtype
        if not self.use_chain_relative:
            rp = self._relpos_indices(res_id=res_id)
            return self.linear_relpos(
                one_hot(rp, num_classes=self.num_bins, dtype=dtype)
            )
        else:
            rp, rp_entity_id, rp_rel_chain = self._relpos_indices(
                res_id=res_id, sym_id=sym_id, asym_id=asym_id, entity_id=entity_id
            )
            rp = one_hot(rp, num_classes=(2 * self.relpos_k + 2), dtype=dtype)
            rp_entity_id = rp_entity_id.type(dtype)
            rp_rel_chain = one_hot(
                rp_rel_chain, num_classes=(2 * self.max_relative_chain + 2), dtype=dtype
            )
            return self.linear_relpos(
                torch.cat([rp, rp_entity_id, rp_rel_chain], dim=-1)
            )
        
    def get_pair_info( # TODO weian: try x**2 or x**-2 or x or log(x) and so on
        self,
        gt_aff,
        residue_index,
        seq_mask,
    ):
        seq_mask=seq_mask.bool()
        # seq_mask_pair = seq_mask[:,:,None] * seq_mask[:,None]
        seq_mask_pair = seq_mask[:,None] * seq_mask[:,:,None]

        loc = gt_aff._t
        # xyz_loc = loc[:,:,None] - loc[:,None]
        xyz_loc = loc[:,None] - loc[:,:,None]
        pair_distance = xyz_loc.norm(dim=-1)

        eye_mask = torch.eye(pair_distance.shape[1],dtype=bool,device=pair_distance.device).repeat(pair_distance.shape[0],1,1)
        # rel_res_idx = residue_index[:,:,None] - residue_index[:,None]
        rel_res_idx = residue_index[:,None] - residue_index[:,:,None]
        connect_mask = rel_res_idx.abs() == 1

        pair_distance_for_topk = copy.deepcopy(pair_distance)
        pair_distance_for_topk[eye_mask] = self.inf
        pair_distance_for_topk[~seq_mask_pair] = self.inf
        pair_distance_for_topk[connect_mask] = -self.inf
        pair_topk_inx = torch.topk(pair_distance_for_topk,self.topk,largest=False,sorted=False)[1]

        topk_pair_distance = torch.gather(pair_distance, dim=-1, index=pair_topk_inx)
        knn_mask = seq_mask_pair & ~eye_mask #remove unknow res
        knn_mask = torch.gather(knn_mask, dim=-1, index=pair_topk_inx)
        assert not (topk_pair_distance[knn_mask]==0).any()


        import time
        torch.cuda.synchronize()
        start_time = time.time()
        # gt_aff[:,:,None].compose(gt_aff[:,None])

        gt_aff_repeated = gt_aff[:,None].repeat([1,gt_aff.shape[1],1])
        gt_aff_topk = gt_aff_repeated.gather(dim=-1, index=pair_topk_inx)
        gt_aff_topk = gt_aff[...,None].decompose(gt_aff_topk)
        torch.cuda.synchronize()
        print(time.time()-start_time)

        # check
        if True:
            q_num, v_num = pair_topk_inx.shape[1], pair_topk_inx.shape[2]

            ori_type = gt_aff.dtype
            gt_aff=gt_aff.type(torch.double)
            gt_aff_topk=gt_aff_topk.type(torch.double)
            topk_pair_distance=topk_pair_distance.type(torch.double)
            pair_distance=pair_distance.type(torch.double)
            xyz_loc=xyz_loc.type(torch.double)
            for i in range(q_num):
                for j in range(v_num):
                    cur_ind = pair_topk_inx[0,i,j]
                    gt_frame = gt_aff[0,i].decompose(gt_aff[0,cur_ind])
                    is_dist_equal = torch.isclose(topk_pair_distance[0,i,j], gt_frame._t.norm(dim=-1), rtol=1e-02, atol=1e-01,).all()
                    world_t = gt_aff[0,i]._r.apply(gt_frame._t)
                    is_xyz_equal = torch.isclose(xyz_loc[0,i,cur_ind], world_t, rtol=1e-02, atol=1e-01,).all()
                    # is_dist_equal = torch.isclose(pair_distance[0,i,cur_ind], gt_frame._t.norm(-1), rtol=1e-02, atol=1e-01,).all()
                    opt_frame = gt_aff_topk[0,i,j]
                    is_equal = torch.isclose(gt_frame._t,opt_frame._t, rtol=1e-02, atol=1e-01,).all() & torch.isclose(gt_frame._r._mat,opt_frame._r._mat, rtol=1e-05, atol=1e-03,).all()
                    if knn_mask[0,i,j]:
                        assert is_equal
                        assert is_dist_equal
                        assert is_xyz_equal
            # print('pass opt check')
            gt_aff=gt_aff.type(ori_type)
            gt_aff_topk=gt_aff_topk.type(ori_type)
            topk_pair_distance=topk_pair_distance.type(ori_type)
            xyz_loc=xyz_loc.type(ori_type)
            print('pass opt check')



        topk_pair_distance.repeat()
        frame_tensor = gt_aff.to_tensor_4x4().repeat()
        frame_tensor

        print('a')
        pair_info = {
            'topk_pair_distance':topk_pair_distance,
            'eye_mask':eye_mask,
            'connect_mask':connect_mask,
            'knn_mask':knn_mask,
            'pair_topk_inx':pair_topk_inx,
        }
        return pair_info
        # print('a')# [b,q,k,c]

    def get_pair_rep(
        self,
        pair_info,
        residue_index,
        seq_mask,
    ):
        inf = 1e10
        # seq_mask_pair = (seq_mask[:,:,None] * seq_mask[:,None]).bool()
        eye_mask = pair_info['eye_mask']
        pair_distance = copy.deepcopy(pair_info['pair_distance'])
        connect_mask = pair_info['connect_mask']


        # rearrange(pair_distance[~eye_mask],'(b q k) c -> b q k c',b = seq_mask.shape[0],q = seq_mask.shape[1], 
        pair_distance = pair_distance.squeeze()
        pair_distance_for_topk = copy.deepcopy(pair_distance)
        pair_distance_for_topk[eye_mask] = inf
        pair_distance_for_topk[~seq_mask_pair] = inf
        pair_distance_for_topk[connect_mask] = -inf
        pair_topk_inx = torch.topk(pair_distance_for_topk,self.topk,largest=False,sorted=False)[1]

        knn_mask = seq_mask_pair & ~eye_mask #remove unknow res

        topk_pair_distance = torch.gather(pair_distance,dim=-1,index=pair_topk_inx)
        knn_mask = torch.gather(knn_mask,dim=-1,index=pair_topk_inx)
        assert not (topk_pair_distance[knn_mask]==0).any()


        direct_angle = pair_info['direct_angle']
        direct_angle[eye_mask] = 0
        direct_angle = torch.gather(direct_angle,dim=-2,index=pair_topk_inx[...,None].repeat(1,1,1,3))




        connect_mask = connect_mask[...,None].to(direct_angle.dtype)
        eye_mask = eye_mask.to(direct_angle.dtype)[...,None]

        pair_rep_angle = torch.cat([eye_mask,connect_mask,direct_angle],dim=-1)
        pair_rep_dist = torch.cat([eye_mask,connect_mask,pair_dist_rep],dim=-1)
        pair_rep_angle = self.linear_pair_angle(pair_rep_angle)
        pair_rep_dist = self.linear_pair_dist(pair_rep_dist)

        relpos_emb = self.relpos_emb(residue_index)

        # pair_rep = residual(pair_rep,self.linear_pair_ffn(pair_rep),self.training)

        pair_rep = {
            "pair_rep_angle": pair_rep_angle,
            "pair_rep_dist": pair_rep_dist,
            "relpos_emb": relpos_emb,
        }
        return pair_rep

    def forward( # TODO weian: add self.gbf of unimol into gragh model
        self,
        tf: torch.Tensor,
        residue_index: torch.Tensor,
        true_frame_tensor: torch.Tensor,
        *arg,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_mask = arg[0]
        with torch.no_grad():
            gt_aff = Frame.from_tensor_4x4(true_frame_tensor)
            pair_info = self.get_pair_info(gt_aff,residue_index,seq_mask) # [b,q,k,c]

        # [*, N_res, d_pair]
        tf = tf.new_ones(tf[...,:1].shape)
        tf = tf.type(self.linear_tf_m.weight.dtype)

        s = self.linear_tf_m(tf)

        pair_emb = self.get_pair_rep(pair_info, residue_index, seq_mask)

        return s, pair_emb

class RecyclingEmbedder(nn.Module):
    def __init__(
        self,
        d_msa: int,
        d_pair: int,
        min_bin: float,
        max_bin: float,
        num_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        super(RecyclingEmbedder, self).__init__()

        self.d_msa = d_msa
        self.d_pair = d_pair
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.num_bins = num_bins
        self.inf = inf

        self.squared_bins = None

        self.linear = Linear(self.num_bins, self.d_pair)
        self.layer_norm_m = LayerNorm(self.d_msa)
        self.layer_norm_z = LayerNorm(self.d_pair)

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        m_update = self.layer_norm_m(m)
        z_update = self.layer_norm_z(z)

        return m_update, z_update

    def recyle_pos(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.squared_bins is None:
            bins = torch.linspace(
                self.min_bin,
                self.max_bin,
                self.num_bins,
                dtype=torch.float if self.training else x.dtype,
                device=x.device,
                requires_grad=False,
            )
            self.squared_bins = bins**2
        upper = torch.cat(
            [self.squared_bins[1:], self.squared_bins.new_tensor([self.inf])], dim=-1
        )
        if self.training:
            x = x.float()
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )
        d = ((d > self.squared_bins) * (d < upper)).type(self.linear.weight.dtype)
        d = self.linear(d)
        return d


class TemplateAngleEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        **kwargs,
    ):
        super(TemplateAngleEmbedder, self).__init__()

        self.d_out = d_out
        self.d_in = d_in

        self.linear_1 = Linear(self.d_in, self.d_out, init="relu")
        self.act = nn.GELU()
        self.linear_2 = Linear(self.d_out, self.d_out, init="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x.type(self.linear_1.weight.dtype))
        x = self.act(x)
        x = self.linear_2(x)
        return x


class TemplatePairEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        v2_d_in: list,
        d_out: int,
        d_pair: int,
        v2_feature: bool = False,
        **kwargs,
    ):
        super(TemplatePairEmbedder, self).__init__()

        self.d_out = d_out
        self.v2_feature = v2_feature
        if self.v2_feature:
            self.d_in = v2_d_in
            self.linear = SimpleModuleList()
            for d_in in self.d_in:
                self.linear.append(Linear(d_in, self.d_out, init="relu"))
            self.z_layer_norm = LayerNorm(d_pair)
            self.z_linear = Linear(d_pair, self.d_out, init="relu")
        else:
            self.d_in = d_in
            self.linear = Linear(self.d_in, self.d_out, init="relu")

    def forward(
        self,
        x,
        z,
    ) -> torch.Tensor:
        if not self.v2_feature:
            x = self.linear(x.type(self.linear.weight.dtype))
            return x
        else:
            dtype = self.z_linear.weight.dtype
            t = self.linear[0](x[0].type(dtype))
            for i, s in enumerate(x[1:]):
                t = residual(t, self.linear[i + 1](s.type(dtype)), self.training)
            t = residual(t, self.z_linear(self.z_layer_norm(z)), self.training)
            return t


class ExtraMSAEmbedder(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        **kwargs,
    ):
        super(ExtraMSAEmbedder, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.linear = Linear(self.d_in, self.d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.type(self.linear.weight.dtype))
