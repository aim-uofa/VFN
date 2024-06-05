import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean
from .common import (Linear, VecLinear, GeoVecMLPMultiLayers, SimpleMLPMultiLayers,GaussianLayer,GeoVecMLPMultiLayers_relu,
                    GeoVecMLPMultiLayers_gvp, vec_to_tensor, tensor_to_vec,init_vec_with_true_atom)
from einops import rearrange
from unicore.modules import LayerNorm

"""============================================================================================="""
""" Graph Encoder """
"""============================================================================================="""

class VecNorm(nn.Module):
    def __init__(self, num_vec):
        super(VecNorm, self).__init__()
        self.num_vec = num_vec
        self.lenth_norm = nn.BatchNorm1d(num_vec)

    def forward(self, vec):
        vec = rearrange(vec, 'l (v c) -> l v c', c = 3)
        vec_len = vec.norm(dim=-1)
        vec_dir = vec / (vec_len[...,None]+1e-6)
        vec_len_normed = self.lenth_norm(vec_len)
        vec = vec_dir * vec_len_normed[...,None]
        vec = rearrange(vec, 'l v c -> l (v c)')
        return vec


#################################### node modules ###############################
class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, edge_drop=0.0, output_mlp=True,cfg = None,
                 vec_mlp_n = 2, vec_num_mlp_layer = 2):
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.edge_drop = edge_drop
        self.output_mlp = output_mlp
        self.num_vec = cfg.num_vec
        self.vec_mlp_n = cfg.vec_mlp_n
        self.vec_num_mlp_layer = cfg.vec_num_mlp_layer
        self.vec_learnable = cfg.vec_learnable
        if not self.vec_learnable:
            #print('use detach vector')
            print('use one vector')

        self.W_V = nn.Sequential(Linear(num_in, num_hidden),
                                nn.GELU(),
                                Linear(num_hidden, num_hidden),
                                nn.GELU(),
                                Linear(num_hidden, num_hidden)
        )
        self.vector_V = VecLinear(self.num_vec, self.num_vec,"geometric")
        self.Bias = nn.Sequential(
                                Linear(num_hidden*3, num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_heads*2)
                                )
        self.W_O = Linear(num_hidden, num_hidden, bias=False)
        self.h_e_reducer = Linear(num_hidden*2+4*self.num_vec, num_hidden*2, bias=False)
        self.edge_vector_field = cfg.edge_vector_field 
        self.use_mlp = 'not' 
        self.edge_vector_field_linear = VecLinear(self.num_vec*2, self.num_vec, "geometric")
        self.edge_vector_field_linear.Linear.Linear._zero_init(False)

        self.vector_dist_bn = nn.BatchNorm1d(self.num_vec)
        self.final_vector_output = Linear(self.num_vec*3, self.num_vec*3, bias=True, init='final')
        self.vec_merge_v_src = VecLinear(self.num_vec*2, self.num_vec,"geometric")
        self.vec_merge_v_src.Linear.Linear._zero_init(False)
        self.use_simple_mlp = cfg.use_simple_mlp 
        self.enable_vec_direct = cfg.enable_vec_direct
        self.dist_gbf = cfg.dist_gbf
        self.decompose  = cfg.decompose
        self.use_simple_relu = cfg.use_simple_relu
        if self.vec_mlp_n*self.vec_num_mlp_layer:
            if not self.use_simple_mlp:
                if self.use_simple_relu:
                    self.vec_merge_v_src_mlp = GeoVecMLPMultiLayers_relu(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                    self.h_V_vector_edge_mlp = GeoVecMLPMultiLayers_relu(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                elif cfg.use_simple_gvp:
                    self.vec_merge_v_src_mlp = GeoVecMLPMultiLayers_gvp(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                    self.h_V_vector_edge_mlp = GeoVecMLPMultiLayers_gvp(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                else:
                    self.vec_merge_v_src_mlp = GeoVecMLPMultiLayers(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                    self.h_V_vector_edge_mlp = GeoVecMLPMultiLayers(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
            else:
                self.vec_merge_v_src_mlp = SimpleMLPMultiLayers(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)
                self.h_V_vector_edge_mlp = SimpleMLPMultiLayers(self.num_vec, n = self.vec_mlp_n, num_layers = self.vec_num_mlp_layer)

            self.use_mlp = 'mlp'
        else:
            print('no mlp')


        self.softmax_dropout = 0.2
        if self.dist_gbf:
            self.gbf = GaussianLayer(num_hidden, self.num_vec)
            self.h_e_reducer = Linear(num_hidden*2+4*self.num_vec+3*self.num_vec, num_hidden*2, bias=False)
            if not self.decompose:
                self.h_e_reducer = Linear(num_hidden*2+3*self.num_vec, num_hidden*2, bias=False)

    def forward(self, h_V, h_E, center_id, batch_id, dst_idx=None, frame_dst = None, h_V_vector = None):
        N = h_V.shape[0]
        E = h_E.shape[0]
        n_heads = self.num_heads
        d = int(self.num_hidden / n_heads)
        if not self.vec_learnable:
            # ori_h_V_vector = h_V_vector
            # h_V_vector = h_V_vector.detach()
            h_V_vector = torch.ones_like(h_V_vector)*0.1   
        h_V_vector_init = h_V_vector

        h_V_vector_src = h_V_vector[center_id]
        h_V_vector_dst = h_V_vector[dst_idx]

        h_V_vector_dst_trans = frame_dst[:,None].apply(h_V_vector_dst)

        h_V_vector_edge = torch.cat([h_V_vector_dst_trans, h_V_vector_src],dim=-2)
        h_V_vector_edge = self.edge_vector_field_linear(h_V_vector_edge) + h_V_vector_dst_trans


        h_V_vector_dist = h_V_vector_edge.norm(dim = -1) + 1e-6
        h_V_vector_edge_sin_cos = h_V_vector_edge/h_V_vector_dist[...,None]
        #print(h_V_vector_edge_sin_cos.shape)
        if not self.dist_gbf:
            h_V_vector_dist = self.vector_dist_bn(h_V_vector_dist)
        else:
            h_V_vector_dist = self.gbf(h_V_vector_dist, h_V[center_id], h_V[dst_idx])
        if not self.enable_vec_direct:
            h_V_vector_edge_sin_cos = h_V_vector_edge_sin_cos.zero_()
        if self.decompose:
            #print(h_V_vector_edge_sin_cos.flatten(start_dim=1).shape)
            vector_field_feat_to_s = torch.cat([h_V_vector_edge_sin_cos.flatten(start_dim=1),h_V_vector_dist],dim=1) # LX(96+self.num_vec)
        else:
            # directly flatten
            vector_field_feat_to_s = h_V_vector_edge.flatten(start_dim=1)
        h_E = torch.cat([vector_field_feat_to_s, h_E],dim=-1)
        h_E = self.h_e_reducer(h_E) # 12720,256
        
        dtype = h_V.dtype
        w = self.Bias(torch.cat([h_V[center_id], h_E],dim=-1)).view(E, n_heads*2, 1) 
        w = w.float()
        attend_logits = w/np.sqrt(d) 

        if self.softmax_dropout > 0:
            if self.training:
                # set -inf for dropout,v2 use mask
                mask = torch.rand_like(attend_logits, dtype=dtype) < self.softmax_dropout
                attend_logits.masked_fill_(mask, -float('inf'))

        V = self.W_V(h_E).view(-1, n_heads, d) 
        attend = scatter_softmax(attend_logits, index=center_id, dim=0)
        attend = attend.type(dtype)
        attend, attend_vector = attend.split([4,4],dim=1)
        h_V = scatter_sum(attend*V, center_id, dim=0).view([-1, self.num_hidden])

        if self.output_mlp:
            h_V_update = self.W_O(h_V)
        else:
            h_V_update = h_V

        h_V_vector_edge_v = self.vector_V(h_V_vector_dst_trans)
        h_V_vector_edge_v = rearrange(h_V_vector_edge_v,'l (h v) c -> l h v c', h = n_heads)
        h_V_vector_v = scatter_sum(attend_vector[...,None]*h_V_vector_edge_v, center_id, dim=0).flatten(1,2)
        h_V_vector_v = self.vec_merge_v_src(torch.cat([h_V_vector_v, h_V_vector],dim=-2)) + h_V_vector_v
        if self.use_mlp == 'mlp':
            h_V_vector_v = self.vec_merge_v_src_mlp(h_V_vector_v)
        h_V_vector_v = vec_to_tensor(h_V_vector_v)
        h_V_vector = self.final_vector_output(h_V_vector_v)
        h_V_vector = tensor_to_vec(h_V_vector)

        h_V_vector = h_V_vector + h_V_vector_init
        if not self.vec_learnable:
            #h_V_vector = ori_h_V_vector
            h_V_vector = torch.ones_like(h_V_vector)*0.1
        return h_V_update, h_V_vector, vector_field_feat_to_s


#################################### edge modules ###############################
class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30,cfg = None):
        super(EdgeMLP, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.num_vec = cfg.num_vec
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        # self.norm = LayerNorm(num_hidden)
        self.W11 = Linear(num_hidden + num_in + self.num_vec*4, num_hidden, bias=True)
        if cfg.dist_gbf:
            self.W11 = Linear(num_hidden + num_in + self.num_vec*7, num_hidden, bias=True)
            if not cfg.decompose:
                self.W11 = Linear(num_hidden + num_in + self.num_vec*3, num_hidden, bias=True)
        self.W12 = Linear(num_hidden, num_hidden, bias=True)
        self.W13 = Linear(num_hidden, num_hidden, bias=True)
        self.use_mlp = cfg.use_mlp_in_edge
        self.act = torch.nn.GELU()
        self.disable = cfg.disable_edge_module

    def forward(self, h_V, h_E, edge_idx, batch_id, vector_field_feat_to_s):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx], vector_field_feat_to_s], dim=-1)
        if self.use_mlp:
            h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        else:
            h_message = self.W11(h_EV)
        h_E = self.norm(h_E + self.dropout(h_message))
        if self.disable:
            h_E = h_E*0
        return h_E

#################################### context modules ###############################
class Context(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_context = False, edge_context = False):
        super(Context, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.node_context = node_context
        self.edge_context = edge_context

        self.V_MLP = nn.Sequential(
                                Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_hidden),
                                )
        
        self.V_MLP_g = nn.Sequential(
                                Linear(num_hidden, num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, h_E, edge_idx, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V, h_E


class GeneralGNN(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = 0, edge_context = 0, vector_field_post_norm=False,cfg = None):
        super(GeneralGNN, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        self.node_net = node_net
        self.edge_net = edge_net
        if node_net == 'AttMLP':
            self.attention = NeighborAttention(num_hidden, num_in, num_heads=4,cfg = cfg) 
        if edge_net == 'None':
            pass
        if edge_net == 'EdgeMLP':
            self.edge_update = EdgeMLP(num_hidden, num_in, num_heads=4, dropout = dropout,cfg = cfg)

        
        self.context = Context(num_hidden, num_in, num_heads=4, node_context=node_context, edge_context=edge_context)
        self.vector_field_post_vec_norm  = cfg.vector_field_post_vec_norm
        self.dense = nn.Sequential(
            Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            Linear(num_hidden*4, num_hidden)
        )
        self.W11 = Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = Linear(num_hidden, num_hidden, bias=True)
        self.W13 = Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.use_vec_update_edge = cfg.use_vec_update_edge
        self.cross_update = cfg.cross_update
        if self.cross_update:
            self.cross_update_module = CrossUpdate(num_hidden, cfg.num_vec , drop_out = dropout,cfg = cfg)
    def forward(self, h_V, h_E, edge_idx, batch_id, Q_neighbors, h_V_vector):
        src_idx = edge_idx[0]
        dst_idx = edge_idx[1]

        dh, h_V_vector, vector_field_feat_to_s = self.attention(h_V, torch.cat([h_E, h_V[dst_idx]], dim=-1), src_idx, batch_id, dst_idx, Q_neighbors, h_V_vector)

        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        if self.cross_update:
            scalar_update, vector_update = self.cross_update_module(h_V, h_V_vector)
            h_V = h_V + scalar_update
            h_V_vector = h_V_vector + vector_update

        if self.use_vec_update_edge:
            h_E = self.edge_update( h_V, h_E, edge_idx, batch_id, vector_field_feat_to_s)
        else:
            h_E = self.edge_update( h_V, h_E, edge_idx, batch_id, vector_field_feat_to_s*0)

        h_V, h_E = self.context(h_V, h_E, edge_idx, batch_id)

        return h_V,h_E, h_V_vector

class CrossUpdate(nn.Module):
    def __init__(self,scalar_dim,vec_num ,drop_out,cfg = None):
        super(CrossUpdate, self).__init__()
        self.scalar_dim = scalar_dim
        self.vec_num = vec_num
        self.mode  = 'nomerge'
        if self.mode == 'nomerge':
            self.vector_to_scalar = nn.Sequential(
                Linear(vec_num*3, scalar_dim),
                nn.ReLU(),
                Linear(scalar_dim, scalar_dim,init='final')
            )
            self.scalar_to_vector = nn.Sequential(
                Linear(scalar_dim, vec_num*4),
                nn.ReLU(),
                Linear(vec_num*4, vec_num*3,init='final')
            )
        elif self.mode == 'merge':
            self.vector_to_hidden = nn.Sequential(
                Linear(vec_num*3, scalar_dim),
                nn.ReLU(),
                Linear(scalar_dim, scalar_dim)
            )
            self.scalar_to_hidden = nn.Sequential(
                Linear(scalar_dim, scalar_dim),
                nn.ReLU(),
                Linear(scalar_dim, scalar_dim)
            )
            self.hidden_to_vector = Linear(scalar_dim, vec_num*3,init='final')
            self.hidden_to_scalar = Linear(scalar_dim, scalar_dim,init='final')
        self.dropout = nn.Dropout(drop_out)
    def forward(self, scalar, vector):
        # n x vec_numx 3 -> n x (vec_num*3)
        vector = vector.view(-1, self.vec_num*3)
        if self.mode == 'nomerge':
            
            scalar = self.vector_to_scalar(vector)
            vector = self.scalar_to_vector(scalar)
        elif self.mode == 'merge':
            hidden = self.vector_to_hidden(vector) + self.scalar_to_hidden(scalar)
            scalar = self.hidden_to_scalar(hidden)
            vector = self.hidden_to_vector(hidden)

        scalar = self.dropout(scalar)
        vector = self.dropout(vector)
        vector = vector.view(-1, self.vec_num, 3)
        return scalar, vector

        
class StructureEncoder(nn.Module):
    def __init__(self,  hidden_dim, num_encoder_layers=3, dropout=0, node_net = 'AttMLP', edge_net = 'EdgeMLP', node_context = True, edge_context = False, vector_field_post_norm=False,cfg = None):
        """ Graph labeling network """
        super(StructureEncoder, self).__init__()
        encoder_layers = []
        
        module = GeneralGNN

        for i in range(num_encoder_layers):
            encoder_layers.append(
                module(hidden_dim, hidden_dim*2, dropout=dropout, node_net = node_net, edge_net = edge_net, node_context = node_context, edge_context = edge_context, vector_field_post_norm = vector_field_post_norm,cfg = cfg),
            )
        self.init_vec = cfg.init_vec 
        self.num_vec = cfg.num_vec
        if self.init_vec == 'static':
            self.init_vec_ = nn.Parameter(torch.randn(1, cfg.num_vec*3))
        elif self.init_vec == 'fromscalar':
            self.init_vec_ = Linear(hidden_dim, cfg.num_vec*3)
        self.multi_layer_output = cfg.per_layer_supervision
        
        self.encoder_layers = nn.Sequential(*encoder_layers)

    def forward(self, h_V, h_P, P_idx, batch_id, Q_neighbors, h_V_vector,V_xyz=None):
        if self.init_vec == 'static':
            h_V_vector = self.init_vec_.repeat(h_V.shape[0],1)
            h_V_vector = h_V_vector.view(-1, self.num_vec, 3)
        elif self.init_vec == 'fromscalar':
            h_V_vector = self.init_vec_(h_V)
            h_V_vector = h_V_vector.view(-1, self.num_vec, 3)
        elif self.init_vec == 'linear':
            pass
        elif self.init_vec == 'true_atom':
            h_V_vector = init_vec_with_true_atom(h_V_vector, V_xyz)
        else:
            raise NotImplementedError
        if self.multi_layer_output:
            h_V_vector_list = []
        for layer in self.encoder_layers:
            h_V, h_P, h_V_vector = layer(h_V, h_P, P_idx, batch_id, Q_neighbors, h_V_vector)
            if self.multi_layer_output:
                h_V_vector_list.append(h_V_vector)
        if self.multi_layer_output: 
            return h_V, h_P, h_V_vector_list
        return h_V, h_P, h_V_vector


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=20):
        super().__init__()
        self.readout = Linear(hidden_dim, vocab, init='final')
    
    def forward(self, h_V, batch_id=None):
        logits = self.readout(h_V)
        return logits


if __name__ == '__main__':
    pass