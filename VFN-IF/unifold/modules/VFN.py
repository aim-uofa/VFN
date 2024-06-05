import torch
import torch.nn as nn
from .prodesign_module import StructureEncoder, MLPDecoder
from unifold.modules.common import Linear
from .common import (Linear, GaussianNondeLayer, FakeLinear, tensor_to_vec)

class VFN(nn.Module):
    def __init__(self, model_config, featurizer_config):
        """
        Initializes the VFN model.

        Args:
            model_config (ModelConfig): Configuration object for the model.
            featurizer_config (FeaturizerConfig): Configuration object for the featurizer.

        Attributes:
            transformation_scale_factor (float): Scale factor for the transformation.
            rm_edge_input (bool): Flag indicating whether to remove edge input.
            rm_node_input (bool): Flag indicating whether to remove node input.
            vector_field_post_norm (bool): Flag indicating whether to perform post-normalization on the vector field.
            decode_with_vector (bool): Flag indicating whether to decode with vector.
            sidechain_reconstruction (bool): Flag indicating whether to perform sidechain reconstruction.
            num_vec (int): Number of vectors.

            node_embedding (Linear): Linear layer for node embedding.
            edge_embedding (Linear): Linear layer for edge embedding.
            norm_nodes (BatchNorm1d): Batch normalization layer for nodes.
            norm_edges (BatchNorm1d): Batch normalization layer for edges.
            W_v (Sequential): Sequential module for node transformation.
            W_e (Linear): Linear layer for edge transformation.
            encoder (StructureEncoder): Structure encoder module.
            decoder (MLPDecoder): MLP decoder module.
            per_layer_supervision (bool): Flag indicating whether to use per-layer supervision.
            init_vector (Linear): Linear layer for initializing vectors.
            vector_dist_bn (BatchNorm1d): Batch normalization layer for vector distribution.
            decode_with_vector_transform (Sequential): Sequential module for decoding with vector transformation.
            vector_field_linear (FakeLinear): Fake linear layer for vector field.
            gbf (GaussianNondeLayer): Gaussian nonde layer.
        """

        super(VFN, self).__init__()

        # self initialization
        node_features = model_config.node_features
        edge_features = model_config.edge_features
        hidden_dim = model_config.hidden_dim
        dropout = model_config.dropout
        num_encoder_layers = model_config.num_encoder_layers

        # transformation scale factor
        self.transformation_scale_factor = model_config.transformation_scale_factor
        self.rm_edge_input = model_config.rm_edge_input
        self.rm_node_input = model_config.rm_node_input
        self.vector_field_post_norm = model_config.vector_field_post_norm
        self.decode_with_vector = model_config.decode_with_vector
        self.sidechain_reconstruction = model_config.sidechain_reconstruction
        self.num_vec = model_config.num_vec

        # # Calculate node input
        # node_in = 0
        # if featurizer_config.node_dist:
        #     pair_num = 6
        #     if featurizer_config.virtual_num>0:
        #         pair_num += featurizer_config.virtual_num*(featurizer_config.virtual_num-1)
        #     node_in += pair_num*featurizer_config.num_rbf
        # if featurizer_config.node_angle:
        #     node_in += 12
        # if featurizer_config.node_direct:
        #     node_in += 9
        
        # # Calculate edge input
        # edge_in = 0
        # if featurizer_config.edge_dist:
        #     pair_num = 16
        #     if featurizer_config.virtual_num>0:
        #         pair_num += featurizer_config.virtual_num**2
        #     edge_in += pair_num*featurizer_config.num_rbf
        # if featurizer_config.edge_angle:
        #     edge_in += 4
        # if featurizer_config.edge_direct:
        #     edge_in += 12


        # Initialize the node and edge embedding
        # Depending on the self.rm_node_input
        self.node_embedding = Linear(224, node_features, bias=True)
        self.edge_embedding = Linear(416, edge_features, bias=True)
        self.norm_nodes = nn.BatchNorm1d(node_features)
        self.norm_edges = nn.BatchNorm1d(edge_features)

        self.W_v = nn.Sequential(
            Linear(node_features, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            Linear(hidden_dim, hidden_dim, bias=True),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hidden_dim),
            Linear(hidden_dim, hidden_dim, bias=True)
        )

        self.W_e = Linear(edge_features, hidden_dim, bias=True)
        self.init_vec_with_true_atom = model_config.init_vec_with_true_atom
        self.encoder = StructureEncoder(hidden_dim, num_encoder_layers, dropout, vector_field_post_norm = self.vector_field_post_norm ,cfg = model_config)
        self.decoder = MLPDecoder(hidden_dim)
        self.per_layer_supervision = model_config.per_layer_supervision
        self.init_vector = Linear(hidden_dim, self.num_vec*3, bias=True)
        self.vector_field_linear = FakeLinear(12, 32, "geometric")
        self.gbf = GaussianNondeLayer(32)

    def forward(self, h_V, h_P, P_idx, batch_id, mask_edge = None, Q_neighbors = None,V_xyz = None):
        V_xyz , ori_V_xyz = V_xyz

        V_xyz = self.vector_field_linear(V_xyz)
        V_xyz_dist = V_xyz.norm(dim = -1) + 1e-6
        V_xyz_sin_cos = V_xyz/V_xyz_dist[...,None]
        V_xyz_dist = self.gbf(V_xyz_dist)
        h_V = torch.cat([V_xyz_sin_cos.flatten(start_dim=1),V_xyz_dist],dim=1)
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_P = self.edge_embedding(h_P.zero_()+1)
        

        h_V_vector = self.init_vector(h_V) #TODO weian: need to wirte a init method
        h_V_vector = tensor_to_vec(h_V_vector)

        def init_vec_with_true_atom(h_V_vector, V_xyz):
            #  h_V_vector: (batch_size, 32, 3)
            #  V_xyz: (batch_size, 3, 3)
            #  return: (batch_size, 32, 3)
            #  32 x 3 -> 4x8x3
            seq_len, vec_num  = h_V_vector.shape[0], h_V_vector.shape[1]
            h_V_vector = h_V_vector.view(-1, 4, vec_num//4, 3)
            # V_xyz: (batch_size, 3, 3) -> batch_size x 4 x 3, padding 0 , Calpha (0,0,0)
            V_xyz = torch.cat([torch.zeros(V_xyz.shape[0], 1, 3).to(V_xyz.device), V_xyz], dim=1)
            h_V_vector[:,:,0,:] = V_xyz
            # 4x8x3 -> 32x3
            h_V_vector = h_V_vector.view(seq_len, vec_num, 3)
            return h_V_vector
        
        h_V_vector = init_vec_with_true_atom(h_V_vector, ori_V_xyz)
        h_V, h_P, h_V_vector = self.encoder(h_V, h_P, P_idx, batch_id, Q_neighbors, h_V_vector,ori_V_xyz)


        logits = self.decoder(h_V, batch_id)
        return logits