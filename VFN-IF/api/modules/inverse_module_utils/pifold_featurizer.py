import torch
import torch.nn as nn
from .pifold_utils import gather_nodes, _dihedrals, _get_rbf, _orientations_coarse_gl_tuple_uni_backen
import logging

class Pifold_featurizer(nn.Module):
    def __init__(self, cfg):
        super(Pifold_featurizer, self).__init__()
        self.top_k = cfg.topk
        self.virtual_num = cfg.virtual_num
        self.virtual_atoms = nn.Parameter(torch.rand(self.virtual_num,3))
        self.num_rbf = cfg.num_rbf
        self.node_dist = cfg.node_dist
        self.node_angle = cfg.node_angle
        self.node_direct = cfg.node_direct
        self.edge_dist = cfg.edge_dist
        self.edge_angle = cfg.edge_angle
        self.edge_direct = cfg.edge_direct
        self.fix_bug = cfg.fix_bug
        self.div = cfg.div
        self.num_drop_edge = cfg.num_drop_edge
        self.num_mask_edge = cfg.num_mask_edge
        self.shuffle_aug = cfg.shuffle_aug
        assert not (self.num_mask_edge>0 and self.num_drop_edge>0)
        
        self.xyz_path = 'dataset/xyz/'
        # self.xyz_mean = torch.load(os.path.join(self.xyz_path,'mean.pt')) # 3x3
        self.xyz_mean = torch.tensor([[-5.2663e-01,  1.3599e+00, -1.2915e-10],
                                      [ 1.5254e+00, -6.1342e-09, -2.8615e-10],
                                      [ 2.1513e+00, -1.4534e-01, -1.3165e-02]], device='cuda:0')
        # self.xyz_std = torch.load(os.path.join(self.xyz_path,'std.pt')) # 3x3
        self.xyz_std = torch.tensor([[6.4174e-02, 2.5685e-02, 3.8159e-08],
                                     [1.1167e-02, 6.7004e-08, 3.5351e-08],
                                     [2.1300e-02, 8.1443e-01, 6.6397e-01]], device='cuda:0')
        if not self.num_mask_edge>=0 and cfg.shuffle_aug and self.num_drop_edge==0:
            logging.warning('featurizer: not augmentation')

        self._init_params()

    def _init_params(self):
        for name, p in self.named_parameters():
            if name == 'virtual_atoms':
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = (1. - mask_2D)*10000 + mask_2D* torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * (D_max+1)
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False) # TODO weian: bug, forget to mask unknow res
        return D_neighbors, E_idx  
    
    def forward(self, S, X, mask, gt_frame,atoms14,atoms14_mask,atoms14_alt,atoms14_alt_mask,atoms14_ambiguous_mask):
        score = torch.ones_like(mask).type(X.dtype)
        device = X.device
        # mask_bool = (mask==1) #TODO weian: it is bug; the sum is different
        mask_bool = mask.bool()
        B, N, _,_ = X.shape

        if self.training and self.div>0.0:
            noise_frame = gt_frame[:]
            div = torch.randn(noise_frame._t.size()[0],device=noise_frame._t.device) * self.div
            noise = torch.randn(noise_frame._t.size(),device=noise_frame._t.device) * div[:,None,None]
            noise_frame._t = noise_frame._t + noise
            X_local = gt_frame[...,None].invert_apply(X) # noise 3
            X_nosie = noise_frame[...,None].apply(X_local)
        else:
            X_nosie = X
            noise_frame = gt_frame

        X_ca = noise_frame._t
        D_neighbors, E_idx = self._full_dist(X_ca, mask, self.top_k)
        if self.training and (self.num_drop_edge!=0):
            # top_k_aug = int(self.top_k*0.9)
            top_k_aug = self.top_k - self.num_drop_edge
            E_idx_index = torch.topk(torch.rand_like(E_idx.float()),k=top_k_aug)[1]
            E_idx = torch.gather(E_idx,dim=-1,index=E_idx_index)
        else:
            top_k_aug = self.top_k


        mask_attend = gather_nodes(mask.unsqueeze(-1), E_idx).squeeze(-1)
        mask_attend = (mask.unsqueeze(-1) * mask_attend) == 1
        edge_mask_select = lambda x:  torch.masked_select(x, mask_attend.unsqueeze(-1)).reshape(-1,x.shape[-1])
        node_mask_select = lambda x: torch.masked_select(x, mask_bool.unsqueeze(-1)).reshape(-1, x.shape[-1])


        if self.training and (self.num_mask_edge!=0):
            mask_index = torch.topk(torch.rand_like(E_idx.half()),k=self.num_mask_edge)[1]
            mask_edge = torch.zeros_like(E_idx,dtype=torch.bool)
            ones_src = torch.ones_like(mask_index,dtype=torch.bool)
            mask_edge = mask_edge.scatter(dim=-1,index=mask_index,src=ones_src)
            mask_edge = edge_mask_select(mask_edge[...,None]).squeeze()
            if self.shuffle_aug:
                shuffle_index = torch.argsort(torch.rand_like(mask_edge.half()))
                mask_edge = mask_edge[shuffle_index]
        else:
            mask_edge = None

        


        if score is not None:
            score = torch.masked_select(score, mask_bool)

        # angle & direction
        V_angles = _dihedrals(X_nosie, 0) 
        V_angles = node_mask_select(V_angles)
        # atoms14 B, N, 14, 3 -> B, N, 14*3
        B, N, _, _ = atoms14.shape
        atoms14 = gt_frame[:,:,None].invert_apply(atoms14)
        atoms14_alt = gt_frame[:,:,None].invert_apply(atoms14_alt)
        atoms14 = atoms14.reshape(B, N, -1)
        atoms14_alt = atoms14_alt.reshape(B, N, -1)
        atoms14_mask = node_mask_select(atoms14_mask)
        atoms14_alt_mask = node_mask_select(atoms14_alt_mask)
        atoms14_ambiguous_mask = node_mask_select(atoms14_ambiguous_mask)
        atoms14 = node_mask_select(atoms14)
        atoms14_alt = node_mask_select(atoms14_alt)
        atoms14 = atoms14.reshape(-1,14,3)
        atoms14_alt = atoms14_alt.reshape(-1,14,3)
        valid_atoms14 = atoms14[atoms14_mask.bool()]
        #print(valid_atoms14.shape,valid_atoms14.max())
        V_direct, E_direct, E_angles, Q_neighbors = _orientations_coarse_gl_tuple_uni_backen(X_nosie, E_idx, gt_frame = noise_frame) #TODO weian: padding bug

        V_direct = node_mask_select(V_direct)
        E_direct = edge_mask_select(E_direct)
        E_angles = edge_mask_select(E_angles)
        Q_neighbors = Q_neighbors[mask_attend]

        # distance
        atom_N = X_nosie[:,:,0,:]
        atom_Ca = X_nosie[:,:,1,:]
        atom_C = X_nosie[:,:,2,:]
        atom_O = X_nosie[:,:,3,:]
        b = atom_Ca - atom_N
        c = atom_C - atom_Ca
        a = torch.cross(b, c, dim=-1)


        # following code contain model

        if self.virtual_num>0:
            virtual_atoms = self.virtual_atoms / torch.norm(self.virtual_atoms, dim=1, keepdim=True)
            for i in range(self.virtual_atoms.shape[0]):
                vars()['atom_v' + str(i)] = virtual_atoms[i][0] * a \
                                        + virtual_atoms[i][1] * b \
                                        + virtual_atoms[i][2] * c \
                                        + 1 * atom_Ca

        node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C']
        node_dist = []
        for pair in node_list:
            atom1, atom2 = pair.split('-')
            node_dist.append( node_mask_select(_get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], None, self.num_rbf).squeeze()))
        
        if self.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                # # true atoms
                for j in range(0, i):
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], None, self.num_rbf).squeeze()))
                    node_dist.append(node_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], None, self.num_rbf).squeeze()))
        V_dist = torch.cat(tuple(node_dist), dim=-1).squeeze()
        # get x,y,z
        Calpha_mask = torch.tensor([True, False, True, True], dtype=torch.bool)
        X_local_noise = gt_frame[:,:,None].invert_apply(X_nosie)
        Calpha = X_local_noise[:, :, 1:2, :]
        X_feat = X_local_noise[:, :, Calpha_mask, :] - Calpha
        # X_feat = gt_frame[:,:,None].invert_apply(X_feat)
        X_feat = X_feat.reshape(X.shape[0], X.shape[1], 3 * 3)
        #atom_under_bb = gt_frame[:,:,None,None].invert_apply(atom_global) # ([1, 222, 1, 1])->([1, 222, 30, 4, 3])
        V_xyz = node_mask_select(X_feat)
        # V_xyz = V_xyz.view(-1)
        # V_xyz = torch.clamp(V_xyz, -3, 3) #  seq_len * 3 * 3 -> seq_len x 3 x 3
        V_xyz = V_xyz.reshape(V_dist.shape[0], 3, 3) # N C O
        # use self.xyz_mean and self.xyz_std to filter the V_xyz
        clamp_max = self.xyz_mean + 8 * self.xyz_std  # 3x3
        clamp_min = self.xyz_mean - 8 * self.xyz_std  # 3x3
        V_xyz_flatten = V_xyz.reshape(-1, 9)
        clamp_max = clamp_max.reshape(1, 9)
        clamp_min = clamp_min.reshape(1, 9)
        V_xyz_flatten_ori = V_xyz_flatten
        V_xyz_flatten = torch.clamp(V_xyz_flatten, clamp_min, clamp_max)
        V_xyz = V_xyz_flatten.reshape(V_xyz.shape)
        masked = torch.eq(V_xyz_flatten_ori,V_xyz_flatten)
        masked_indices = torch.nonzero(masked == False)
        # print("masked_indices", masked_indices.shape)
        # print("N", V_xyz[:10, 0, :])
        # print("C", V_xyz[:10, 1, :])
        # print("O", V_xyz[:10, 2, :])
        # save V_xyz to file
       
        
        if V_xyz.max() > 3 or V_xyz.min() < -3: # TODO zmz: need to be modified
            # find which node is wrong, and print x y z
            index = torch.where(torch.abs(V_xyz)>3) 
            # replace abnormal value with mean value
            V_xyz[index] = self.xyz_mean[index[1], index[2]]
        ori_V_xyz = V_xyz
        V_xyz = torch.cat([torch.zeros(V_xyz.shape[0], 1, 3, device = V_xyz.device), V_xyz], dim=1)
        gt_frame_selected = gt_frame[mask_bool]
        V_xyz = gt_frame_selected[:,None].apply(V_xyz)
        V_xyz_left = torch.cat([V_xyz[1:],V_xyz[-1,0][None,None].repeat(1,4,1)])
        V_xyz_right = torch.cat([V_xyz[0,0][None,None].repeat(1,4,1),V_xyz[:-1]])
        V_xyz = torch.cat([V_xyz,V_xyz_left,V_xyz_right],dim=1)
        V_xyz = gt_frame_selected[:,None].invert_apply(V_xyz)
        
        

        pair_lst = ['Ca-Ca', 'Ca-C', 'C-Ca', 'Ca-N', 'N-Ca', 'Ca-O', 'O-Ca', 'C-C', 'C-N', 'N-C', 'C-O', 'O-C', 'N-N', 'N-O', 'O-N', 'O-O']

        edge_dist = [] #Ca-Ca
        for pair in pair_lst:
            atom1, atom2 = pair.split('-')
            rbf = _get_rbf(vars()['atom_' + atom1], vars()['atom_' + atom2], E_idx, self.num_rbf)
            edge_dist.append(edge_mask_select(rbf))

        if self.virtual_num>0:
            for i in range(self.virtual_atoms.shape[0]):
                edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))
                for j in range(0, i):
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(i)], vars()['atom_v' + str(j)], E_idx, self.num_rbf)))
                    edge_dist.append(edge_mask_select(_get_rbf(vars()['atom_v' + str(j)], vars()['atom_v' + str(i)], E_idx, self.num_rbf)))

        
        E_dist = torch.cat(tuple(edge_dist), dim=-1)

        h_V = []
        if self.node_dist:
            h_V.append(V_dist)
        if self.node_angle:
            h_V.append(V_angles)
        if self.node_direct:
            h_V.append(V_direct)
        
        h_E = []
        if self.edge_dist:
            h_E.append(E_dist)
        if self.edge_angle:
            h_E.append(E_angles)
        if self.edge_direct:
            h_E.append(E_direct)
        
        _V = torch.cat(h_V, dim=-1)
        _E = torch.cat(h_E, dim=-1)

        src = E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        dst = torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx_w_gap = torch.cat((dst, src), dim=0).long()

        mask_index = mask.cumsum(dim=1)-1 # weian: there are some gap in 
        mask_src_index = mask_index[:,None].repeat(1,mask_index.shape[1],1)
        mask_dst_index = mask_index[:,:,None].repeat(1,1,top_k_aug)
        E_idx = torch.gather(mask_src_index,dim=-1,index = E_idx)
        # edge index
        shift = mask.sum(dim=1).cumsum(dim=0) - mask.sum(dim=1)
        src = shift.view(B,1,1) + E_idx
        src = torch.masked_select(src, mask_attend).view(1,-1)
        # dst = shift.view(B,1,1) + torch.arange(0, N, device=src.device).view(1,-1,1).expand_as(mask_attend)
        dst = shift.view(B,1,1) + mask_dst_index
        dst = torch.masked_select(dst, mask_attend).view(1,-1)
        E_idx = torch.cat((dst, src), dim=0).long()

        # decoding_order = node_mask_select((decoding_order+shift.view(-1,1)).unsqueeze(-1)).squeeze().long()
        
        # 3D point
        sparse_idx = mask.nonzero()  # index of non-zero values
        X_nosie = X_nosie[sparse_idx[:,0], sparse_idx[:,1], :, :]
        batch_id = sparse_idx[:,0]
        V_xyz = (V_xyz,ori_V_xyz)
        return _V, _E, E_idx, batch_id, mask_edge, E_idx_w_gap, Q_neighbors, V_xyz,atoms14,atoms14_mask,atoms14_alt, atoms14_alt_mask,atoms14_ambiguous_mask