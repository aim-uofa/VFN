import torch
import torch.nn.functional as F
from typing import Dict
from einops import rearrange,repeat

from unicore.utils import one_hot
from .utils import masked_mean
from api.data import residue_constants as rc

def inverse_cls_loss(
    res_score: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    loss_dict=None,
) -> torch.Tensor:
    assert not (aatype[seq_mask.bool()]>20).any()

    seq_mask = seq_mask.float()
    seq_mask = seq_mask.bool()
    total_res = seq_mask.sum()
    pred_cls = torch.argmax(res_score,dim=-1)
    corr_num = (aatype[seq_mask] == pred_cls).sum()
    acc = corr_num / total_res
    loss_dict["acc"] = acc.data
    loss_dict["corr_num"] = corr_num.float().data
    loss_dict["total_res"] = total_res.float().data


    aatype_gt = aatype[seq_mask]
    assert (aatype_gt<=19).all()
    loss = F.cross_entropy(res_score,aatype_gt,reduction='mean',label_smoothing=0.0)

    # res_score = rearrange(res_score,'i b r t -> b t r i')
    # aatype = repeat(aatype,'b r -> b r i', i = res_score.shape[-1])
    # loss = F.cross_entropy(res_score,aatype,reduction='none') # TODO weian: use the unicore softmax_cross_entropy implement
    # loss_mask = repeat(seq_mask,'b r -> b (r i)', i = res_score.shape[-1])
    # loss = rearrange(loss,'b r i -> b (r i)')
    # loss = masked_mean(loss_mask, loss, dim=-1)
    # loss *= loss_mask
    # loss = loss.flatten(start_dim=1).sum(-1,keepdim=True) / loss_mask.sum() # TODO weian: check the loss shape
    #loss_dict["res_loss"] = F.cross_entropy(res_score,aatype_gt,reduction='none').data
    loss_dict["loss"] = loss.data

    loss = loss.float()
    torch.isnan(loss)
    #print('cls_loss: ',loss)
    return loss

def edge_cls_loss(
    edge_score: torch.Tensor,
    edge_type: torch.Tensor,
    edge_mask: torch.Tensor,
    loss_dict=None,
) -> torch.Tensor:
    # edge_mask = edge_mask.bool()
    total_edge = edge_score.shape[0]
    if edge_type.shape[-1] !=3:
        pred_cls = torch.argmax(edge_score,dim=-1)
        # corr_num = (edge_type[edge_mask] == pred_cls[edge_mask]).sum()
        corr_num = (edge_type == pred_cls).sum()
        acc = corr_num / total_edge
        loss_dict["acc_edge"] = acc.data
        loss_dict["corr_num_edge"] = corr_num.float().data
        # loss_dict["total_edge"] = total_edge
        # if edge_score.shape[-1] == 2:
        #     loss_weight = torch.ones(2).to(edge_score.device)
        #     loss_weight[0] = 1/30
        # else: 
        #     loss_weight = None  # for hbound we don't need use weight
        loss_weight = None
        loss = F.cross_entropy(edge_score,edge_type,reduction='mean',weight=loss_weight)
    else:
        # use mse loss
        loss = F.mse_loss(edge_score,edge_type,reduction='mean')

    loss_dict["loss_edge"] = loss.data

    loss = loss.float()
    return loss

def side_chain_reconstruction(
    side_chain_pred: torch.Tensor,
    side_chain_gt: torch.Tensor,
    edge_mask: torch.Tensor,
    loss_dict=None,
    side_chain_gt_alt = None,
) -> torch.Tensor:
    assert side_chain_gt.shape[-1] == 3
    edge_mask = edge_mask.bool()
    # also remove abnormals 
    if side_chain_gt_alt is None:
        if not isinstance(side_chain_pred,list):
            side_chain_pred = side_chain_pred[edge_mask]
            side_chain_gt = side_chain_gt[edge_mask]
            assert side_chain_gt.max() < 8
            loss = F.mse_loss(side_chain_pred,side_chain_gt,reduction='mean')

            loss_dict["loss_sidechain"] = loss.data

            loss = loss.float()
        else:
            loss = 0
            # 为10层分层不同的权重
            weight_list = [0.1,0.2,0.2,0.5,0.5,0.5,0.5,0.8,0.8,0.8]
            # 和为1
            # weight_list = [i/sum(weight_list) for i in weight_list]
            for i in range(len(side_chain_pred)):
                side_chain_pred_i = side_chain_pred[i][edge_mask]
                side_chain_gt_i = side_chain_gt[edge_mask]
                loss_i = F.mse_loss(side_chain_pred_i,side_chain_gt_i,reduction='mean')
                loss += loss_i*weight_list[i]
                loss_dict["loss_sidechain_{}".format(i)] = loss_i.data
            #loss /= len(side_chain_pred) # 不用除
            loss_dict["loss_sidechain"] = loss.data
            loss = loss.float()
    else:
        if not isinstance(side_chain_pred,list):
            side_chain_pred = side_chain_pred[edge_mask]
            side_chain_gt = side_chain_gt[edge_mask]
            side_chain_gt_alt = side_chain_gt_alt[edge_mask]
            assert side_chain_gt.max() < 8
            loss1 = F.mse_loss(side_chain_pred,side_chain_gt,reduction='none')
            loss2 = F.mse_loss(side_chain_pred,side_chain_gt_alt,reduction='none')
            loss1 = loss1.mean(dim=-1)
            loss2 = loss2.mean(dim=-1)
            loss = torch.min(loss1,loss2)
            loss = loss.mean()
            loss_dict["loss_sidechain"] = loss.data
            loss = loss.float()
        else:
            loss1 = 0
            loss2 = 0
            # 为10层分层不同的权重
            weight_list = [0.1,0.2,0.2,0.5,0.5,0.5,0.5,0.8,0.8,0.8]
            # 和为1
            # weight_list = [i/sum(weight_list) for i in weight_list]
            for i in range(len(side_chain_pred)):
                side_chain_pred_i = side_chain_pred[i][edge_mask]
                side_chain_gt_i = side_chain_gt[edge_mask]
                side_chain_gt_alt_i = side_chain_gt_alt[edge_mask]
                loss1_i = F.mse_loss(side_chain_pred_i,side_chain_gt_i,reduction='none')
                loss2_i = F.mse_loss(side_chain_pred_i,side_chain_gt_alt_i,reduction='none')
                loss_i = torch.min(loss1_i,loss2_i)
                loss_i = loss_i.mean()
                loss1 += loss1_i*weight_list[i]
                loss2 += loss2_i*weight_list[i]
                loss_dict["loss_sidechain_{}".format(i)] = loss_i.data
            #loss /= len(side_chain_pred) # 不用除
            loss = torch.min(loss1,loss2)
            loss = loss.mean()
            loss_dict["loss_sidechain"] = loss.data
            loss = loss.float()
    #print('side_chain',loss)
    return loss

def inverse_cls_loss_backup(
    res_score: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    loss_dict=None,
) -> torch.Tensor:
    assert not (aatype[seq_mask.bool()]>20).any()

    seq_mask = seq_mask.bool()
    total_res = seq_mask.sum()
    pred_cls = torch.argmax(res_score[-1],dim=-1)
    corr_num = (aatype[seq_mask] == pred_cls[seq_mask]).sum()
    acc = corr_num / total_res
    loss_dict["acc"] = acc.data
    loss_dict["corr_num"] = corr_num.float().data
    loss_dict["total_res"] = total_res.float().data


    res_score = rearrange(res_score,'i b r t -> b t r i')
    aatype = repeat(aatype,'b r -> b r i', i = res_score.shape[-1])
    loss = F.cross_entropy(res_score,aatype,reduction='none') # TODO weian: use the unicore softmax_cross_entropy implement
    loss_mask = repeat(seq_mask,'b r -> b (r i)', i = res_score.shape[-1])
    loss = rearrange(loss,'b r i -> b (r i)')
    loss = masked_mean(loss_mask, loss, dim=-1)
    # loss *= loss_mask
    # loss = loss.flatten(start_dim=1).sum(-1,keepdim=True) / loss_mask.sum() # TODO weian: check the loss shape

    loss_dict["loss"] = loss.data

    loss = loss.float()
    return loss