import torch
from unifold.data import residue_constants as rc
import numpy as np
import time
from unicore.logging import metrics

def compute_rmsd(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    msd = np.mean(sd)
    return np.sqrt(msd + eps)


def compute_tm(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    sd = np.square(true_atom_pos - pred_atom_pos).sum(axis=-1)
    num_res = true_atom_pos.shape[0]
    d0 = 1.24 * (num_res - 15) ** (1.0 / 3) - 1.8
    nsd = 1.0 / (1.0 + (sd) / (d0**2.0))
    return nsd.mean()


def compute_gdt(true_atom_pos, pred_atom_pos, eps: float = 1e-6):
    d = np.sqrt(np.square(true_atom_pos - pred_atom_pos).sum(axis=-1))

    def p(d, k):
        return (d <= k).astype(np.float32).sum() / d.size

    p0_5 = p(d, 0.5)
    p1 = p(d, 1)
    p2 = p(d, 2)
    p4 = p(d, 4)
    p8 = p(d, 8)
    return 0.25 * (p1 + p2 + p4 + p8), 0.25 * (p0_5 + p1 + p2 + p4)


def compute_lddt(
    true_atom_pos,
    pred_atom_pos,
    cutoff: float = 15.0,
    eps: float = 1e-10,
):
    n = true_atom_pos.shape[-2]
    dmat_true = np.sqrt(
        eps
        + np.sum(
            (true_atom_pos[..., None, :] - true_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )

    dmat_pred = np.sqrt(
        eps
        + np.sum(
            (pred_atom_pos[..., None, :] - pred_atom_pos[..., None, :, :]) ** 2,
            axis=-1,
        )
    )
    dists_to_score = (dmat_true < cutoff).astype(np.float32) * (1.0 - np.eye(n))

    dist_l1 = np.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).astype(np.float32)
        + (dist_l1 < 1.0).astype(np.float32)
        + (dist_l1 < 2.0).astype(np.float32)
        + (dist_l1 < 4.0).astype(np.float32)
    )
    score = score * 0.25

    norm = 1.0 / (eps + np.sum(dists_to_score, axis=-1))
    score = norm * (eps + np.sum(dists_to_score * score, axis=-1))
    return score.mean()

def kabsch_rotation(P, Q):
    C = P.transpose(-1, -2) @ Q
    V, _, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0
    if d:
        V[:, -1] = -V[:, -1]
    U = V @ W
    return U


def get_optimal_transform(src_atoms, tgt_atoms):
    src_center = src_atoms.mean(-2)[None, :]
    tgt_center = tgt_atoms.mean(-2)[None, :]
    r = kabsch_rotation(src_atoms - src_center, tgt_atoms - tgt_center)
    x = tgt_center - src_center @ r
    return r, x

def cul_matrix(gt_coords, pred_coords):
    r, x = get_optimal_transform(pred_coords, gt_coords)
    pred_coords = pred_coords @ r + x
    best_rmsd = compute_rmsd(gt_coords, pred_coords)
    best_tm = compute_tm(gt_coords, pred_coords)
    best_lddt = compute_lddt(gt_coords, pred_coords)
    best_gdt_ts, best_gdt_ha = compute_gdt(gt_coords, pred_coords)
    # return best_rmsd, best_tm, best_lddt, best_gdt_ts, best_gdt_ha
    return {
        "rmsd": float(best_rmsd),
        "tm": float(best_tm),
        "lddt": float(best_lddt),
        "gdt_ts": float(best_gdt_ts),
        "gdt_ha": float(best_gdt_ha),
    }

def cum_dict_value(cum_target_dict, new_input_dict):
    for k,v in new_input_dict.items():
        cum_target_dict[k]+=v
    return cum_target_dict

def norm_dict_value(input_dict, norm_factor):
    for k,v in input_dict.items():
        input_dict[k]/=norm_factor
    return input_dict

def eval_matrix(pred_atom_positions,true_atom_positions,atom_mask,loss_dict,rm=False):
    torch.cuda.synchronize()
    metrics.log_start_time('eval_time' if not rm else 'eval_time_rm', priority=790, round=5)

    pred_atom_positions = pred_atom_positions.detach()
    
    start_time = time.time()

    batch_size = atom_mask.shape[0]
    atom_mask = atom_mask.bool()
    
    bb_matrixs = {"rmsd": float(0),"tm": float(0),"lddt": float(0),"gdt_ts": float(0),"gdt_ha": float(0),}
    all_atom_matrixs = {"rmsd": float(0),"tm": float(0),"lddt": float(0),"gdt_ts": float(0),"gdt_ha": float(0),}
    rmsd_all, tm_all, lddt_all, gdt_ts_all, gdt_ha_all = 0,0,0,0,0
    for bs_id in range(batch_size):
        atom_mask_single = atom_mask[bs_id]
        pred_atom_positions_single = pred_atom_positions[bs_id]
        true_atom_positions_single = true_atom_positions[bs_id]

        ca_idx = rc.atom_order["CA"]
        bb_atom_mask_single = atom_mask_single[:,ca_idx]
        bb_pred_atom_positions_single = pred_atom_positions_single[:,ca_idx]
        bb_true_atom_positions_single = true_atom_positions_single[:,ca_idx]

        pred_atom_positions_single = pred_atom_positions_single[atom_mask_single]
        true_atom_positions_single = true_atom_positions_single[atom_mask_single]

        bb_pred_atom_positions_single = bb_pred_atom_positions_single[bb_atom_mask_single]
        bb_true_atom_positions_single = bb_true_atom_positions_single[bb_atom_mask_single]

        pred_atom_positions_single = pred_atom_positions_single.cpu().numpy()
        true_atom_positions_single = true_atom_positions_single.cpu().numpy()
        bb_pred_atom_positions_single = bb_pred_atom_positions_single.cpu().numpy()
        bb_true_atom_positions_single = bb_true_atom_positions_single.cpu().numpy()

        bb_matrix = cul_matrix(bb_true_atom_positions_single,bb_pred_atom_positions_single)
        all_atom_matrix = cul_matrix(true_atom_positions_single,pred_atom_positions_single)

        bb_matrixs = cum_dict_value(bb_matrixs,bb_matrix)
        all_atom_matrixs = cum_dict_value(all_atom_matrixs,all_atom_matrix)

    bb_matrixs = norm_dict_value(bb_matrixs,batch_size)
    all_atom_matrixs = norm_dict_value(all_atom_matrixs,batch_size)

    for k,v in bb_matrixs.items():
        loss_dict.update({k+'_bb' if not rm else k +'_bb_rm':torch.tensor([v])})

    for k,v in all_atom_matrixs.items():
        loss_dict.update({k if not rm else k +'_rm':torch.tensor([v])})
    
    metrics.log_stop_time('eval_time' if not rm else 'eval_time_rm')
    # end_time = time.time()
    # total_time = end_time - start_time
    # loss_dict.update({'eval_time' if not rm else 'eval_time_rm':torch.tensor([total_time])})

    return torch.tensor([0.0]).to(atom_mask.device).type(pred_atom_positions.dtype)