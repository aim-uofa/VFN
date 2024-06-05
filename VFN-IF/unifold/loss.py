import logging
import torch

from unicore import metrics
from unicore.utils import tensor_tree_map
from unicore.losses import UnicoreLoss, register_loss
from unicore.data import data_utils

from unifold.losses.geometry import compute_renamed_ground_truth, compute_metric
from unifold.losses.violation import find_structural_violations, violation_loss
from unifold.losses.inverse import inverse_cls_loss, edge_cls_loss, side_chain_reconstruction
from unifold.losses.fape import fape_loss,fape_rm_loss
from unifold.losses.auxillary import (
    chain_centre_mass_loss,
    distogram_loss,
    experimentally_resolved_loss,
    masked_msa_loss,
    pae_loss,
    plddt_loss,
    repr_norm_loss,
    masked_msa_loss,
    supervised_chi_loss,
)
from unifold.losses.eval import eval_matrix
from unifold.losses.chain_align import multi_chain_perm_align


@register_loss("af2")
class AlphafoldLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, batch, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # assert batch['aatype'].shape[0]==1
        batch['seq_mask'][(batch['aatype']>=20)] = 0
        t = batch['true_frame_tensor'][..., :3, 3]
        t = t[0]
        unknown_loc_mask = ((t == 0).sum(-1) == 3)
        batch['seq_mask'][unknown_loc_mask[None]] = 0

        # return config in model.
        out, config = model(batch)
        # num_recycling = batch["msa_feat"].shape[0]
        
        # remove recyling dim
        batch = tensor_tree_map(lambda t: t[-1, ...], batch)
        # assert  pred.shape[0] == batch['aatype'].shape[0]
        loss, sample_size, logging_output = self.loss(out, batch, config)
        #pred_prob = F.softmax(res_score,dim=-1)
        # # # 
        # logging_output['pred_prob'] =  out['res_score'].softmax(-1)
        # pred = out['res_score'].argmax(-1)
        # logging_output["pred"] = pred
        # logging_output["seq_mask"] = batch["seq_mask"]
        # # #
        # logging_output["seq_mask"] = batch["seq_mask"]
        # logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output

    def loss(self, out, batch, config):

        if 'res_score' not in out:
            if "violation" not in out.keys() and config.violation.weight:
                out["violation"] = find_structural_violations(
                    batch, out["sm"]["positions"], **config.violation)

            if "renamed_atom14_gt_positions" not in out.keys():
                batch.update(
                    compute_renamed_ground_truth(batch, out["sm"]["positions"]))

            vaild_enable = not out["final_atom_positions_rm"].requires_grad
            out['res_score'] = None
        
        loss_dict = {}
        loss_fns = {
            "chain_centre_mass": lambda: chain_centre_mass_loss(
                pred_atom_positions=out["final_atom_positions"],
                true_atom_positions=batch["all_atom_positions"],
                atom_mask=batch["all_atom_mask"],
                asym_id=batch["asym_id"],
                **config.chain_centre_mass,
                loss_dict=loss_dict,
            ),
            "distogram": lambda: distogram_loss(
                logits=out["distogram_logits"],
                pseudo_beta=batch["pseudo_beta"],
                pseudo_beta_mask=batch["pseudo_beta_mask"],
                **config.distogram,
                loss_dict=loss_dict,
            ),
            "experimentally_resolved": lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                atom37_atom_exists=batch["atom37_atom_exists"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.experimentally_resolved,
                loss_dict=loss_dict,
            ),
            "fape": lambda: fape_loss(
                out,
                batch,
                config.fape,
                loss_dict=loss_dict,
            ),
            "fape_rm": lambda: fape_rm_loss(
                out,
                batch,
                config.fape,
                loss_dict=loss_dict,
            ),
            "masked_msa": lambda: masked_msa_loss(
                logits=out["masked_msa_logits"],
                true_msa=batch["true_msa"],
                bert_mask=batch["bert_mask"],
                loss_dict=loss_dict,
            ),
            "pae": lambda: pae_loss(
                logits=out["pae_logits"],
                pred_frame_tensor=out["pred_frame_tensor"],
                true_frame_tensor=batch["true_frame_tensor"],
                frame_mask=batch["frame_mask"],
                resolution=batch["resolution"],
                **config.pae,
                loss_dict=loss_dict,
            ),
            "plddt": lambda: plddt_loss(
                logits=out["plddt_logits"],
                all_atom_pred_pos=out["final_atom_positions"],
                all_atom_positions=batch["all_atom_positions"],
                all_atom_mask=batch["all_atom_mask"],
                resolution=batch["resolution"],
                **config.plddt,
                loss_dict=loss_dict,
            ),
            "repr_norm": lambda: repr_norm_loss(
                out["delta_msa"],
                out["delta_pair"],
                out["msa_norm_mask"],
                batch["pseudo_beta_mask"],
                **config.repr_norm,
                loss_dict=loss_dict,
            ),
            "supervised_chi": lambda: supervised_chi_loss(
                pred_angles_sin_cos=out["sm"]["angles"],
                pred_unnormed_angles_sin_cos=out["sm"]["unnormalized_angles"],
                true_angles_sin_cos=batch["chi_angles_sin_cos"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                chi_mask=batch["chi_mask"],
                **config.supervised_chi,
                loss_dict=loss_dict,
            ),
            "supervised_chi_rm": lambda: supervised_chi_loss(
                pred_angles_sin_cos=out["rm"]["angles"],
                pred_unnormed_angles_sin_cos=out["rm"]["unnormalized_angles"],
                true_angles_sin_cos=batch["chi_angles_sin_cos"],
                aatype=batch["aatype"],
                seq_mask=batch["seq_mask"],
                chi_mask=batch["chi_mask"],
                **config.supervised_chi,
                loss_dict=loss_dict,
                rm = True,
            ),
            "violation": lambda: violation_loss(
                out["violation"],
                loss_dict=loss_dict,
                bond_angle_loss_weight=config.violation.bond_angle_loss_weight,
            ),
            "inverse_cls": lambda: inverse_cls_loss(
                res_score = out["res_score"],
                aatype = batch["aatype"],
                seq_mask = batch["seq_mask"],
                loss_dict=loss_dict,
            ),
            "edge_cls": lambda: edge_cls_loss(
                edge_score = out["edge_score"],
                edge_type = out["edge_type"],
                edge_mask = batch["seq_mask"],
                loss_dict=loss_dict,
            ),
            "side_chain_reconstruction": lambda: side_chain_reconstruction(
                side_chain_pred= out["side_chain_pred"],
                side_chain_gt = out["atoms14"],
                edge_mask= out["atoms14_mask"],
                loss_dict=loss_dict,
                side_chain_gt_alt = out.get("atoms14_alt", None),
            ),
            
            # "violation_rm": lambda: violation_loss(
            #     out["violation_rm"],
            #     loss_dict=loss_dict,
            #     bond_angle_loss_weight=config.violation.bond_angle_loss_weight,
            #     rm = True,
            # ),
            "eval": lambda: eval_matrix(
                pred_atom_positions=out["final_atom_positions"],
                true_atom_positions=batch["all_atom_positions"],
                atom_mask=batch["all_atom_mask"],
                loss_dict=loss_dict,
            ),
            "eval_rm": lambda: eval_matrix(
                pred_atom_positions=out["final_atom_positions_rm"],
                true_atom_positions=batch["all_atom_positions"],
                atom_mask=batch["all_atom_mask"],
                loss_dict=loss_dict,
                rm = True,
            ),
        }

        cum_loss = 0
        bsz = batch["seq_mask"].shape[0]
        with torch.no_grad():
            seq_len = torch.sum(batch["seq_mask"].float(), dim=-1)
            # seq_length_weight = seq_len**0.5 # weian: **0.5 not linear; it is for fape, which complexity is n**2
            seq_length_weight = seq_len
        
        assert (
            len(seq_length_weight.shape) == 1 and seq_length_weight.shape[0] == bsz
        ), seq_length_weight.shape
        
        for loss_name, loss_fn in loss_fns.items():
            weight = config[loss_name].weight
            if weight > 0.:
                loss = loss_fn()
                # always use float type for loss
                assert loss.dtype == torch.float, loss.dtype
                # assert len(loss.shape) == 1 and loss.shape[0] == bsz, loss.shape

                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"{loss_name} loss is NaN. Skipping...")
                    loss = loss.new_tensor(0.0, requires_grad=True)
                
                cum_loss = cum_loss + weight * loss

        for key in loss_dict:
            if key == "res_loss":
                continue
            loss_dict[key] = float((loss_dict[key]).mean())

        # loss = (cum_loss * seq_length_weight).mean() # **0.5 not linear
        # loss = ((cum_loss * seq_length_weight).sum()) / seq_length_weight.sum() # TODO weian: loss must be rewriten to be averaged by sequence length
        loss = cum_loss

        logging_output = loss_dict
        # sample size fix to 1, so the loss (and gradients) will be averaged on all workers.
        sample_size = 1
        logging_output["loss"] = loss.data
        logging_output["bsz"] = bsz
        logging_output["sample_size"] = sample_size
        logging_output["seq_len"] = seq_len
        # logging_output["num_recycling"] = num_recycling
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=4)

        if split=="valid":
            corr_num_sum = sum(log.get('corr_num', 0) for log in logging_outputs)
            total_res_sum = sum(log.get('total_res', 0) for log in logging_outputs)
            acc = corr_num_sum/total_res_sum
            print("acc: ", acc) 
            # 上面是对residue的准确率，下面是对蛋白质的准确率
            mean_acc  = sum (log.get('corr_num', 0)/log.get('total_res', 0) for log in logging_outputs)/ len(logging_outputs)
            print("mean_acc: ", mean_acc)
            median_acc = sorted([log.get('corr_num', 0)/log.get('total_res', 0) for log in logging_outputs])[len(logging_outputs)//2]
            print("orimedian_acc: ", median_acc)
            # use numpy to calculate median
            import numpy as np 
            median_acc = np.median([log.get('corr_num', 0)/log.get('total_res', 0) for log in logging_outputs])
            print("median_acc: ", median_acc)
            worst_acc = sorted([log.get('corr_num', 0)/log.get('total_res', 0) for log in logging_outputs])[0]
            print("worst_acc: ", worst_acc)
            acc = median_acc
            metrics.log_scalar('acc', acc, sample_size, round=4)
        elif split=="train":
            loss_sum = sum(log.get('acc', 0) for log in logging_outputs)
            metrics.log_scalar('acc', loss_sum / sample_size, sample_size, round=4)
        else:
            raise

        for key in logging_outputs[0]:
            if split == "train":
                if key in ["sample_size", "bsz", "seq_len", "acc", "corr_num", "total_res"]:
                    continue
                loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, loss_sum / sample_size, sample_size, round=4)
            else:
                assert split == "valid"
                if key in ["sample_size", "bsz", "seq_len", "acc", "corr_num", "total_res",'res_loss','pred','pred_prob','seq_mask']:
                    continue
                loss_sum = sum(log.get(key, 0) for log in logging_outputs) / sample_size
                print('original loss_sum', loss_sum)
                # this loss sum is averaged use sample_size
                # but we need loss weighted by seq_len
                total_loss = sum(log.get("loss", 0)*log.get('total_res') for log in logging_outputs)
                total_res = sum(log.get('total_res') for log in logging_outputs)
                loss_list = [log.get("loss", 0).item() for log in logging_outputs]
                loss_len_list = [log.get('total_res') for log in logging_outputs]
                import numpy as np
                loss_numpy = np.array(loss_list)
                loss_max = loss_numpy.max()
                loss_min = loss_numpy.min()
                loss_mean = loss_numpy.mean()
                loss_median = np.median(loss_numpy)
                print('loss_max', loss_max)
                print('loss_min', loss_min)
                print('loss_mean', loss_mean)
                print('loss_median', loss_median)
                loss_sum = total_loss/total_res
                perplexity = torch.exp(loss_sum)
                metrics.log_scalar(key, loss_sum, sample_size, round=4)
                metrics.log_scalar('perplexity', perplexity, sample_size, round=4)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_loss("afm")
class AlphafoldMultimerLoss(AlphafoldLoss):
    def forward(self, model, batch, reduce=True):
        features, labels = batch
        assert isinstance(features, dict)

        # return config in model.
        out, config = model(features)
        num_recycling = features["msa_feat"].shape[0]
        
        # remove recycling dim
        features = tensor_tree_map(lambda t: t[-1, ...], features)
        
        # perform multi-chain permutation alignment.
        if labels:
            with torch.no_grad():
                batch_size = out["final_atom_positions"].shape[0]
                new_labels = []
                for batch_idx in range(batch_size):
                    cur_out = {
                        k: out[k][batch_idx]
                        for k in out
                        if k in {"final_atom_positions", "final_atom_mask"}
                    }
                    cur_feature = {k: features[k][batch_idx] for k in features}
                    cur_label = labels[batch_idx]
                    cur_new_labels = multi_chain_perm_align(
                        cur_out, cur_feature, cur_label
                    )
                    new_labels.append(cur_new_labels)
            new_labels = data_utils.collate_dict(new_labels, dim=0)
            
            # check for consistency of label and feature.
            assert (new_labels["aatype"] == features["aatype"]).all()
            features.update(new_labels)

        loss, sample_size, logging_output = self.loss(out, features, config)
        logging_output["num_recycling"] = num_recycling
        
        return loss, sample_size, logging_output
