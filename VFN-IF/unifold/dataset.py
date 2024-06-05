import os
import json
import ml_collections as mlc
import numpy as np
import copy
import torch
from typing import *
from unifold.data import utils
from unifold.data import residue_constants as rc
from unifold.data.data_ops import NumpyDict, TorchDict, inverse_folding_featurizer
from unifold.data.process import process_features, process_labels
from unifold.data.process_multimer import (
    pair_and_merge,
    add_assembly_features,
    convert_monomer_features,
    post_process,
    merge_msas,
)

from unicore.data import UnicoreDataset, data_utils
from unicore.distributed import utils as distributed_utils

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]


import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def make_data_config(
    config: mlc.ConfigDict,
    mode: str,
    num_res: int,
) -> Tuple[mlc.ConfigDict, List[str]]:
    cfg = copy.deepcopy(config)
    mode_cfg = cfg[mode]
    with cfg.unlocked():
        if mode_cfg.crop_size is None:
            mode_cfg.crop_size = num_res
    feature_names = cfg.common.unsupervised_features + cfg.common.recycling_features
    if cfg.common.use_templates:
        feature_names += cfg.common.template_features
    if cfg.common.is_multimer:
        feature_names += cfg.common.multimer_features
    if cfg[mode].supervised:
        feature_names += cfg.supervised.supervised_features

    return cfg, feature_names


def process_label(all_atom_positions: np.ndarray, operation: Operation) -> np.ndarray:
    if operation == "I":
        return all_atom_positions
    rot, trans = operation
    rot = np.array(rot).reshape(3, 3)
    trans = np.array(trans).reshape(3)
    return all_atom_positions @ rot.T + trans


@utils.lru_cache(maxsize=8, copy=True)
def load_single_feature(
    sequence_id: str,
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    is_monomer: bool = False,
) -> NumpyDict:

    monomer_feature = utils.load_pickle(
        os.path.join(monomer_feature_dir, f"{sequence_id}.feature.pkl.gz")
    )
    monomer_feature = convert_monomer_features(monomer_feature)
    chain_feature = {**monomer_feature}

    if uniprot_msa_dir is not None:
        all_seq_feature = utils.load_pickle(
            os.path.join(uniprot_msa_dir, f"{sequence_id}.uniprot.pkl.gz")
        )
        if is_monomer:
            chain_feature["msa"], chain_feature["deletion_matrix"] = merge_msas(
                chain_feature["msa"],
                chain_feature["deletion_matrix"],
                all_seq_feature["msa"],
                all_seq_feature["deletion_matrix"],
            )
        else:
            all_seq_feature = utils.convert_all_seq_feature(all_seq_feature)
            for key in [
                "msa_all_seq",
                "msa_species_identifiers_all_seq",
                "deletion_matrix_all_seq",
            ]:
                chain_feature[key] = all_seq_feature[key]

    return chain_feature


def load_single_label(
    label_id: str,
    label_dir: str,
    symmetry_operation: Optional[Operation] = None,
) -> NumpyDict:
    label = utils.load_pickle(os.path.join(label_dir, f"{label_id}.label.pkl.gz"))
    if symmetry_operation is not None:
        label["all_atom_positions"] = process_label(
            label["all_atom_positions"], symmetry_operation
        )
    label = {
        k: v
        for k, v in label.items()
        if k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution", "aatype_index"]
    }
    return label
restypes = rc.restypes
word2id = { restypes[i] : i for i in range(len(restypes))}
def load_ts50(json_path):
    idx = json_path[1]
    json_path = json_path[0]
    with open(json_path) as f:
        data = json.load(f)
    seq = data[idx]['seq']
    coords = np.array(data[idx]['coords'])
    seq = [word2id[aa] for aa in seq]
    seq = np.array(seq)
    data = {'aatype': seq, 'all_atom_positions': coords}
    label = {}
    assert len(seq) == len(coords)
    return data,label
def load(
    sequence_ids: List[str],
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
) -> NumpyExample:

    all_chain_features = [
        load_single_feature(s, monomer_feature_dir, uniprot_msa_dir, is_monomer)
        for s in sequence_ids
    ]

    if label_ids is not None:
        # load labels
        assert len(label_ids) == len(sequence_ids)
        assert label_dir is not None
        if symmetry_operations is None:
            symmetry_operations = ["I" for _ in label_ids]
        all_chain_labels = [
            load_single_label(l, label_dir, o)
            for l, o in zip(label_ids, symmetry_operations)
        ]
        # update labels into features to calculate spatial cropping etc.
        [f.update(l) for f, l in zip(all_chain_features, all_chain_labels)]

    len_label = all_chain_labels[0]['all_atom_positions'].shape[0]
    len_feat = all_chain_features[0]['aatype'].shape[0]
    assert len_label == len_feat, sequence_ids
    assert (all_chain_labels[0]['aatype_index'] == all_chain_features[0]['aatype']).all()
    all_chain_features = add_assembly_features(all_chain_features)

    # get labels back from features, as add_assembly_features may alter the order of inputs.
    if label_ids is not None:
        all_chain_labels = [
            {
                k: f[k]
                for k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
            }
            for f in all_chain_features
        ]
    else:
        all_chain_labels = None

    asym_len = np.array([c["seq_length"] for c in all_chain_features], dtype=np.int64)
    if is_monomer:
        all_chain_features = all_chain_features[0]
    else:
        all_chain_features = pair_and_merge(all_chain_features)
        all_chain_features = post_process(all_chain_features)
    all_chain_features["asym_len"] = asym_len

    return all_chain_features, all_chain_labels

def load_fast(
    sequence_ids: List[str],
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
) -> NumpyExample:

    # all_chain_features = [
    #     load_single_feature(s, monomer_feature_dir, uniprot_msa_dir, is_monomer)
    #     for s in sequence_ids
    # ]

    if label_ids is not None:
        # load labels
        assert len(label_ids) == len(sequence_ids)
        assert label_dir is not None
        if symmetry_operations is None:
            symmetry_operations = ["I" for _ in label_ids]
        all_chain_labels = [
            load_single_label(l, label_dir, o)
            for l, o in zip(label_ids, symmetry_operations)
        ]
        # update labels into features to calculate spatial cropping etc.
        # [f.update(l) for f, l in zip(all_chain_features, all_chain_labels)]

    # len_label = all_chain_labels[0]['all_atom_positions'].shape[0]
    # len_feat = all_chain_features[0]['aatype'].shape[0]
    # assert len_label == len_feat, sequence_ids
    # assert (all_chain_labels[0]['aatype_index'] == all_chain_features[0]['aatype']).all()
    # all_chain_features = add_assembly_features(all_chain_features)

    # get labels back from features, as add_assembly_features may alter the order of inputs.
    # if label_ids is not None:
    #     all_chain_labels = [
    #         {
    #             k: f[k]
    #             for k in ["aatype", "all_atom_positions", "all_atom_mask", "resolution"]
    #         }
    #         for f in all_chain_features
    #     ]
    # else:
    #     all_chain_labels = None

    # asym_len = np.array([c["seq_length"] for c in all_chain_features], dtype=np.int64)
    # if is_monomer:
    #     all_chain_features = all_chain_features[0]
    # else:
    #     all_chain_features = pair_and_merge(all_chain_features)
    #     all_chain_features = post_process(all_chain_features)
    # all_chain_features["asym_len"] = asym_len

    #check
    # residue_index = all_chain_features['residue_index']
    # assert ((np.arange(len(all_chain_features['aatype']))) == residue_index).all()
    # assert len(all_chain_labels)==1
    # assert (all_chain_labels[0]['aatype_index'] == all_chain_features['aatype']).all()
    # assert (np.array(len(all_chain_labels[0]['aatype_index'])) == all_chain_features['seq_length']).all()

    all_chain_features ={}
    all_chain_features['aatype'] = all_chain_labels[0]['aatype_index']
    all_chain_features['residue_index'] = np.arange(len(all_chain_features['aatype']))
    all_chain_features['seq_length'] = np.array(len(all_chain_features['aatype']))
    [f.update(l) for f, l in zip([all_chain_features], all_chain_labels)]

    # all_chain_labels

    return all_chain_features, all_chain_labels


def process(
    config: mlc.ConfigDict,
    mode: str,
    features: NumpyDict,
    labels: Optional[List[NumpyDict]] = None,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
) -> TorchExample:

    if mode == "train":
        assert batch_idx is not None
        with data_utils.numpy_seed(seed, batch_idx, key="recycling"):
            num_iters = np.random.randint(0, config.common.max_recycling_iters + 1)
            use_clamped_fape = np.random.rand() < config[mode].use_clamped_fape_prob
    else:
        num_iters = config.common.max_recycling_iters
        use_clamped_fape = 1

    features["num_recycling_iters"] = int(num_iters)
    features["use_clamped_fape"] = int(use_clamped_fape)
    features["is_distillation"] = int(is_distillation)
    if is_distillation and "msa_chains" in features:
        features.pop("msa_chains")

    num_res = int(features["seq_length"])
    cfg, feature_names = make_data_config(config, mode=mode, num_res=num_res)

    if labels is not None:
        features["resolution"] = labels[0]["resolution"].reshape(-1)

    with data_utils.numpy_seed(seed, data_idx, key="protein_feature"):
        features["crop_and_fix_size_seed"] = np.random.randint(0, 63355)
        features = utils.filter(features, desired_keys=feature_names)
        features = {k: torch.tensor(v) for k, v in features.items()}
        with torch.no_grad():
            features = process_features(features, cfg.common, cfg[mode])

    if (labels is not None) and (not config.common.fast_mode):
        labels = [{k: torch.tensor(v) for k, v in l.items()} for l in labels]
        with torch.no_grad():
            labels = process_labels(labels)
    else:
        labels = None

    features = inverse_folding_featurizer(features,config)

    return features, labels


def load_and_process(
    config: mlc.ConfigDict,
    mode: str,
    seed: int = 0,
    batch_idx: Optional[int] = None,
    data_idx: Optional[int] = None,
    is_distillation: bool = False,
    **load_kwargs,
):
    is_monomer = (
        is_distillation
        if "is_monomer" not in load_kwargs
        else load_kwargs.pop("is_monomer")
    )
    label_dir = load_kwargs['label_dir']
    if 'ts50' in label_dir[0]:
        features, _ = load_ts50(label_dir)
        features['resolution'] = np.array([1.0])
        features['seq_length'] = np.array([len(features['aatype'])])
        features['residue_index'] = np.arange(len(features['aatype']))
        tmp = features['all_atom_positions']
        features['all_atom_positions'] = np.zeros((len(features['aatype']),37, 3)) 
        # 0,1,2,4 filled with tmp
        index = np.array([0,1,2,4])
        features['all_atom_positions'][:,index,:] = tmp
        features['all_atom_mask'] = np.zeros((len(features['aatype']),37))
        features['all_atom_mask'][:,:5] = 1
        labels = []
        labels.append(features)
    else:
        if config.common.fast_mode:
            features, labels = load_fast(**load_kwargs, is_monomer=is_monomer)
        else:
            features, labels = load(**load_kwargs, is_monomer=is_monomer)
    # features:'aatype', 'residue_index', 'seq_length', 'aatype_index', 'all_atom_positions', 'all_atom_mask', 'resolution'
    # labels[0].keys() (['aatype_index', 'all_atom_positions', 'all_atom_mask', 'resolution'])
    features, labels = process(
        config, mode, features, labels, seed, batch_idx, data_idx, is_distillation
    )
    return features, labels


class UnifoldDataset(UnicoreDataset):
    def __init__(
        self,
        args,
        seed,
        config,
        data_path,
        mode="train",
        max_step=None,
        disable_sd=False,
        json_prefix="",
    ):
        # if mode.startswith("eval"):
        #     if os.path.exists(data_path[:-1]+'_'+mode+'/'):
        #         data_path = data_path[:-1]+'_'+mode+'/'
        self.path = data_path

        def load_json(filename):
            return json.load(open(filename, "r"))
        if 'ts5' in json_prefix:
            # load json file from data_path
            json_path = data_path
            self.feature_path = data_path
            self.data = load_json(data_path)
            self.data_len = len(self.data)
            self.config = config.data
            self.seed = seed
            self.sd_prob = args.sd_prob
            self.mode = mode
            self.batch_size = (
                args.batch_size
                * distributed_utils.get_data_parallel_world_size()
                * args.update_freq[0]
            )
        else:
            if self.path == './example_data/':
                sample_weight = load_json(
                    os.path.join(self.path, mode + "_sample_weight.json")
                )
                self.multi_label = load_json(
                    os.path.join(self.path, mode + "_multi_label.json")
                )
            else:
                json_path = './dataset/json/' + json_prefix + '/'
                sample_weight = load_json(
                    os.path.join(json_path, mode + "_sample_weight.json")
                )
                self.multi_label = load_json(
                    os.path.join(json_path, mode + "_multi_label.json")
                )
            self.inverse_multi_label = self._inverse_map(self.multi_label)

            inverse_multi_label_len = len(self.inverse_multi_label)
            if data_path != './example_data/':
                if mode == 'train':
                    if inverse_multi_label_len == 613682:
                        print('pass train dataset check 613682')
                    elif inverse_multi_label_len == 16638:
                        print('pass train dataset check 16638')
                        assert disable_sd
                    elif inverse_multi_label_len == 18024:
                        print('pass train dataset check 18024')
                        assert disable_sd
                    elif inverse_multi_label_len == 616718:
                        print('pass train dataset check 616718')
                        # assert disable_sd
                    elif inverse_multi_label_len == 591565:
                        print('pass train dataset check 591565')
                        # assert disable_sd
                    elif inverse_multi_label_len == 94:
                        print('pass train dataset check 94')
                        # assert disable_sd
                    elif inverse_multi_label_len == 103:
                        print('pass train dataset check 103')
                        # assert disable_sd
                    else:
                        raise
                elif mode == 'eval':
                    assert inverse_multi_label_len == 1842 or inverse_multi_label_len == 1120 or inverse_multi_label_len == 18024 \
                    or inverse_multi_label_len == 16638 or inverse_multi_label_len == 616718 or inverse_multi_label_len == 591565 \
                        or inverse_multi_label_len == 94 or inverse_multi_label_len == 103
                    print('pass eval dataset check')
                else:
                    raise

            self.sample_weight = {}
            for chain in self.inverse_multi_label:
                entity = self.inverse_multi_label[chain]
                self.sample_weight[chain] = sample_weight[entity]
                assert (
                    sample_weight[entity] ==1
                ), f"weight wrong!"
            self.seq_sample_weight = sample_weight
            logger.info(
                "load {} chains (unique {} sequences)".format(
                    len(self.sample_weight), len(self.seq_sample_weight)
                )
            )
            self.feature_path = os.path.join(self.path, "pdb_features")
            self.label_path = os.path.join(self.path, "pdb_labels")
            if self.path == './example_data/':
                sd_sample_weight_path = os.path.join(
                    self.path, "sd_train_sample_weight.json"
                )
            else:
                sd_sample_weight_path = os.path.join(
                    json_path, "sd_train_sample_weight.json"
                )
            if mode == "train" and os.path.isfile(sd_sample_weight_path) and not disable_sd:
                self.sd_sample_weight = load_json(sd_sample_weight_path)
                logger.info(
                    "load {} self-distillation samples.".format(len(self.sd_sample_weight))
                )
                self.sd_feature_path = os.path.join(self.path, "sd_features")
                self.sd_label_path = os.path.join(self.path, "sd_labels")
            else:
                self.sd_sample_weight = None
            self.batch_size = (
                args.batch_size
                * distributed_utils.get_data_parallel_world_size()
                * args.update_freq[0]
            )
            self.data_len = (
                max_step * self.batch_size
                if max_step is not None
                else len(self.sample_weight)
            )
            self.mode = mode
            self.num_seq, self.seq_keys, self.seq_sample_prob = self.cal_sample_weight(
                self.seq_sample_weight
            )
            self.num_chain, self.chain_keys, self.sample_prob = self.cal_sample_weight(
                self.sample_weight
            )
            if self.sd_sample_weight is not None:
                (
                    self.sd_num_chain,
                    self.sd_chain_keys,
                    self.sd_sample_prob,
                ) = self.cal_sample_weight(self.sd_sample_weight)
            self.config = config.data
            self.seed = seed
            self.sd_prob = args.sd_prob

    def cal_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    def sample_chain(self, idx, sample_by_seq=False):
        is_distillation = False
        if self.mode == "train":
            with data_utils.numpy_seed(self.seed, idx, key="data_sample"):
                is_distillation = (
                    (np.random.rand(1)[0] < self.sd_prob)
                    if self.sd_sample_weight is not None
                    else False
                )
                if is_distillation:
                    prot_idx = np.random.choice(
                        self.sd_num_chain, p=self.sd_sample_prob
                    )
                    label_name = self.sd_chain_keys[prot_idx]
                    seq_name = label_name
                else:
                    if not sample_by_seq:
                        prot_idx = np.random.choice(self.num_chain, p=self.sample_prob)
                        label_name = self.chain_keys[prot_idx]
                        seq_name = self.inverse_multi_label[label_name]
                    else:
                        seq_idx = np.random.choice(self.num_seq, p=self.seq_sample_prob)
                        seq_name = self.seq_keys[seq_idx]
                        label_name = np.random.choice(self.multi_label[seq_name])
        else:
            label_name = self.chain_keys[idx]
            seq_name = self.inverse_multi_label[label_name]
        return seq_name, label_name, is_distillation

    def __getitem__(self, idx):
        if 'ts50' in self.feature_path:
            sequence_id = self.data[idx]['seq']
            label_id = self.data[idx]['seq']
            feature_dir = self.feature_path
            label_dir = (self.feature_path,idx)
            is_distillation = False
        else:   
            sequence_id, label_id, is_distillation = self.sample_chain(
                idx, sample_by_seq=True
            )
            feature_dir, label_dir = (
                (self.feature_path, self.label_path)
                if not is_distillation
                else (self.sd_feature_path, self.sd_label_path)
            )
        features, _ = load_and_process(
            self.config,
            self.mode,
            self.seed,
            batch_idx=(idx // self.batch_size),
            data_idx=idx,
            is_distillation=is_distillation,
            sequence_ids=[sequence_id],
            monomer_feature_dir=feature_dir,
            uniprot_msa_dir=None,
            label_ids=[label_id],
            label_dir=label_dir,
            symmetry_operations=None,
            is_monomer=True,
        )
        # seq_name = sequence_id
        # # write seq_name to a tmp file
        # tmp_file = os.path.join( "tmp_seq_name.txt")
        # with open(tmp_file, "w") as f:
        #     f.write(seq_name)
        # # convert seq_name to ascii then convert to tensors
        # seq_name = torch.tensor(
        #     [ord(c) for c in seq_name], dtype=torch.long, device=features['aatype'].device # type: ignore
        # ) 
        # features["seq_name"] = seq_name.float()
        # # convert back
        # # print(seq_name)
        # seq_name = "".join([chr(c) for c in seq_name.tolist()])
        # assert seq_name == sequence_id
        # print("seq_name", seq_name)
        # print('aatype', features['aatype'])
        if self.mode == 'train':
            if self.config.train.mask_node > 0:
                seq_mask = torch.rand_like(features['seq_mask']) >= self.config.train.mask_node
                features['seq_mask'] = features['seq_mask'] * (seq_mask)
        return features

    def __len__(self):
        return self.data_len

    @staticmethod
    def collater(samples):
        # first dim is recyling. bsz is at the 2nd dim
        return data_utils.collate_dict(samples, dim=1)

    @staticmethod
    def _inverse_map(mapping: Dict[str, List[str]]):
        inverse_mapping = {}
        for ent, refs in mapping.items():
            for ref in refs:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    ent_2 = inverse_mapping[ref]
                    assert (
                        ent == ent_2
                    ), f"multiple entities ({ent_2}, {ent}) exist for reference {ref}."
                inverse_mapping[ref] = ent
        return inverse_mapping



