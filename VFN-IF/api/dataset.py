import os
import json
import numpy as np
from typing import *
import ml_collections as mlc
from contextlib import contextmanager

import copy
from .data import utils
from .data import residue_constants as rc
from .data.data_ops import NumpyDict, TorchDict, inverse_folding_featurizer
from .data.process import process_features, process_labels
import torch
from torch.utils.data import Dataset, DataLoader
from .data.process_multimer import (
    pair_and_merge,
    add_assembly_features,
    convert_monomer_features,
    post_process,
    merge_msas,
)

Rotation = Iterable[Iterable]
Translation = Iterable
Operation = Union[str, Tuple[Rotation, Translation]]
NumpyExample = Tuple[NumpyDict, Optional[List[NumpyDict]]]
TorchExample = Tuple[TorchDict, Optional[List[TorchDict]]]

def load_json(filename):
    return json.load(open(filename, 'r'))

def str_hash(text:str):
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

@contextmanager
def numpy_seed(seed, *addl_seeds, key=None):
    """
    Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward
    """
    if seed is None:
        yield
        return
    def check_seed(s):
        assert type(s) == int or type(s) == np.int32 or type(s) == np.int64
    check_seed(seed)
    if len(addl_seeds) > 0:
        for s in addl_seeds:
            check_seed(s)
        seed = int(hash((seed, *addl_seeds)) % 1e8)
        if key is not None:
            seed = int(hash((seed, str_hash(key))) % 1e8)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

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

def process_label(all_atom_positions: np.ndarray, operation: Operation) -> np.ndarray:
    if operation == "I":
        return all_atom_positions
    rot, trans = operation
    rot = np.array(rot).reshape(3, 3)
    trans = np.array(trans).reshape(3)
    return all_atom_positions @ rot.T + trans

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

def load_fast(
    sequence_ids: List[str],
    monomer_feature_dir: str,
    uniprot_msa_dir: Optional[str] = None,
    label_ids: Optional[List[str]] = None,
    label_dir: Optional[str] = None,
    symmetry_operations: Optional[List[Operation]] = None,
    is_monomer: bool = False,
) -> NumpyExample:
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

    all_chain_features ={}
    all_chain_features['aatype'] = all_chain_labels[0]['aatype_index']
    all_chain_features['residue_index'] = np.arange(len(all_chain_features['aatype']))
    all_chain_features['seq_length'] = np.array(len(all_chain_features['aatype']))
    [f.update(l) for f, l in zip([all_chain_features], all_chain_labels)]

    return all_chain_features, all_chain_labels

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
        with numpy_seed(seed, batch_idx, key="recycling"):
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

    with numpy_seed(seed, data_idx, key="protein_feature"):
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
    
    features, labels = process(
        config, mode, features, labels, seed, batch_idx, data_idx, is_distillation
    )
    return features, labels



def get_dataloader(
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    use_distributed_sampler: bool = False
):
    """
    The entry function of Unifold Dataset Load
    """
    dataset = UnifoldDataset()
    if use_distributed_sampler:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler
    )

class UnifoldDataset(Dataset):
    def __init__(
        self,
        args,
        config,
        data_path: str,
        seed: int = 0,
        mode="train",
        max_step=None,
        disable_sd=False,
        json_prefix="",
    ):
        self.path = data_path
        
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
            self.batch_size = args.batch_size
        else:
            if self.path == "./example_data":
                sample_weight = load_json (
                    os.path.join(self.path, mode + "_sample_weight.json")
                )
                self.muti_label = load_json(
                    os.path.join(self.path, mode + "_multi_label.json")
                )
            else:
                json_path = "./dataset/json" + json_prefix + '/'
                sample_weight = load_json(
                    os.path.join(json_path, mode + "_sample_weight.json")
                )
                self.multi_label = load_json(
                    os.path.join(json_path, mode + "_multi_label.json")
                )
            self.inverse_multi_label = self._inverse_map(self.muti_label)

            inverse_multi_label_len = len(self.inverse_multi_label)
            # TODO: what's the meaning?
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
            else:
                pass # TODO:need to fix whether need example data processing
        
        self.sample_weight = {}
        for chain in self.inverse_multi_label:
            entity = self.inverse_multi_label[chain]
            self.sample_weight[chain] = sample_weight[entity]
            assert (
                    sample_weight[entity] ==1
            ), f"weight wrong!"
        self.seq_sample_weight = sample_weight

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
            self.sd_feature_path = os.path.join(self.path, "sd_features")
            self.sd_label_path = os.path.join(self.path, "sd_labels")
        else:
            self.sd_sample_weight = None
        self.batch_size = args.batch_size
        self.data_len = len(self.sample_weight)

        self.mode = mode
        self.num_seq, self.seq_keys, self.seq_sample_prob = self.calculate_sample_weight(
            self.seq_sample_weight
        )
        self.num_chain, self.chain_keys, self.sample_prob = self.calculate_sample_weight(
            self.sample_weight
        )
        if self.sd_sample_weight is not None:
            (
                self.sd_num_chain,
                self.sd_chain_keys,
                self.sd_sample_prob,
            ) = self.calculate_sample_weight(self.sd_sample_weight)
        self.config = config.data
        self.seed = seed
        self.sd_prob = args.sd_prob

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
        
        if self.mode == 'train':
            if self.config.train.mask_node > 0:
                seq_mask = torch.rand_like(features['seq_mask']) >= self.config.train.mask_node
                features['seq_mask'] = features['seq_mask'] * (seq_mask)
        return features

    def __len__(self):
        return self.data_len

    def calculate_sample_weight(self, sample_weight):
        prot_keys = list(sample_weight.keys())
        sum_weight = sum(sample_weight.values())
        sample_prob = [sample_weight[k] / sum_weight for k in prot_keys]
        num_prot = len(prot_keys)
        return num_prot, prot_keys, sample_prob

    @staticmethod
    def _inverse_map(mapping: Dict[str, List[str]]) -> Dict[str, str]:
        inverse_mapping = {}
        for entity, reference in mapping.items():
            for ref in reference:
                if ref in inverse_mapping:  # duplicated ent for this ref.
                    entity_2 = inverse_mapping[ref]
                    assert (
                        entity == entity_2
                    ), f"multiple entities ({entity_2}, {entity}) exist for reference {ref}."
                inverse_mapping[ref] = entity
        return inverse_mapping
    
    def sample_chain(self, idx, sample_by_seq=False):
        is_distillation = False
        if self.mode == "train":
            with numpy_seed(self.seed, idx, key="data_sample"):
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
