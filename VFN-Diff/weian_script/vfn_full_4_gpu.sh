export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=6000
export NCCL_P2P_DISABLE=1
python -m torch.distributed.run \
    --nnodes 1 \
    --nproc_per_node=4 \
    experiments/train_se3_diffusion.py \
    --config-name=vfn_full_4_4090