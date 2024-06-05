export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p ./output_dir/nofeat/VFN_baseline_wo_node_feat_12l/
python train.py ./processed/  \
       --user-dir user \
       --num-workers 16 \
       --ddp-backend=no_c10d \
       --task de --loss af2 --arch de \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --wd 0.1 --clip-norm 1.0 --allreduce-fp32-grad  \
       --lr-scheduler onecycle --lr 1e-3 --warmup-updates 1000 --decay-ratio 0.1 --decay-steps 100000 \
       --batch-size 8 \
       --update-freq 1 --seed 3  \
       --max-update 100000 --log-interval 10 --save-interval-updates 5000 --validate-interval-updates 10000 --keep-interval-updates 5 \
       --log-format simple \
       --tensorboard-logdir ./output_dir/nofeat/VFN_baseline_wo_node_feat_12l/tsb \
       --save-dir ./output_dir/nofeat/VFN_baseline_wo_node_feat_12l/ \
       --tmp-save-dir ./output_dir/nofeat/VFN_baseline_wo_node_feat_12l/tmp/ \
       --required-batch-size-multiple 1 \
       --ema-decay 0.999 \
       --model-name VFN_baseline_wo_node_feat_12l \
       --batch-size-valid 1 \
       --json-prefix CATH4.2 \
       --disable-sd \
       --data-buffer-size 32 >> ./output_dir/nofeat/VFN_baseline_wo_node_feat_12l/output.txt 2>&1