export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p ./output_dir/x2/VFN_baseline_16vec_v2/
python train.py ./processed/  \
       --user-dir user \
       --num-workers 16 \
       --ddp-backend=no_c10d \
       --task de --loss af2 --arch de \
       --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --wd 0.1 --clip-norm 1.0 --allreduce-fp32-grad  \
       --lr-scheduler onecycle --lr 1e-3 --warmup-updates 1000 --decay-ratio 0.1 --decay-steps 100000 \
       --batch-size 8 \
       --update-freq 1 --seed 2  \
       --max-update 100000 --log-interval 10 --save-interval-updates 5000 --validate-interval-updates 10000 --keep-interval-updates 5 \
       --log-format simple \
       --tensorboard-logdir ./output_dir/x2/VFN_baseline_16vec_v2/tsb \
       --save-dir ./output_dir/x2/VFN_baseline_16vec_v2/ \
       --tmp-save-dir ./output_dir/x2/VFN_baseline_16vec_v2/tmp/ \
       --required-batch-size-multiple 1 \
       --ema-decay 0.999 \
       --model-name VFN_baseline_16vec_v2 \
       --batch-size-valid 1 \
       --json-prefix CATH4.2 \
       --disable-sd \
       --data-buffer-size 32 >> ./output_dir/x2/VFN_baseline_16vec_v2/output.txt 2>&1
