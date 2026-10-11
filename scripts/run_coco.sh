#!/bin/bash
# CS-Aligner on MSCOCO. Run from anywhere:  bash scripts/run_coco.sh
cd "$(dirname "$0")/.."

# (1) CS-Aligner with negative prompt
torchrun --nproc_per_node=2 train_coco.py \
    --devices 2 --batch_size 1024 --lr 0.0001 --aux_weight 0.01 \
    --negative_prompt "low quality, bad quality" \
    --expname coco-2dev-1024bs-aux0.01-negprompt

# (2) CS-Aligner without negative prompt
torchrun --nproc_per_node=2 train_coco.py \
    --devices 2 --batch_size 1024 --lr 0.0001 --aux_weight 0.01 \
    --expname coco-2dev-1024bs-aux0.01

# Resume a run:  add  --resume save/<expname>/cs-aligner-coco-periodic-epoch=XXXX.ckpt
