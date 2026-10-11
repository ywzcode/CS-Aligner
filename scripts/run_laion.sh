#!/bin/bash
# CS-Aligner on LAION (high-resolution WebDataset shards), validated on MSCOCO val2014.
# Run from anywhere:  bash scripts/run_laion.sh
cd "$(dirname "$0")/.."

# original run: save/new-laion-1dev-1024bs0.01
python train_laion.py \
    --devices 1 --batch_size 1024 --lr 0.00005 \
    --aux_weight 0.01 \
    --expname laion-1dev-1024bs-aux0.01

# 2-GPU variant (original run name: exp-laion-2dev-2048bs0.001); batch_size is per GPU
# torchrun --nproc_per_node=2 train_laion.py \
#     --devices 2 --batch_size 1024 --lr 0.00005 \
#     --aux_weight 0.001 \
#     --expname laion-2dev-2048bs-aux0.001
