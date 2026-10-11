#!/bin/bash
# Generate T2I-CompBench images from a trained CS-Aligner checkpoint.
# Usage: bash scripts/eval_t2i_compbench.sh save/<exp>/<ckpt>.ckpt [path/to/T2I-CompBench]
cd "$(dirname "$0")/.."
CKPT=${1:?usage: $0 CKPT [COMPBENCH_DIR]}
BENCH=${2:-../T2I-CompBench/examples/dataset}

for split in color_val shape_val texture_val spatial_val non_spatial_val; do
    python sample_t2i.py --ckpt "$CKPT" --prompts "$BENCH/$split.txt"
done
