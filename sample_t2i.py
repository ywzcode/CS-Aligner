"""Generate images for a prompt file (e.g. T2I-CompBench) with a trained CS-Aligner prior.

    python sample_t2i.py --ckpt save/<exp>/cs-aligner-coco-periodic-epoch=0259.ckpt \
        --prompts ../T2I-CompBench/examples/dataset/color_val.txt

Images are written to <outdir>/<exp>_epoch<E>_<prompt-file>/samples/<prompt>_<k>.png
"""
import argparse
import os

import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

from cs_aligner.model import CSAlignerBackbone, load_prior_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="CS-Aligner Lightning checkpoint")
    p.add_argument("--prompts", required=True, type=str, help="text file, one prompt per line (tab-separated ok)")
    p.add_argument("--outdir", default="/home/wenzhey/path-to-scratch/T2I-bench-Results", type=str)
    p.add_argument("--n_iter", default=10, type=int, help="images per prompt")
    p.add_argument("--batch_size", default=5, type=int, help="prompts per batch")
    p.add_argument("--steps", default=50, type=int)
    p.add_argument("--guidance_scale", default=7.5, type=float)
    p.add_argument("--negative_prompt", default="", type=str)
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    exp = os.path.basename(os.path.dirname(args.ckpt))
    epoch = os.path.basename(args.ckpt).split("epoch=")[-1].split(".")[0]
    dataset = os.path.splitext(os.path.basename(args.prompts))[0]
    sample_dir = os.path.join(args.outdir, f"{exp}_epoch{epoch}_{dataset}", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_prior_checkpoint(CSAlignerBackbone(device), args.ckpt, strict=True, map_location=device)
    model.eval()

    with open(args.prompts) as f:
        prompts = [line.strip("\n").split("\t")[0] for line in f if line.strip()]
    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    for batch in tqdm(batches, desc="prompt batches"):
        neg = [args.negative_prompt] * len(batch) if args.negative_prompt else None
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_embs, neg_embs = model.pipe_prior(batch, negative_prompt=neg).to_tuple()
            images = model.decode(img_embs, neg_embs, num_inference_steps=args.steps,
                                  guidance_scale=args.guidance_scale, num_images_per_prompt=args.n_iter)
        for i, prompt in enumerate(batch):
            for k in range(args.n_iter):
                images[i * args.n_iter + k].save(os.path.join(sample_dir, f"{prompt.strip('.').strip()}_{k:06d}.png"))


if __name__ == "__main__":
    main()
