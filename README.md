# CS-Aligner

Cauchy-Schwarz (CS) divergence based alignment of text and image embeddings (ICLR).
The text-to-image prior of ECLIPSE / Kandinsky 2.2 is trained so that the distribution of
predicted image embeddings matches the CLIP (ViT-bigG) image-embedding distribution:

```
loss = D_CS(img, text) + aux_weight * InfoNCE(img, text)
```

Only the prior is trained; the CLIP encoders and the Kandinsky 2.2 decoder are frozen.

## Layout

```
cs_aligner/
  losses.py              CS divergence, InfoNCE
  model.py               CSAlignerBackbone (frozen CLIP + Kandinsky 2.2, trainable ECLIPSE prior)
  lightning_module.py    CSAligner LightningModule (loss, validation, FID, t-SNE) + shared CLI args
  prior/                 vendored ECLIPSE PriorTransformer and Kandinsky prior pipeline
  metrics/               FID (pytorch-fid)
train_coco.py            MSCOCO train2014 -> val2014
train_laion.py           LAION high-res WebDataset shards -> MSCOCO val2014
sample_t2i.py            generate images for a prompt file (T2I-CompBench)
scripts/
  run_coco.sh            COCO experiments
  run_laion.sh           LAION experiments
  eval_t2i_compbench.sh  T2I-CompBench generation for a checkpoint
```

## Usage

```bash
bash scripts/run_coco.sh
bash scripts/run_laion.sh
bash scripts/eval_t2i_compbench.sh save/<exp>/<ckpt>.ckpt
```

Data paths (`--coco_root`, `--laion_shards`, `--laion_cache`, `--fid_stat`) default to the
Snellius locations used in the original runs; override them on the command line.

Environment: the `dalle2` conda env (torch 2.3.1, pytorch-lightning 2.1.0, diffusers 0.20.2,
transformers 4.46.0, webdataset).

