# Video 1 — Compute it once, store it as a column

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-1/01_finetune_vlm_lance.py)

Source code for video 1 of the *materialized model features* series, which breaks
down the LanceDB blog post
[Faster VLM fine-tuning with materialized model features](https://www.lancedb.com/blog/faster-vlm-fine-tuning-with-materialized-model-features-in-lancedb)
into a short, educational video.

The story: a vision-language model's frozen vision tower re-encodes the same images
on every training read. Compute those features once, store them as a column in a
Lance table (a cheap, zero-copy append), and the training loop reads them straight
off disk. The result is roughly 2x faster steps and about 1.3 GB less GPU memory.

## What's here

```
video-1/
├── 01_finetune_vlm_lance.py   # marimo notebook: the full fine-tune loop off one Lance table
├── vlm/                        # helper package, frozen from lancedb/tmls-2026-demo
│   ├── schema.py               # the schema-enforced table definition
│   ├── dataloader.py           # LanceDB Permutation-API dataloader
│   ├── train_qwen25vl_lora.py  # QLoRA training building blocks
│   └── eval.py                 # generation, scoring, thumbnail helpers
└── slides/
    └── slides.md               # slidev deck (the video's opening frames)
```

The `vlm/` package is a self-contained copy, frozen to match what appears on screen.
It is not a dependency on the demo repo.

## The notebook

`01_finetune_vlm_lance.py` runs the whole loop end to end, off a single Lance table:

1. **Download** a pre-baked, curated `text_dense` slice of TextVQA (the vision
   features are already computed and stored as the `vision_tower_hiddens` column).
2. **Explore** it with LanceDB: distributions plus a cross-modal text-to-image
   vector search over the shipped CLIP embeddings.
3. **Benchmark** read throughput, sequential vs shuffled, LanceDB vs Parquet.
4. **QLoRA fine-tune** `Qwen2.5-VL-3B-Instruct`, reading the cached column off disk
   with the vision tower never loaded.
5. **Score** base vs tuned on the held-out val split (the 0.799 to 0.820 lift).

Dependencies are declared inline in a PEP 723 header at the top of the file, so the
notebook carries its own environment. The two GPU-heavy sections (fine-tune and
eval) sit behind run buttons, so opening the notebook never starts a training run.

## How to run

### Locally (macOS / no GPU): the data-layer sections

Run from this directory so the `vlm/` package resolves:

```bash
cd vlm-materialized-features/video-1
uvx marimo edit 01_finetune_vlm_lance.py --sandbox
```

`--sandbox` builds an isolated environment from the PEP 723 header, so nothing
touches your root env. The download, explore, and throughput cells run fine on CPU.
The fine-tune and eval sections stay parked behind their run buttons (they need a
GPU). The first sandbox install pulls the full stack (torch, transformers,
bitsandbytes) and takes a few minutes, cached after that.

To open it read-only as an app instead of the editor:

```bash
uvx marimo run 01_finetune_vlm_lance.py --sandbox
```

### On molab (full run, with GPU)

The fine-tune and eval sections need a CUDA GPU (about 5 GB VRAM is plenty). molab
provides one:

1. Open the notebook on molab (or click the badge at the top of this README):
   `https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-1/01_finetune_vlm_lance.py`
2. Toggle the GPU on via the notebook specs button in the app header.
3. Run all, then click **Run fine-tune** and **Run before/after eval**.

## The slides

The slidev deck in `slides/` is styled with the shared LanceDB brand addon. Preview
it from the repo root:

```bash
npm install                     # first time only, from the repo root
npx slidev vlm-materialized-features/video-1/slides/slides.md --open
```

## Notes

- **GPU torch:** the header's plain `torch` installs the CPU build from PyPI, which
  is fine for the data-layer cells. On molab with a GPU the environment supplies the
  CUDA build. On your own GPU box, install a CUDA torch wheel explicitly.
- **lancedb version:** `dataloader.py` imports `lancedb.permutation.Permutation`.
  Confirm that path still exists in the lancedb version that resolves, since the
  header pins to recent floors. It surfaces when the fine-tune loader is built.
- **Data:** the curated subset is the public Hugging Face dataset
  [`lance-format/textvqa-lance-colab`](https://huggingface.co/datasets/lance-format/textvqa-lance-colab)
  (600 train / 400 val rows). No token needed.
