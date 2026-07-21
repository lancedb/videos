# Video 2 — Feature engineering for fine-tuning pipelines

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-2/02_feature_engineering.py)

Source code for video 2 of the *materialized model features* series, which breaks
down the LanceDB blog post
[Faster VLM fine-tuning with materialized model features](https://www.lancedb.com/blog/faster-vlm-fine-tuning-with-materialized-model-features-in-lancedb)
into short, educational videos.

The story: a fine-tuning pipeline runs on derived data, not raw inputs. You define
each feature as a plain Python function (a UDF) and LanceDB's feature engineering
materializes it as a new column on the table that already holds your raw data. It
handles batching, checkpointing, and distributing the compute, so the same function
runs on a laptop or a cluster.

## What's here

```
video-2/
├── 02_feature_engineering.py   # marimo notebook: three tiers of features on one table
└── slides/
    └── slides.md               # slidev deck (the video's opening frames)
```

The notebook is a single self-contained file. Each feature UDF is defined inline and
stays visible, since the UDFs are the point of this video. Only the download helper
is tucked away. Opening the notebook anywhere, including "Open in molab", brings
everything along.

## The notebook

`02_feature_engineering.py` builds three tiers of features on one TextVQA Lance
table, cheapest to most expensive, using one abstraction:

1. **Tier 1, `question_type`** — a plain `@udf` function over the question text.
   Backfilled live on CPU; the result's distribution is charted.
2. **Tier 2, `dhash`** — a 64-bit perceptual hash for near-duplicate images. Same
   decorator, same backfill call, only the body changed. Live on CPU.
3. **Tier 3, `vision_tower_hiddens`** — a stateful class UDF that lazy-loads a frozen
   vision model and emits a fixed-size vector per image. Gated behind a run button
   (needs a GPU). It is already present on the shipped table, so the cell walks the
   UDF and the exact backfill call that produced it.

A hidden appendix cell shows the same Tier-3 work written by hand (a single-process
runner) for contrast: the plumbing you would otherwise maintain yourself, on one
machine.

Dependencies are declared inline in a PEP 723 header, so the notebook carries its own
environment.

## How to run

### Locally (macOS / no GPU): Tiers 1 and 2

```bash
cd vlm-materialized-features/video-2
uvx marimo edit 02_feature_engineering.py --sandbox
```

`--sandbox` builds an isolated environment from the PEP 723 header. The download and
the Tier 1 / Tier 2 backfills run on CPU. Tier 3 stays parked behind its run button
(it needs a GPU). The first sandbox install pulls a large stack (torch, transformers)
and takes a few minutes, cached after that.

### On molab (all three tiers, with GPU)

The Tier 3 backfill needs a CUDA GPU (about 5 GB VRAM). molab provides one:

1. Open the notebook on molab (or click the badge at the top of this README):
   `https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-2/02_feature_engineering.py`
2. Toggle the GPU on via the notebook specs button in the app header.
3. Run all, then click **Run Tier 3 backfill**.

## The slides

The slidev deck in `slides/` is styled with the shared LanceDB brand addon. Preview
it from the repo root:

```bash
npm install                     # first time only, from the repo root
npx slidev vlm-materialized-features/video-2/slides/slides.md --open
```

## Notes

- **GPU torch:** the header's plain `torch` installs the CPU build from PyPI, which
  is fine for Tiers 1 and 2. On molab with a GPU the environment supplies the CUDA
  build. On your own GPU box, install a CUDA torch wheel explicitly.
- **First run:** the backfill path (a local compute context for Tiers 1 and 2, and
  Tier 3 on GPU) should be exercised on molab before recording, the same way video 1
  was verified.
- **Data:** the subset is the public Hugging Face dataset
  [`lance-format/textvqa-lance-colab`](https://huggingface.co/datasets/lance-format/textvqa-lance-colab)
  (600 train rows). No token needed.
