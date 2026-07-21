# videos

Code behind LanceDB demo videos. Each video project is a self-contained directory
with its runnable materials (marimo notebooks, helper code, and slides).

The notebooks are [marimo](https://marimo.io) notebooks: Jupyter-style, but stored
as plain Python files. Each project's "Open in molab" badge runs its notebook on
[molab](https://molab.marimo.io), marimo's hosted notebook service, similar to
Google Colab.

See [AGENTS.md](AGENTS.md) for how the repo is organized and how to work in it.

## Projects

### [vlm-materialized-features](vlm-materialized-features/)

Breaking down the blog post
[Faster VLM fine-tuning with materialized model features](https://www.lancedb.com/blog/faster-vlm-fine-tuning-with-materialized-model-features-in-lancedb)
into short, educational videos.

- **[video-1](vlm-materialized-features/video-1/)**: Compute it once, store it as a
  column. A marimo notebook that fine-tunes a vision-language model off a single
  Lance table, plus the slide deck for the video's opening frames.
- **[video-2](vlm-materialized-features/video-2/)**: Feature engineering for
  fine-tuning pipelines. A marimo notebook that builds three tiers of feature
  columns on one Lance table with UDF backfills, plus its slide deck.

Each project README has the run steps, both local and on molab.
