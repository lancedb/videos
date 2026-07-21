---
# Video 1 — run with:  npx slidev vlm-materialized-features/video-1/slides/slides.md --open
theme: seriph
# Shared LanceDB brand (palette, footer, layouts, components), linked in the
# root package.json as file:../slidev/addon-lancedb.
addons:
  - slidev-addon-lancedb
title: Materialize model features for faster VLM fine-tuning
info: |
  Breaking down "Faster VLM fine-tuning with materialized
  model features in LanceDB" into a 5-minute video.
# The template is designed at 1280×720, so match that canvas for 1:1 sizing.
canvasWidth: 1280
aspectRatio: 16/9
fonts:
  sans: Geist
  mono: Geist Mono
  weights: '300,400,500,600,700,800'
transition: slide-left
# The headmatter is also slide 1's frontmatter, so slide 1 is the cover.
layout: cover
---

<Eyebrow>Vision model fine-tuning</Eyebrow>

# Fine-tune your models faster <span class="gradient-text">with materialized model features</span>

<p class="subtitle">
How Lance's zero-copy data evolution and fast retrieval let you fine-tune a VLM faster in LanceDB.
</p>

::hero::

![Vector computer illustration](./assets/hero.png)

<!--
0:00–0:15 · ~15s · SAY:

If you fine-tune vision language models, you're probably wasting a lot of GPU
time recomputing something that never changes. In the next five minutes I'll
show you how to fix that with your data format, not your model.

[advance]
-->

---

# The task: read the image, then <span class="gradient-text">fine-tune on it</span>

<div class="task-cols">
<div>
  <img src="/textvqa-diff.png" alt="TextVQA examples on a TWA sugar packet — questions and the answers read from the image" class="task-img" />
  <div class="task-cap">Each example: an image + a question + the answer + the OCR text read off it. Answers come straight from the packet's print.<br>Dataset: <a href="https://textvqa.org/" target="_blank">textvqa.org ↗</a></div>
</div>
<div>
  <p class="task-lead">
  <strong>TextVQA</strong>: answer a question whose answer is text written <em>in</em> the image, so the model has to read the picture, not just recognize objects. It works in two stages: an <strong>image encoder</strong> turns the image into visual embeddings, then a <strong>language model</strong> reads those embeddings plus the question and writes the answer.
  </p>
  <ul class="bullet-list task-bul">
    <li>A general base model is broad but misses domain specifics, like reading the small <strong>"Domino"</strong> print to name the brand.</li>
    <li><strong>Supervised fine-tuning (SFT)</strong> shows it many (image, question, answer) examples, so it learns to answer <em>our</em> questions better.</li>
  </ul>
</div>
</div>

<style>
.task-cols {
  display: grid;
  grid-template-columns: 1fr 1.08fr;
  gap: 40px;
  align-items: center;
  margin-top: 28px;
}
.task-img { width: 100%; border-radius: 12px; border: 1px solid var(--border); }
.task-cap { margin-top: 9px; font-size: 12px; color: var(--fg-dim); line-height: 1.5; }
.task-cap a { color: var(--accent-soft); text-decoration: none; border-bottom: 1px solid var(--accent); }
.task-lead { font-size: 16px; color: var(--fg-muted); line-height: 1.55; margin-top: 0; }
.task-lead strong { color: var(--fg); }
.task-bul { margin-top: 16px; }
/* The theme's ul style outranks .bullet-list; kill native markers here. */
.task-bul, .task-bul li { list-style: none !important; }
.task-bul li::marker { content: none; }
</style>

<!--
0:15–0:55 · ~40s · SAY:

Our task is TextVQA: answer a question about an image where the answer is
literally written in the picture. What brand is the sugar? You have to read
the packet to say "Domino".

A vision language model answers in two stages. An image encoder turns the
picture into visual embeddings, then a language model reads those embeddings
plus your question and writes the answer.

Base models are broad, but they miss small domain details, like that tiny
print on the label. So we fine-tune: we show the model lots of image,
question, answer examples until it gets good at our kind of questions.

[advance]
-->

---
class: flex flex-col justify-center
---

# The hidden waste in VLM fine-tuning

<p class="lede">Every time the training loop reads a row, the frozen vision tower re-encodes the same image into the same expensive features. Same numbers, every epoch.</p>

<div class="pc-cols">
  <div class="pc">
    <div class="pc-h">Without Lance</div>
    <div class="pc-b">Re-encode every epoch (wasted GPU), or precompute into <strong>sidecar files</strong> (.npy / HDF5) you keep aligned by hand. And adding a column to Parquet or Iceberg rewrites the whole table.<br><span class="pc-n">a second artifact to manage, or a full rewrite</span></div>
  </div>
  <div class="pc pc-on">
    <div class="pc-h" style="color: var(--accent-soft);">With Lance</div>
    <div class="pc-b">Precompute once and add them as a <strong>column on the same table</strong>: a cheap append, no rewrite. The loader reads them straight from the table.<br><span class="pc-n">one table, nothing to keep in sync</span></div>
  </div>
</div>

<div class="callout" style="margin-top: 16px;">
  Precomputing is the speedup; <strong>Lance makes it painless</strong>: a cheap column add instead of a table rewrite, with no sidecar files. <strong>~2× faster steps, 1.3 GB less GPU memory.</strong>
</div>

<style>
.pc-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-top: 24px; }
.pc { border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; background: var(--bg-elev); }
.pc.pc-on { border-color: var(--accent); }
.pc-h { font-family: 'Geist Mono', ui-monospace, monospace; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--fg-dim); margin-bottom: 8px; }
.pc-b { font-size: 15px; color: var(--fg-muted); line-height: 1.5; }
.pc-n { display: inline-block; margin-top: 6px; font-size: 12.5px; color: var(--fg-dim); }
</style>

<!--
0:55–1:35 · ~40s · SAY:

Here's the hidden waste. During fine-tuning, the image encoder is frozen. Same
image in, same embeddings out, every single epoch. Yet the standard training
loop re-encodes every image on every pass. That's pure wasted GPU.

The fix is to precompute those embeddings once. But where do you put them?
Sidecar files you keep aligned by hand? And adding a column to a Parquet table
means rewriting the whole table.

With Lance, you add them as a column on the same table, and the dataloader
reads them straight off disk. That's roughly two times faster training steps,
and over a gigabyte of GPU memory back.

[advance]
-->

---

# Zero-copy data evolution

<p class="lede">Backfill a new column, without rewriting the table.</p>

<div class="zc-wrap">

<div class="zc-diagram">
  <div class="zc-topline">
    <div class="zc-main">
      <div class="zc-half zc-left">
        <div class="zc-title">Large dataset</div>
        <div class="zc-sub">text, images, etc.</div>
      </div>
      <div class="zc-half">
        <div class="zc-title">Current features</div>
      </div>
    </div>
    <div class="zc-newcols">
      <div class="zc-collabel">New features ↓</div>
      <div class="zc-col"></div>
      <div class="zc-col"></div>
    </div>
  </div>
  <div class="zc-newrows">
    <div class="zc-row"></div>
    <div class="zc-row"></div>
    <div class="zc-rowlabel">↑ New observations</div>
  </div>
</div>

<div>

<ul class="bullet-list" style="margin-top: 8px;">
  <li><strong>Only the new bytes get written</strong>, so adding a column never rewrites the table</li>
  <li>Parquet-based table formats rewrite files to change a schema (row groups). Lance just <strong>adds the new column to a new data file</strong></li>
  <li>Features live <strong>next to the existing data</strong>, so there are no sidecar files to keep in sync</li>
</ul>

<div class="callout" style="margin-top: 28px; font-size: 15px;">
With Lance, creating new features is cheap enough that materializing results from an expensive computation becomes
<strong>second nature</strong>.
</div>

</div>

</div>

<style>
.zc-wrap {
  display: grid;
  grid-template-columns: 1.05fr 1fr;
  gap: 56px;
  align-items: start;
  margin-top: 56px;
}
/* The theme's ul style outranks .bullet-list; kill native markers here. */
.zc-wrap ul.bullet-list,
.zc-wrap ul.bullet-list li { list-style: none !important; }
.zc-wrap ul.bullet-list li::marker { content: none; }
.zc-topline { display: flex; gap: 18px; align-items: stretch; }
.zc-main {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  border: 1.5px solid var(--accent-soft);
  border-radius: 18px;
  min-height: 280px;
  flex: 1;
  background: rgba(255, 115, 74, 0.05);
}
.zc-half {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 16px;
  text-align: center;
}
.zc-half.zc-left { border-right: 1.5px dashed var(--accent-soft); }
.zc-title { font-size: 17px; font-weight: 600; color: var(--accent-soft); }
.zc-sub { font-size: 12.5px; color: var(--fg-muted); }
.zc-newcols { display: flex; gap: 10px; align-items: stretch; position: relative; }
.zc-col {
  width: 26px;
  border: 1.5px solid var(--accent-soft);
  border-radius: 10px;
  background: rgba(255, 115, 74, 0.10);
}
.zc-collabel {
  position: absolute;
  right: 0;
  bottom: calc(100% + 10px);
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 12px;
  color: var(--accent);
  white-space: nowrap;
}
.zc-newrows { margin-top: 14px; position: relative; padding-right: 80px; }
.zc-row {
  height: 22px;
  border: 1.5px solid var(--accent-soft);
  border-radius: 10px;
  background: rgba(255, 115, 74, 0.10);
  margin-bottom: 10px;
  /* match the main box width (exclude the new-feature columns to the right) */
  width: calc(100% - 20px);
}
.zc-rowlabel {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 12px;
  color: var(--accent);
}
</style>

<!--
1:35–2:00 · ~25s · SAY:

So why is adding that column cheap? Lance tables grow in two directions. New
feature columns attach alongside the existing data, and new rows append below.
Neither touches the bytes you already wrote; only the new column's data gets
written. No table rewrite, no sidecar files.

That's what makes materializing an expensive computation second nature.
Alright, let's do exactly that, on a real table.

[cut to the notebook — 3:00 for the code walkthrough]
-->
