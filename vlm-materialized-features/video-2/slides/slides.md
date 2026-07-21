---
# Video 2 — run with:  npx slidev vlm-materialized-features/video-2/slides/slides.md --open
theme: seriph
# Shared LanceDB brand (palette, footer, layouts, components), linked in the
# root package.json as file:../slidev/addon-lancedb.
addons:
  - slidev-addon-lancedb
title: Feature engineering for fine-tuning pipelines
info: |
  Video 2 of the materialized model features series: deriving the feature
  columns a fine-tuning pipeline needs, directly on a LanceDB table.
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

<Eyebrow>Training data pipelines</Eyebrow>

# Feature engineering for <span class="gradient-text">fine-tuning pipelines</span>

<p class="subtitle">
Turn raw data into training-ready feature columns, directly on your LanceDB
table, without paying an infrastructure tax to compute them at scale.
</p>

::hero::

![Vector computer illustration](./assets/hero.png)

<!--
0:00–0:20 · ~20s · SAY:

Fine-tuning doesn't run on raw images and text. It runs on derived data:
embeddings, tokenized prompts, quality signals. Computing all of that over a
real corpus is where a lot of research time quietly disappears. This video is
about making that part cheap.

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
  width: calc(100% - 20px);
}
.zc-rowlabel {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 12px;
  color: var(--accent);
}
</style>

<!--
0:20–1:05 · ~45s · SAY:

Here's the idea the whole video rests on. In a LanceDB table, every feature you
derive is just a new column on the table that already holds your raw data.

Adding that column writes only the new column's bytes. It doesn't rewrite the
table, and it doesn't scatter your features into separate files you have to keep
lined up with the source rows by hand.

So the natural home for a feature isn't some separate feature store. It's the
source table itself. You compute a column, you attach it, and later you read the
raw inputs and the derived features back together in one shot.

[advance]
-->

---
class: flex flex-col justify-center
---

# Three tiers of features, <span class="gradient-text">one abstraction</span>

<p class="lede">The features a fine-tuning pipeline needs span a wide cost range. The way you define them shouldn't.</p>

<div class="ft-wrap">

<div class="ft-tiers">
  <div class="ft-tier">
    <div class="ft-badge">Tier 1</div>
    <div class="ft-body"><strong>Text signals</strong> · question type, lengths, token counts</div>
    <div class="ft-cost">CPU · seconds</div>
  </div>
  <div class="ft-tier">
    <div class="ft-badge">Tier 2</div>
    <div class="ft-body"><strong>Image-decode features</strong> · a perceptual hash to catch near-duplicates</div>
    <div class="ft-cost">CPU · minutes</div>
  </div>
  <div class="ft-tier ft-hot">
    <div class="ft-badge">Tier 3</div>
    <div class="ft-body"><strong>Model features</strong> · frozen vision-tower embeddings, pre-tokenized prompts</div>
    <div class="ft-cost">GPU · the expensive one</div>
  </div>
</div>

<div class="ft-abs">
  <div class="ft-abs-row">
    <div class="ft-abs-k">You write</div>
    <div class="ft-abs-v">a plain Python function that turns one row into a feature (a <strong>UDF</strong>)</div>
  </div>
  <div class="ft-abs-row">
    <div class="ft-abs-k">It handles</div>
    <div class="ft-abs-v">batching, checkpointing, resuming after a failure, and spreading the work across your compute</div>
  </div>
  <div class="ft-note">The same function runs on a laptop or a whole cluster.</div>
</div>

</div>

<style>
.ft-wrap { display: grid; grid-template-columns: 1.2fr 1fr; gap: 44px; align-items: center; margin-top: 32px; }
.ft-tiers { display: flex; flex-direction: column; gap: 12px; }
.ft-tier {
  display: grid;
  grid-template-columns: auto 1fr auto;
  align-items: center;
  gap: 16px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent-soft);
  border-radius: 12px;
  padding: 14px 18px;
  background: var(--bg-elev);
}
.ft-tier.ft-hot { border-left-color: var(--accent); background: rgba(255, 115, 74, 0.06); }
.ft-badge {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--accent-soft);
}
.ft-body { font-size: 15px; color: var(--fg-muted); line-height: 1.45; }
.ft-body strong { color: var(--fg); }
.ft-cost {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 11.5px; color: var(--fg-dim); white-space: nowrap; text-align: right;
}
.ft-abs {
  border: 1px solid var(--accent); border-radius: 14px;
  padding: 20px 22px; background: rgba(255, 115, 74, 0.05);
  display: flex; flex-direction: column; gap: 14px;
}
.ft-abs-row { display: grid; grid-template-columns: 88px 1fr; gap: 14px; align-items: baseline; }
.ft-abs-k {
  font-family: 'Geist Mono', ui-monospace, monospace;
  font-size: 11px; letter-spacing: .1em; text-transform: uppercase;
  color: var(--accent); padding-top: 2px;
}
.ft-abs-v { font-size: 15px; color: var(--fg); line-height: 1.5; }
.ft-abs-v strong { color: var(--accent-soft); }
.ft-note { font-size: 13px; color: var(--fg-muted); border-top: 1px solid var(--border); padding-top: 12px; }
</style>

<!--
1:05–1:55 · ~50s · SAY:

The features you need cover a huge cost range. At the cheap end, text signals
like the question type or a token count, pure CPU, done in seconds. In the
middle, features that decode the image, like a perceptual hash to flag
near-duplicate pictures. At the expensive end, model features: running a frozen
vision model over every image, and pre-tokenizing the prompts.

Three very different costs. But here's the point: the way you define them does
not change. You write a plain Python function that turns one row into a feature.
That's a UDF.

LanceDB's feature engineering takes it from there. It batches the work,
checkpoints as it goes so a crash resumes instead of restarting, and spreads the
compute out for you. The same function you tested on your laptop runs across a
cluster without changes. You think about the feature; the distribution is
handled.

Let's build all three, on a real table.

[cut to the notebook]
-->
