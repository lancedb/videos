# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "lancedb>=0.30",
#     "pylance>=0.18",
#     "transformers>=4.49",
#     "peft>=0.13",
#     "accelerate>=1.0",
#     "bitsandbytes>=0.43",
#     "qwen-vl-utils>=0.0.8",
#     "huggingface-hub>=0.24",
#     "matplotlib>=3.7",
#     "numpy",
#     "pyarrow",
#     "pillow",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fine-tune a VLM on scene-text Q&A, backed by one Lance table

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-1/01_finetune_vlm_lance.py)

    This notebook runs the whole vision-language fine-tuning loop end to end, off a
    single Lance table: download, explore, benchmark reads, QLoRA fine-tune, and
    score before/after. It is the Colab-sized slice of a pipeline that runs the full
    34,602-row corpus on an H100.

    **The model:** `Qwen2.5-VL-3B-Instruct`, LoRA-tuned for
    [TextVQA](https://textvqa.org) (read the text *in* an image, answer a question
    about it).

    **The data:** a curated **text_dense** slice of TextVQA, the images packed with
    the most OCR text. That is the slice where LoRA gives the clearest lift over the
    already-strong base model.

    **Why it fits a small GPU:** the vision tower is the expensive part of a VLM. We
    run it **once**, offline, and store its output (`vision_tower_hiddens`) as a
    column in the Lance table. Training reads that column off disk and skips the
    vision tower entirely, so the loop only holds the 4-bit language model plus a
    LoRA adapter.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Enable a GPU

    The fine-tune and eval sections need a CUDA GPU (~5 GB VRAM is plenty). On
    **molab**, open the notebook specs in the app header and toggle the GPU on. The
    data-layer sections (download, explore, read benchmark) run fine on CPU, so you
    can walk through those without one.
    """)
    return


@app.cell
def _():
    # One import cell: every shared name is defined exactly once (marimo forbids
    # redefining a top-level name in another cell), then passed into the cells below.
    import gc
    import io
    import json
    import os
    import re
    import time
    import warnings

    import lancedb
    import numpy as np
    import torch
    from PIL import Image

    # bitsandbytes emits a noisy _check_is_size FutureWarning on every 4-bit
    # weight load; silence just that one so the eval/train output stays readable.
    warnings.filterwarnings("ignore", message=r"_check_is_size.*", category=FutureWarning)

    HAS_GPU = torch.cuda.is_available()
    return HAS_GPU, Image, gc, io, json, lancedb, np, os, re, time, torch


@app.cell(hide_code=True)
def _(HAS_GPU, mo, torch):
    _gpu = torch.cuda.get_device_name(0) if HAS_GPU else "none detected (CPU only)"
    mo.md(
        f"**torch** `{torch.__version__}`  ·  **GPU:** {_gpu}"
        + ("" if HAS_GPU else "  ·  ⚠️ enable a GPU to run the fine-tune and eval sections")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Download the curated Lance subset

    Computing `vision_tower_hiddens` needs a GPU pass over the images, so we did that
    once, offline, and hosted the result. This notebook just downloads it:

    - `textvqa_colab_train.lance` — curated train subset **with** the cached vision
      features (`vision_tower_hiddens`) plus tokenised prompts
    - `textvqa_colab_val.lance` — held-out curated val subset (raw images, for
      before/after)
    """)
    return


@app.cell
def _(json, lancedb, os):
    from huggingface_hub import snapshot_download

    # Public dataset, no token needed. Override to point at your own bake.
    HF_REPO = os.environ.get("TEXTVQA_COLAB_REPO", "lance-format/textvqa-lance-colab")
    local = snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir="data/colab")
    TRAIN_LANCE = f"{local}/textvqa_colab_train.lance"
    VAL_LANCE = f"{local}/textvqa_colab_val.lance"

    def open_tbl(path):
        name = os.path.basename(path)
        name = name[: -len(".lance")] if name.endswith(".lance") else name
        return lancedb.connect(os.path.dirname(path)).open_table(name)

    train_tbl, val_tbl = open_tbl(TRAIN_LANCE), open_tbl(VAL_LANCE)

    info_path = f"{local}/slice_info.json"
    slice_info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    _cached = [
        c
        for c in train_tbl.schema.names
        if c in ("vision_tower_hiddens", "input_ids", "attention_mask", "labels", "sft_tokens")
    ]
    print("curated slice :", slice_info.get("slice", "(see README)"))
    print("train rows    :", train_tbl.count_rows())
    print("val rows      :", val_tbl.count_rows())
    print("cached columns:", _cached)
    return TRAIN_LANCE, train_tbl, val_tbl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Explore the curated data with LanceDB

    Everything here reads straight from the Lance table through the LanceDB API: no
    full-corpus load into pandas, no separate feature store. The table already ships
    **CLIP image and text embeddings** (`image_emb`, `question_emb`, 512-d), **OCR
    tokens**, and **object classes** next to the raw image bytes, so exploration and
    the vector-search demo need zero extra compute.
    """)
    return


@app.cell
def _(re, train_tbl):
    from collections import Counter

    import matplotlib.pyplot as plt

    # Pull the lightweight columns into a DataFrame (LanceDB -> Arrow -> pandas).
    df = (
        train_tbl.search()
        .select(["question", "answer", "ocr_tokens", "image_classes"])
        .limit(train_tbl.count_rows())
        .to_pandas()
    )

    # Derive a question_type by regex (the kind of column you would Geneva-backfill).
    QPATS = [
        ("how many", r"^\s*how\s+many"),
        ("what is/are", r"^\s*what\s+(is|are)"),
        ("what", r"^\s*what"),
        ("which", r"^\s*which"),
        ("who", r"^\s*who"),
        ("where", r"^\s*where"),
        ("is/does", r"^\s*(is|are|do|does|can)"),
    ]

    def qtype(q):
        for lab, pat in QPATS:
            if re.search(pat, q or "", re.I):
                return lab
        return "other"

    df["qtype"] = df["question"].map(qtype)
    df["ans_words"] = df["answer"].fillna("").map(lambda s: len(s.split()))
    df["ocr_n"] = df["ocr_tokens"].map(lambda x: len(x) if x is not None else 0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
    vc = df["qtype"].value_counts()
    ax[0].barh(vc.index[::-1], vc.values[::-1], color="#4C72B0")
    ax[0].set_title("question type")
    ax[1].hist(df["ans_words"].clip(upper=8), bins=range(0, 10), color="#55A868", align="left")
    ax[1].set_title("answer length (words)")
    ax[1].set_xlabel("words")
    ax[2].hist(df["ocr_n"].clip(upper=60), bins=20, color="#C44E52")
    ax[2].set_title("OCR tokens per image")
    ax[2].set_xlabel("# ocr tokens")
    plt.tight_layout()

    cc = Counter(c for cl in df["image_classes"] if cl is not None for c in cl)
    print("top object classes:", ", ".join(f"{k} ({v})" for k, v in cc.most_common(8)))
    print(
        "median OCR tokens/image:",
        int(df["ocr_n"].median()),
        "| median answer length:",
        int(df["ans_words"].median()),
        "word(s)",
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-modal vector search (text → image), straight from LanceDB

    The table ships CLIP embeddings for the question text (`question_emb`) and the
    image (`image_emb`). So we can take one question's text embedding and ask LanceDB
    for the images whose CLIP embedding is nearest: a text→image retrieval, no model
    to load, just `tbl.search(...)`.
    """)
    return


@app.cell
def _(b64_thumb, mo, np, train_tbl):
    # Pick a query row, use its question_emb as the query vector against image_emb.
    q = train_tbl.search().select(["question", "question_emb"]).limit(40).to_arrow().to_pylist()[11]
    qvec = np.asarray(q["question_emb"], dtype=np.float32)
    hits = (
        train_tbl.search(qvec, vector_column_name="image_emb")
        .select(["image", "question", "answer", "_distance"])
        .limit(5)
        .to_arrow()
        .to_pylist()
    )
    hit_cards = [
        mo.vstack(
            [
                mo.image(f"data:image/jpeg;base64,{b64_thumb(h['image'], 150)}", width=150),
                mo.md(f"d={h['_distance']:.2f}  \nQ: {h['question']}  \n**A: {h['answer']}**"),
            ],
            align="center",
            gap=0.25,
        )
        for h in hits
    ]
    mo.vstack(
        [
            mo.md(f"**Query question:** *{q['question']}*  \nNearest images by CLIP image embedding (L2 distance):"),
            mo.hstack(hit_cards, justify="start", wrap=True, gap=1),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A few curated examples

    Raw image + question + ground-truth answer, read straight from the table.
    """)
    return


@app.cell
def _(b64_thumb, mo, train_tbl):
    samples = (
        train_tbl.search()
        .select(["image", "question", "answer", "ocr_tokens"])
        .limit(4)
        .to_arrow()
        .to_pylist()
    )
    example_cards = [
        mo.vstack(
            [
                mo.image(f"data:image/jpeg;base64,{b64_thumb(s['image'], 170)}", width=170),
                mo.md(
                    f"**{s['question']}**  \n"
                    f"answer: {s['answer']}  \n"
                    f"ocr: {' '.join((s['ocr_tokens'] or [])[:8])}"
                ),
            ],
            align="center",
            gap=0.25,
        )
        for s in samples
    ]
    mo.hstack(example_cards, justify="start", wrap=True, gap=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Read throughput: sequential vs shuffled, LanceDB vs Parquet

    How fast can a dataloader read off disk? We mirror two column groups to plain
    (uncompressed) Parquet and time both access patterns against each, so this
    measures the **access pattern and layout**, not a codec:

    - the **raw multimodal** columns (`image` bytes + `question` + `answer`), and
    - the cached **fixed-size fp16 vision vectors** (`vision_tower_hiddens`).

    **Sequential** streams the split in order (`to_batches`); **shuffled** is what
    training does every epoch, a random batch of rows by index (`.take`). Numbers
    print live from your runtime.
    """)
    return


@app.cell
def _(np, time, train_tbl):
    import pyarrow.dataset as pds
    import pyarrow.parquet as pq

    RAW = ["image", "question", "answer"]  # raw multimodal inputs
    VEC = ["vision_tower_hiddens"]  # cached fixed-size fp16 vision vectors
    BATCH = 8
    lance_ds = train_tbl.to_lance()
    n = train_tbl.count_rows()

    # Mirror each group to a plain Parquet file (uncompressed) to compare formats.
    pq.write_table(lance_ds.to_table(columns=RAW), "raw.parquet", compression="none", row_group_size=64)
    pq.write_table(lance_ds.to_table(columns=VEC), "vec.parquet", compression="none", row_group_size=64)
    raw_pq = pds.dataset("raw.parquet", format="parquet")
    vec_pq = pds.dataset("vec.parquet", format="parquet")
    rng = np.random.default_rng(0)

    def seq(ds, cols):
        t0 = time.time()
        for _b in ds.to_batches(columns=cols, batch_size=BATCH):
            pass
        return n / (time.time() - t0)

    def shuf(ds, cols, NB=20):
        bs = [sorted(rng.choice(n, BATCH, replace=False).tolist()) for _ in range(NB)]
        t0 = time.time()
        for idx in bs:
            ds.take(idx, columns=cols)
        return (NB * BATCH) / (time.time() - t0)

    print(f"{'throughput, rows/s':36}{'LanceDB':>10}{'Parquet':>10}")
    print(f"{'image+Q+A (raw)     sequential':34}{seq(lance_ds, RAW):9.0f}{seq(raw_pq, RAW):9.0f}")
    print(f"{'image+Q+A (raw)     shuffled':34}{shuf(lance_ds, RAW):9.0f}{shuf(raw_pq, RAW):9.0f}")
    print(f"{'vision vectors fp16 sequential':34}{seq(lance_ds, VEC):9.0f}{seq(vec_pq, VEC):9.0f}")
    # Parquet fp16 shuffled re-decodes whole row groups per random batch (slow), so
    # skip it. The sequential row already shows the gap; LanceDB shuffled stays fast.
    print(f"{'vision vectors fp16 shuffled':34}{shuf(lance_ds, VEC):9.0f}{'--':>9}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · QLoRA fine-tune, from the cached columns

    We use the notebook's own helpers (`build_model`, `forward_cached`,
    `make_cached_loader` — hidden cells in the appendix at the bottom) so this is
    the real code path, driven inline.

    `build_model(..., load_4bit=True)` loads the LLM in 4-bit NF4 (~2 GB instead of
    ~7.5 GB), **deletes the vision tower** (we have its output cached), and wraps the
    LLM's q/k/v/o with a LoRA adapter. The loop pulls `vision_tower_hiddens` +
    `input_ids` + `labels` from Lance and injects the cached hiddens at the
    `<|image_pad|>` positions. No vision tower, no image decode, no tokenization in
    the loop.
    """)
    return


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="▶ Run fine-tune (needs GPU)")
    train_button
    return (train_button,)


@app.cell
def _(
    HAS_GPU,
    IMAGE_PAD_TOKEN,
    QWEN_MODEL_ID,
    build_model,
    mo,
    torch,
    train_button,
):
    mo.stop(
        not HAS_GPU,
        mo.md("> ⚠️ **GPU required.** Enable a GPU in the notebook specs, then re-run."),
    )
    mo.stop(not train_button.value, mo.md("> ▶ Click **Run fine-tune** above to build the model and train."))

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    image_pad_id = tok.convert_tokens_to_ids(IMAGE_PAD_TOKEN)

    model = build_model(use_lora=True, lora_r=16, load_4bit=True)
    model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=3e-5, betas=(0.9, 0.95))
    device = torch.device("cuda:0")
    return device, image_pad_id, model, optim, trainable


@app.cell
def _(
    TRAIN_LANCE,
    device,
    forward_cached,
    image_pad_id,
    make_cached_loader,
    model,
    optim,
    time,
    torch,
    trainable,
):
    MAX_STEPS = 300  # demo scale; the curated lift comes from ~a couple of epochs
    GRAD_ACCUM = 4
    loader = make_cached_loader(TRAIN_LANCE, batch_size=2, num_workers=0, shuffle=True, seed=0)

    step, accum, t0 = 0, 0, time.time()
    optim.zero_grad(set_to_none=True)
    done = False
    for epoch in range(10):
        if done:
            break
        for batch in loader:
            batch = batch.to(device)
            loss = forward_cached(model, batch, image_pad_id)
            (loss / GRAD_ACCUM).backward()
            accum += 1
            if accum >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                accum = 0
                step += 1
                sps = (step * GRAD_ACCUM * 2) / (time.time() - t0)
                if step % 10 == 0 or step == MAX_STEPS:
                    print(f"step {step:3d}/{MAX_STEPS}  loss={loss.item():.4f}  {sps:.1f} samples/s")
                if step >= MAX_STEPS:
                    done = True
                    break

    ADAPTER_DIR = "runs/colab_lora/lora"
    model.save_pretrained(ADAPTER_DIR)
    print("saved adapter to", ADAPTER_DIR, "| peak VRAM %.1f GB" % (torch.cuda.max_memory_allocated() / 1e9))
    return ADAPTER_DIR, loader


@app.cell
def _(gc, loader, model, optim, torch):
    # Free the training model before loading the full model for eval.
    del model, optim, loader
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM after cleanup: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Before / after: does the tuned model read text better?

    Now we load the full model (vision tower included, in 4-bit) and generate on the
    held-out curated val split, once with the base weights and once with our LoRA
    adapter. We score every val row (official TextVQA accuracy) for the headline
    number and show a handful side by side.
    """)
    return


@app.cell
def _(mo):
    eval_button = mo.ui.run_button(label="▶ Run before/after eval (needs GPU)")
    eval_button
    return (eval_button,)


@app.cell
def _(
    ADAPTER_DIR,
    Image,
    eval_button,
    gc,
    generate,
    io,
    load_model,
    mo,
    torch,
    val_tbl,
):
    mo.stop(not eval_button.value, mo.md("> ▶ Click **Run before/after eval** above (run the fine-tune first)."))

    EVAL_N = min(val_tbl.count_rows(), 256)  # held-out curated val rows to score
    GRID_K = 6  # how many to show side by side
    rows = (
        val_tbl.search()
        .select(["image", "question", "answer", "answers"])
        .limit(EVAL_N)
        .to_arrow()
        .to_pylist()
    )

    def run(adapter):
        m, proc = load_model(adapter_dir=adapter, load_4bit=True)
        outs = []
        for r in rows:
            img = Image.open(io.BytesIO(r["image"])).convert("RGB")
            outs.append(generate(m, proc, img, r["question"]))
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return outs

    print(f"scoring {EVAL_N} held-out curated val rows with base, then tuned ...")
    base_ans = run(None)
    tuned_ans = run(ADAPTER_DIR)
    return EVAL_N, GRID_K, base_ans, rows, tuned_ans


@app.cell
def _(EVAL_N, GRID_K, b64_thumb, base_ans, mo, rows, score_one, tuned_ans):
    base_score = sum(score_one(b, r["answers"]) for b, r in zip(base_ans, rows)) / len(rows)
    tuned_score = sum(score_one(t, r["answers"]) for t, r in zip(tuned_ans, rows)) / len(rows)

    # One dict per example; a native marimo table renders it (no raw HTML).
    win_idx = set()
    table_rows = []
    for i, (r, b, t) in enumerate(list(zip(rows, base_ans, tuned_ans))[:GRID_K]):
        bs, ts = score_one(b, r["answers"]), score_one(t, r["answers"])
        if ts > bs:
            win_idx.add(i)
        table_rows.append(
            {
                "Image": f"data:image/jpeg;base64,{b64_thumb(r['image'])}",
                "Question": r["question"],
                "Base": b,
                "Tuned": t,
                "Ground truth": ", ".join(r["answers"][:5]),
                "Tuned wins": "✅" if ts > bs else "",
            }
        )

    def _style(row_id, _column, _value):
        # Highlight rows where the tuned model beat the base model.
        try:
            return {"backgroundColor": "#e6ffe6"} if int(row_id) in win_idx else {}
        except (TypeError, ValueError):
            return {}

    results_table = mo.ui.table(
        table_rows,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        format_mapping={"Image": lambda uri: mo.image(uri, width=150)},
        wrapped_columns=["Question", "Base", "Tuned", "Ground truth"],
        text_justify_columns={"Tuned wins": "center"},
        style_cell=_style,
    )

    mo.vstack(
        [
            mo.md(
                f"**TextVQA accuracy on {EVAL_N} held-out curated val rows**  \n"
                f"base: `{base_score:.3f}`  ·  tuned: `{tuned_score:.3f}` "
                f"({(tuned_score - base_score) * 100:+.1f} pp)  ·  green rows = tuned beat base"
            ),
            results_table,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    You ran the full shape of a VLM fine-tune on a curated **text_dense** slice, all
    off one Lance table:

    1. explored the slice (distributions + cross-modal vector search),
    2. read the vision-tower output that was computed once and stored as a column
       (`vision_tower_hiddens`),
    3. trained a vision-tower-free, 4-bit LoRA loop reading that column off disk,
    4. compared base vs tuned on held-out images.

    The same code runs the full 34,602-row corpus on an H100. On this curated slice
    the lift was **0.799 → 0.820** (+2.1 pp).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Appendix — helper code

    The three hidden cells below hold the data and model plumbing, frozen from
    [lancedb/tmls-2026-demo](https://github.com/lancedb/tmls-2026-demo): the official
    TextVQA scorer plus model loading/generation, the LanceDB Permutation-API
    dataloader, and the QLoRA training helpers. They live inside the notebook so it
    is fully self-contained — "Open in molab" needs only this one file. Click a cell
    to expand its code.
    """)
    return


@app.cell(hide_code=True)
def _(Image, io, re, torch):
    # ── Eval helpers (from vlm/eval.py) ─────────────────────────────────────
    import base64

    QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    TEXTVQA_HINT = "Answer the question using a single word or short phrase, no explanation."

    # Official TextVQA scoring: min(matches/3, 1) with text normalisation.
    PUNCT_RE = re.compile(r"[\.\,\!\?\;\:\(\)\"]+")
    WHITE_RE = re.compile(r"\s+")
    ARTICLES = {"a", "an", "the"}

    def normalise(s):
        s = s.strip().lower()
        s = PUNCT_RE.sub("", s)
        s = WHITE_RE.sub(" ", s)
        return " ".join(t for t in s.split() if t not in ARTICLES)

    def score_one(pred, gts):
        if not pred:
            return 0.0
        p = normalise(pred)
        return min(sum(1 for g in gts if normalise(g) == p) / 3.0, 1.0)

    def b64_thumb(image_bytes, size=192):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def load_model(adapter_dir, load_4bit=False):
        """Full Qwen2.5-VL (vision tower included), optionally with a LoRA adapter."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        kwargs = dict(attn_implementation="sdpa")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "cuda:0"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL_ID, **kwargs)
        if adapter_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_dir)
            # merge_and_unload can't fold LoRA into 4-bit weights; keep the
            # adapter active for generation in that case.
            if not load_4bit:
                model = model.merge_and_unload()
        model.eval()
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        return model, processor

    @torch.no_grad()
    def generate(model, processor, image, question, max_new_tokens=16):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question}\n\n{TEXTVQA_HINT}"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return processor.tokenizer.decode(gen, skip_special_tokens=True).strip()

    return QWEN_MODEL_ID, b64_thumb, generate, load_model, score_one


@app.cell(hide_code=True)
def _(lancedb, np, torch):
    # ── LanceDB dataloader (from vlm/dataloader.py + vlm/schema.py) ─────────
    from dataclasses import dataclass
    from pathlib import Path

    import pyarrow as pa
    from lancedb.permutation import Permutation

    # Locked vision params: 560 px, patch 14, spatial merge 2
    # -> 400 LLM tokens per image x 2048 hidden dims, 512-token text budget.
    LLM_TOKENS_PER_IMAGE = 400
    VISION_HIDDEN = 2048
    MAX_TEXT_TOKENS = 512

    TOKEN_FIELDS = ("input_ids", "attention_mask", "labels")
    CACHED_FLAT_COLS = ["vision_tower_hiddens", *TOKEN_FIELDS]
    CACHED_STRUCT_COLS = ["vision_tower_hiddens", "sft_tokens"]

    def split_db(db):
        """``data/textvqa.lance`` -> ``("data", "textvqa")`` for lancedb.connect."""
        p = Path(db)
        name = p.name[: -len(".lance")] if p.name.endswith(".lance") else p.name
        return str(p.parent), name

    def as_array(col):
        """RecordBatch columns are Arrays; Table columns are ChunkedArrays."""
        return col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col

    @dataclass
    class CachedBatch:
        vision_hiddens: torch.Tensor  # fp16 [B, LLM_TOKENS_PER_IMAGE, VISION_HIDDEN]
        input_ids: torch.Tensor  # int64 [B, MAX_TEXT_TOKENS]
        attention_mask: torch.Tensor
        labels: torch.Tensor

        def to(self, device, non_blocking=True):
            return CachedBatch(
                vision_hiddens=self.vision_hiddens.to(device, non_blocking=non_blocking),
                input_ids=self.input_ids.to(device, non_blocking=non_blocking),
                attention_mask=self.attention_mask.to(device, non_blocking=non_blocking),
                labels=self.labels.to(device, non_blocking=non_blocking),
            )

    def cached_collate(batch):
        bsz = batch.num_rows
        flat_v = as_array(batch.column("vision_tower_hiddens")).values.to_numpy(zero_copy_only=False)
        vision = torch.from_numpy(flat_v.reshape(bsz, LLM_TOKENS_PER_IMAGE, VISION_HIDDEN))

        # Tokens live either as flat columns (direct backfill) or inside an
        # ``sft_tokens`` struct (Geneva sft_tokens UDF). Handle both.
        if "sft_tokens" in batch.schema.names:
            struct = as_array(batch.column("sft_tokens"))
            get = struct.field
        else:
            get = lambda f: as_array(batch.column(f))

        def to_long(arr):
            flat = arr.values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            return torch.from_numpy(flat.reshape(bsz, MAX_TEXT_TOKENS)).to(torch.long)

        return CachedBatch(
            vision_hiddens=vision,
            input_ids=to_long(get("input_ids")),
            attention_mask=to_long(get("attention_mask")),
            labels=to_long(get("labels")),
        )

    class LancePermutationDataset(torch.utils.data.Dataset):
        """Stores connection params; each worker reopens its own Permutation."""

        def __init__(self, uri, table_name, columns):
            self.uri = uri
            self.table_name = table_name
            self.columns = columns
            self.perm = None
            self.length = len(lancedb.connect(uri).open_table(table_name))

        def __len__(self):
            return self.length

        def __getstate__(self):
            # Permutation holds Rust async state — zero it so each worker reopens its own.
            state = self.__dict__.copy()
            state["perm"] = None
            return state

        def ensure_open(self):
            if self.perm is None:
                db = lancedb.connect(self.uri)
                self.perm = (
                    Permutation.identity(db.open_table(self.table_name))
                    .select_columns(self.columns)
                    .with_format("arrow")
                )

        def __getitem__(self, idx):
            self.ensure_open()
            return self.perm[idx]

        def __getitems__(self, indices):
            self.ensure_open()
            return self.perm.__getitems__(indices)

    def make_cached_loader(db, batch_size=2, num_workers=0, shuffle=True, seed=42):
        """DataLoader over the cached tier-3 columns (zero vision-tower work)."""
        uri, table_name = split_db(db)
        tbl_names = set(lancedb.connect(uri).open_table(table_name).schema.names)
        struct = "sft_tokens" in tbl_names and not set(TOKEN_FIELDS) <= tbl_names
        columns = CACHED_STRUCT_COLS if struct else CACHED_FLAT_COLS
        dataset = LancePermutationDataset(uri, table_name, columns)
        sampler = None
        if shuffle:
            g = torch.Generator()
            g.manual_seed(seed)
            sampler = torch.utils.data.RandomSampler(dataset, generator=g)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=cached_collate,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0),
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )

    return (make_cached_loader,)


@app.cell(hide_code=True)
def _(QWEN_MODEL_ID, torch):
    # ── Training helpers (from vlm/train_qwen25vl_lora.py) ──────────────────
    IMAGE_PAD_TOKEN = "<|image_pad|>"

    def build_model(use_lora, lora_r, load_4bit=False):
        """Load Qwen2.5-VL with the vision tower deleted; optional NF4 + LoRA.

        The vision tower is ~1.3 GB; not loading it frees that much VRAM. With
        ``load_4bit`` the LLM weights are NF4-quantised (bitsandbytes) so the
        3.75 B-param model plus LoRA fits a small GPU.
        """
        from transformers import Qwen2_5_VLForConditionalGeneration

        kwargs = dict(attn_implementation="sdpa")
        if load_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = "cuda:0"

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_MODEL_ID, **kwargs)

        # Free the vision tower weights — its output is already cached in Lance.
        del model.model.visual
        model.model.visual = None
        torch.cuda.empty_cache()

        if load_4bit and use_lora:
            # QLoRA: cast norms to fp32, enable input grads, gradient checkpointing.
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if use_lora:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_r * 2,
                lora_dropout=0.05,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            model.print_trainable_parameters()
        else:
            for p in model.parameters():
                p.requires_grad = True

        return model

    def forward_cached(model, batch, image_pad_id):
        """Inject cached vision hiddens at <|image_pad|> positions and run the LLM."""
        # Discover the actual nn.Module behind PEFT wrappers.
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        embed = base.model.get_input_embeddings()
        inputs_embeds = embed(batch.input_ids)  # [B, T, D]

        # Mask = (input_ids == <|image_pad|>) broadcast over the hidden dim.
        mask = (batch.input_ids == image_pad_id).unsqueeze(-1).expand_as(inputs_embeds)

        # vision_hiddens: fp16 [B, tokens, D] -> LLM dtype; masked_scatter
        # consumes the matching number of elements row-major.
        vision_flat = batch.vision_hiddens.to(inputs_embeds.dtype).reshape(-1, inputs_embeds.shape[-1])
        inputs_embeds = inputs_embeds.masked_scatter(mask, vision_flat)

        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask,
            labels=batch.labels,
        )
        return out.loss

    return IMAGE_PAD_TOKEN, build_model, forward_cached


if __name__ == "__main__":
    app.run()
