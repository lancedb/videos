# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "lancedb==0.30.2",
#     "pylance==3.0.0",
#     "geneva==0.12.0",
#     "pyarrow",
#     "numpy",
#     "pillow",
#     "matplotlib>=3.7",
#     "huggingface-hub>=0.24",
#     "transformers>=4.49",
#     "accelerate>=1.0",
#     "qwen-vl-utils>=0.0.8",
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
    # Feature engineering for fine-tuning pipelines

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/lancedb/videos/blob/main/vlm-materialized-features/video-2/02_feature_engineering.py)

    A fine-tuning pipeline doesn't run on raw images and text. It runs on *derived*
    data. This notebook builds those feature columns directly on one LanceDB table,
    cheapest to most expensive, with one abstraction.

    The dataset is a small [TextVQA](https://textvqa.org) subset: 600 rows of image
    plus question plus answer. We add three tiers of features on top of it:

    - **Tier 1**, text signals (CPU, seconds)
    - **Tier 2**, an image fingerprint for near-duplicates (CPU)
    - **Tier 3**, a frozen vision model's embeddings per image (GPU)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Setup

    Feature engineering here uses the **geneva** package. You define each
    feature as a plain Python function (a UDF); LanceDB's feature engineering runs it
    over the table, handles batching and checkpointing, and can spread the work
    across your compute. Tiers 1 and 2 run on CPU. Tier 3 needs a GPU (about 5 GB
    VRAM); on **molab**, toggle the GPU on via the notebook specs button.
    """)
    return


@app.cell
def _():
    # One import cell: shared names defined once, passed into the cells below.
    import re
    import warnings

    import geneva
    import lance
    import lancedb
    import numpy as np
    import pyarrow as pa
    import torch
    from geneva.transformer import udf

    warnings.filterwarnings("ignore", message=r"_check_is_size.*", category=FutureWarning)

    HAS_GPU = torch.cuda.is_available()
    return HAS_GPU, geneva, lance, lancedb, np, pa, re, torch, udf


@app.cell(hide_code=True)
def _(HAS_GPU, mo, torch):
    _gpu = torch.cuda.get_device_name(0) if HAS_GPU else "none detected (CPU only)"
    mo.md(
        f"**torch** `{torch.__version__}`  ·  **GPU:** {_gpu}"
        + ("" if HAS_GPU else "  ·  Tiers 1 and 2 run here; enable a GPU for Tier 3")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The table

    Download the public subset, then project out just the **raw** columns (`id`,
    `image`, `question`, `answer`) into a fresh local table. The download ships with
    precomputed feature columns, so starting from a raw-only copy means every column
    this notebook adds is genuinely new. The raw image, question, and answer never
    move again.
    """)
    return


@app.cell
def _(geneva, lance, lancedb):
    from huggingface_hub import snapshot_download

    LOCAL = snapshot_download(
        repo_id="lance-format/textvqa-lance-colab",
        repo_type="dataset",
        local_dir="data/colab",
    )

    # Project just the raw columns into a fresh table: every feature column the
    # notebook adds from here on is genuinely new. (The download also ships the
    # precomputed features; we deliberately leave those behind.)
    RAW_COLS = ["id", "image", "question", "answer"]
    raw = lance.dataset(f"{LOCAL}/textvqa_colab_train.lance").to_table(columns=RAW_COLS)

    DEMO_DIR = "data/demo"
    TABLE_NAME = "textvqa_raw"
    DEMO_PATH = f"{DEMO_DIR}/{TABLE_NAME}.lance"
    lance.write_dataset(raw, DEMO_PATH, mode="overwrite")

    # geneva handle for feature engineering (backfills); plain Lance handles for reads.
    gdb = geneva.connect(DEMO_DIR)

    # A LanceDB handle on the downloaded table (it ships CLIP embeddings), for the
    # cross-modal search demo below.
    train_tbl = lancedb.connect(LOCAL).open_table("textvqa_colab_train")

    def read_columns(cols, limit=None):
        ds = lance.dataset(DEMO_PATH)  # reopened each call to pick up new columns
        n = ds.count_rows() if limit is None else min(limit, ds.count_rows())
        return ds.to_table(columns=cols, limit=n).to_pandas()

    print("rows   :", raw.num_rows)
    print("columns:", raw.schema.names)
    return DEMO_PATH, TABLE_NAME, gdb, read_columns, train_tbl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cross-modal vector search (text → image), straight from LanceDB

    The table ships CLIP embeddings for the question text (`question_emb`) and the
    image (`image_emb`). So we can take one question's text embedding and ask LanceDB
    for the images whose CLIP embedding is nearest. That gives us text→image
    retrieval without loading any model: the whole thing is one `tbl.search(...)`
    call.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Thumbnail helper for the image galleries.
    import base64
    import io

    from PIL import Image

    def b64_thumb(image_bytes, size=192):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img.thumbnail((size, size))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    return (b64_thumb,)


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
    ## 2 · Tier 1: text signals

    The cheapest features are plain functions over a text column. `question_type`
    buckets each question by its leading words. The `@udf` decorator declares the
    output type and which columns feed it; the body is ordinary Python.
    """)
    return


@app.cell
def _(pa, re, udf):
    QUESTION_PATTERNS = [
        ("how_many", re.compile(r"^\s*how\s+many\b", re.I)),
        ("what_color", re.compile(r"^\s*what\s+(is\s+the\s+)?color\b", re.I)),
        ("what_brand", re.compile(r"^\s*what\s+(is\s+the\s+)?(brand|company|make)\b", re.I)),
        ("what_number", re.compile(r"^\s*what\s+number\b", re.I)),
        ("what", re.compile(r"^\s*what\b", re.I)),
        ("which", re.compile(r"^\s*which\b", re.I)),
        ("who", re.compile(r"^\s*who\b", re.I)),
        ("where", re.compile(r"^\s*where\b", re.I)),
        ("is_does", re.compile(r"^\s*(is|are|does|do|can)\b", re.I)),
    ]

    @udf(data_type=pa.string(), input_columns=["question"])
    def question_type(question: str) -> str:
        if not question:
            return "other"
        for label, pat in QUESTION_PATTERNS:
            if pat.search(question):
                return label
        return "other"

    return (question_type,)


@app.cell
def _(TABLE_NAME, gdb, question_type):
    # Define what the column is, then materialize it. LanceDB's feature engineering
    # runs the UDF over the table inside a local compute context (workers on this
    # machine); at scale the same call fans out across a cluster.
    _table = gdb.open_table(TABLE_NAME)
    if "question_type" not in {f.name for f in _table.schema}:
        _table.add_columns({"question_type": question_type})
        with gdb.local_ray_context():
            _table.backfill("question_type", udf=question_type, concurrency=2, task_size=256)
    tier1_done = "question_type"
    return (tier1_done,)


@app.cell
def _(mo, plt_counts, read_columns, tier1_done):
    df1 = read_columns([tier1_done])
    fig1 = plt_counts(df1[tier1_done].value_counts(), "question type")
    mo.vstack([mo.md(f"New column **`{tier1_done}`**, materialized on the table:"), fig1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Tier 2: an image fingerprint

    Same abstraction, heavier input. `dhash` decodes each image once and computes a
    64-bit perceptual hash; near-duplicate images share most of their bits. The
    decorator and the backfill call are identical to Tier 1; only the body changed.
    """)
    return


@app.cell
def _(pa, udf):
    @udf(data_type=pa.uint64(), input_columns=["image"])
    def dhash(image: bytes) -> int:
        # Shrink to 9x8 grayscale, compare adjacent pixels, pack 64 bits into a uint64.
        import io

        import numpy as np
        from PIL import Image

        if not image:
            return 0
        img = Image.open(io.BytesIO(image)).convert("L").resize((9, 8), Image.LANCZOS)
        a = np.asarray(img, dtype=np.int16)
        bits = (a[:, 1:] > a[:, :-1]).flatten()
        out = 0
        for b in bits:
            out = (out << 1) | int(b)
        return out & ((1 << 64) - 1)

    return (dhash,)


@app.cell
def _(TABLE_NAME, dhash, gdb):
    _table = gdb.open_table(TABLE_NAME)
    if "dhash" not in {f.name for f in _table.schema}:
        _table.add_columns({"dhash": dhash})
        with gdb.local_ray_context():
            _table.backfill("dhash", udf=dhash, concurrency=2, task_size=256)
    tier2_done = "dhash"
    return (tier2_done,)


@app.cell
def _(mo, read_columns, tier2_done):
    df2 = read_columns([tier2_done])
    n_unique = df2[tier2_done].nunique()
    mo.md(
        f"New column **`{tier2_done}`**: {n_unique} distinct hashes across "
        f"{len(df2)} rows. Rows sharing a hash are near-duplicate images."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Tier 3: a model feature (GPU)

    The headline feature: run a frozen vision model over every image once and store
    its output as a fixed-size vector per row. It is a stateful UDF, a class that
    lazy-loads the model in the worker so the driver never touches the GPU. The
    abstraction is the same; only the cost is different.

    This is gated behind a button because it needs a GPU (and the first run also
    downloads the ~7 GB model). On molab, toggle the GPU on and click the button:
    the backfill runs for real over all 600 images, with the knobs that matter at
    scale (`concurrency`, `task_size`, `checkpoint_size`).
    """)
    return


@app.cell
def _(mo):
    t3_button = mo.ui.run_button(label="▶ Run Tier 3 backfill (needs GPU)")
    t3_button
    return (t3_button,)


@app.cell
def _(pa, torch, udf):
    IMAGE_PX = 560
    LLM_TOKENS_PER_IMAGE = 400
    VISION_HIDDEN = 2048
    QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

    class VisionTowerEmbedder:
        """Decode JPEG, run Qwen2.5-VL's frozen vision tower, return its merger output."""

        def __init__(self):
            self._model = None
            self._processor = None

        def _lazy_load(self):
            if self._model is not None:
                return
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
            ).model.visual.eval()
            self._processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
            self._dtype = next(self._model.parameters()).dtype

        def __call__(self, image: bytes) -> list:
            import io

            from PIL import Image

            self._lazy_load()
            if not image:
                return [0.0] * (LLM_TOKENS_PER_IMAGE * VISION_HIDDEN)
            img = Image.open(io.BytesIO(image)).convert("RGB").resize((IMAGE_PX, IMAGE_PX), Image.LANCZOS)
            messages = [{"role": "user", "content": [
                {"type": "image", "image": img,
                 "min_pixels": IMAGE_PX * IMAGE_PX, "max_pixels": IMAGE_PX * IMAGE_PX},
                {"type": "text", "text": "x"},
            ]}]
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self._processor(text=[text], images=[img], return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = self._model(inputs["pixel_values"].to(self._dtype), grid_thw=inputs["image_grid_thw"])
            return out.pooler_output.to(torch.float16).flatten().cpu().numpy().tolist()

    # Wrap the class as a UDF with a fixed-size-list output type. num_gpus=1
    # tells the compute context to schedule this UDF's worker on a GPU.
    vision_tower_hiddens = udf(
        data_type=pa.list_(pa.float16(), LLM_TOKENS_PER_IMAGE * VISION_HIDDEN),
        input_columns=["image"],
        num_gpus=1,
    )(VisionTowerEmbedder)()  # final () creates the Geneva UDF
    return (vision_tower_hiddens,)


@app.cell
def _(HAS_GPU, TABLE_NAME, gdb, mo, t3_button, vision_tower_hiddens):
    mo.stop(not HAS_GPU, mo.md("> ⚠️ **GPU required.** Enable a GPU in the notebook specs, then re-run."))
    mo.stop(not t3_button.value, mo.md("> ▶ Click **Run Tier 3 backfill** above."))

    import time

    _table = gdb.open_table(TABLE_NAME)
    if "vision_tower_hiddens" in {f.name for f in _table.schema}:
        _msg = "`vision_tower_hiddens` is already on the table (backfilled earlier in this session)."
    else:
        _t0 = time.time()
        _table.add_columns({"vision_tower_hiddens": vision_tower_hiddens})
        with gdb.local_ray_context():
            _table.backfill(
                "vision_tower_hiddens", udf=vision_tower_hiddens,
                concurrency=1, task_size=128, checkpoint_size=64,  # one worker per GPU
            )
        _msg = f"Backfilled `vision_tower_hiddens` in {time.time() - _t0:.0f}s."
    tier3_done = "vision_tower_hiddens"
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Without the abstraction

    The hidden cell below is the same Tier-3 work written by hand: a single-process
    loop that reads batches, runs the model, packs fixed-size-list arrays, and writes
    the column, plus the bookkeeping to make a re-run safe. It works, but it is the
    runner you would otherwise maintain yourself, and it still only uses one machine.
    That plumbing, and scaling past one machine, is what the UDF abstraction removes.
    """)
    return


@app.cell(hide_code=True)
def _(lance, np, pa):
    # The hand-rolled single-process alternative to a feature-engineering backfill.
    # Reuses the same embedder logic, but you own the batching, the Arrow packing,
    # and the partial-state teardown, and there is no distribution.
    def hand_rolled_tier3(train_path, embed, llm_tokens, vision_hidden, batch_size=8):
        ds = lance.dataset(train_path)
        v_dim = llm_tokens * vision_hidden

        name = "vision_tower_hiddens"
        if name in ds.schema.names:  # partial-state teardown so add_columns is atomic
            ds.drop_columns([name])
            ds = lance.dataset(train_path)

        def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
            images = batch.column("image").to_pylist()
            vis = np.asarray([embed(img) for img in images], dtype=np.float16)
            fsl = pa.FixedSizeListArray.from_arrays(pa.array(vis.reshape(-1), type=pa.float16()), v_dim)
            return pa.RecordBatch.from_arrays([fsl], names=[name])

        ds.add_columns(transform, read_columns=["image"], batch_size=batch_size)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap: one table, raw to training-ready

    Start with image, question, answer; end with three tiers of features attached,
    each a zero-copy column add. The expensive computations are now materialized on
    the same table that holds the raw data, ready for a training loop to read them
    straight off disk. Here is the final schema, with the columns this notebook
    added highlighted:
    """)
    return


@app.cell
def _(DEMO_PATH, lance, mo, tier1_done, tier2_done):
    _ = (tier1_done, tier2_done)  # depend on the backfills so this runs after them

    FEATURES = {
        "question_type": "Tier 1 backfill · question_type UDF",
        "dhash": "Tier 2 backfill · dhash UDF",
        "vision_tower_hiddens": "Tier 3 backfill · vision-tower UDF",
    }
    schema = lance.dataset(DEMO_PATH).schema
    new_idx = {i for i, f in enumerate(schema) if f.name in FEATURES}
    schema_rows = [
        {
            "Column": f.name,
            "Type": str(f.type),
            "Source": FEATURES.get(f.name, "raw data"),
            "New": "✅" if f.name in FEATURES else "",
        }
        for f in schema
    ]

    def _style(row_id, _column, _value):
        # Highlight the feature columns this notebook added.
        try:
            return {"backgroundColor": "#e6ffe6"} if int(row_id) in new_idx else {}
        except (TypeError, ValueError):
            return {}

    schema_table = mo.ui.table(
        schema_rows,
        selection=None,
        pagination=False,
        show_column_summaries=False,
        show_data_types=False,
        wrapped_columns=["Type", "Source"],
        text_justify_columns={"New": "center"},
        style_cell=_style,
    )
    mo.vstack(
        [
            mo.md(
                f"**{len(schema.names)} columns**: the raw data plus "
                f"**{len(new_idx)} feature columns**, each defined as a UDF and "
                "materialized one backfill at a time (highlighted rows)."
            ),
            schema_table,
        ]
    )
    return


@app.cell
def _(plt):
    # Small helper: horizontal bar chart of a value_counts Series.
    def plt_counts(counts, title):
        counts = counts.sort_values()
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(counts.index.astype(str), counts.values, color="#e8593c")
        ax.set_title(title)
        fig.tight_layout()
        return fig

    return (plt_counts,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


if __name__ == "__main__":
    app.run()
