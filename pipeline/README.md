# Pipeline

Everything used to build the search app and the notebook. Small files only — the corpus,
the embedding arrays and the model caches are deliberately kept out of the repo.

## Where the big files live

```
~/.careernet-cache/
  release/       the release CSVs, unless CAREERNET_SRC points elsewhere
  corpus/        questions.jsonl, answers.jsonl        (cleaned text + labels)
  emb/           *.npy embeddings, gemma_dense_head.npy
  tjs/           node_modules for the transformers.js checks
```

None of that is in git, and all of it is reproducible from the scripts here.
`emb/` is the expensive part: roughly ten minutes on the RTX 5080 to regenerate.

## Order to run things

Every script takes its paths from `paths.py`. Set `CAREERNET_CACHE` to move the cache and
`CAREERNET_SRC` to point at the release CSVs (the `Datasets/` folder of
[careernet-data](https://github.com/RenaissancePhilanthropy/careernet-data), v1.1); nothing
needs editing. `export_retrieval.py`
writes straight into this repo's `docs/search/data/`.

| Step | Script | Does |
|---|---|---|
| 1 | `prep_corpus.py` | Release CSVs to `corpus/*.jsonl` — strips HTML, parses `soc_code` |
| 2 | `augment_corpus.py` | Joins dates and question/answer links back on, **by id, without reordering** |
| 3 | `embed.py` | Embeds both corpora with each model. Per-model dtype; asserts no NaN |
| 4 | `encode_queries.py` | Encodes the held-out test questions and the hand-written queries |
| 5 | `eval.py` | Label-overlap scoring for every model plus BM25 |
| 6 | `export_retrieval.py` | Builds the app's `docs/search/data/` files |
| 7 | `build_nb.py` | Regenerates the Colab notebook |

Step 2 must never rebuild the corpus. The embedding arrays are positional, so any change to
row order silently misaligns every vector. The script asserts order and text are unchanged.

Optional, used to settle specific questions and kept as evidence:

- `hybrid.py` — BM25 and dense fusion swept across the weight range
- `rerank.py`, `rerank_qual.py` — the three cross-encoders, scored and read by hand
- `show_queries.py` — side-by-side top-k for the hand-written queries
- `embed_onnx.py`, `onnx_full.py` — ONNX quantisation checks and the dense-head composition

## Two traps worth knowing

**EmbeddingGemma returns all-NaN in fp16.** Use bfloat16, or float32 where bf16 is absent
(the Colab T4 has none). The failure is silent and reads as a merely bad model — it was first
caught by the *shape* of the scores, not by a crash. Every encode path now asserts on NaN.

**`onnx-community/embeddinggemma-300m-ONNX` omits the model's two Dense layers.** Encoding
through it alone gives vectors orthogonal to the real model, mean cosine 0.004, that still
rank plausibly enough to look fine. Both layers are bias-free and linear, so they compose into
one 768x768 matrix — `gemma_dense_head.npy`, shipped as `dense_head.bin`. Applying it
reproduces the full pipeline to cosine 0.996. The browser and the notebook both do this.

## results/

Evidence behind the model, fusion and reranking choices described above.

| File | Holds |
|---|---|
| `eval_results.csv` | Every model and BM25, both corpora, both label definitions |
| `hybrid_gemma-300m.json` | The fusion weight sweep |
| `rerank_results.json` | All three rerankers, before and after |
| `hand_queries.json` | The queries used for qualitative checks |

## packaging/

What goes in the zip alongside `static/`, which is the repo's `docs/` renamed (GitHub Pages
needs `docs/`; in a zip that name reads as documentation). `serve.py` binds `127.0.0.1` only and stops itself
when its console closes; the launchers just locate a Python and call it. Build the zip with
`ZipInfo.from_file` so timestamps survive, and set `start.command` to mode 0755 by hand —
Windows has no exec bit to carry across.
