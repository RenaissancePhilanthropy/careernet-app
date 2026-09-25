import json, os, sys, time, gc
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer

from paths import CACHE
CORPUS, EMB = CACHE/"corpus", CACHE/"emb"
EMB.mkdir(exist_ok=True)

MODELS = {
  "yuan-2.0-en":  dict(repo="IEITYuan/Yuan-embedding-2.0-en", bs=64),
  "qwen3-0.6b":   dict(repo="Qwen/Qwen3-Embedding-0.6B",      bs=64),
  "qwen3-4b":     dict(repo="Qwen/Qwen3-Embedding-4B",        bs=16),
  "gemma-300m":   dict(repo="google/embeddinggemma-300m",     bs=64, dtype="bfloat16"),
}
MAXLEN = {"questions": 512, "answers": 1024}

def load_corpus(name):
    return [json.loads(l) for l in open(CORPUS/f"{name}.jsonl", encoding="utf-8")]

def run(key):
    cfg = MODELS[key]
    t0=time.time()
    m = SentenceTransformer(cfg["repo"], device="cuda", model_kwargs={"dtype": getattr(torch, cfg.get("dtype","float16"))})
    prompts = dict(m.prompts or {})
    print(f"[{key}] loaded in {time.time()-t0:.0f}s  dim={m.get_sentence_embedding_dimension()}  "
          f"default_maxlen={m.max_seq_length}  prompts={list(prompts)}", flush=True)
    # pick a document-side prompt if the model defines one; else none
    doc_pn = next((p for p in ("document","passage","corpus") if p in prompts), None)
    qry_pn = next((p for p in ("query","question") if p in prompts), None)
    json.dump({"dim": m.get_sentence_embedding_dimension(), "prompts": prompts,
               "doc_prompt_name": doc_pn, "query_prompt_name": qry_pn, "repo": cfg["repo"],
               "dtype": cfg.get("dtype","float16")},
              open(EMB/f"{key}.info.json","w"), indent=2)
    for corpus in ("questions","answers"):
        out = EMB/f"{key}.{corpus}.npy"
        if out.exists(): print(f"[{key}] {corpus} exists, skip", flush=True); continue
        recs = load_corpus(corpus)
        m.max_seq_length = MAXLEN[corpus]
        t=time.time()
        kw = {"prompt_name": doc_pn} if doc_pn else {}
        v = m.encode([r["text"] for r in recs], batch_size=cfg["bs"], normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True, **kw)
        assert not np.isnan(v).any(), f"{key}/{corpus}: NaN in embeddings"
        np.save(out, v.astype(np.float32))
        print(f"[{key}] {corpus}: {v.shape} in {time.time()-t:.0f}s -> {out.name}", flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

for k in (sys.argv[1:] or MODELS):
    try: run(k)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[{k}] FAILED: {e}", flush=True)
