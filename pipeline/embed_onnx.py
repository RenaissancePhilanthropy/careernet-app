import json, sys, time, gc
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from paths import CACHE, EMB
REPO="onnx-community/embeddinggemma-300m-ONNX"
VARIANTS={"gemma-onnx-q4f16":"onnx/model_q4f16.onnx", "gemma-onnx-q4":"onnx/model_q4.onnx",
          "gemma-onnx-int8":"onnx/model_quantized.onnx"}
HQ=json.load(open(CACHE/"hand_queries.json"))

def run(key):
    fn=VARIANTS[key]
    m=SentenceTransformer(REPO, backend="onnx", model_kwargs={"file_name":fn}, device="cpu")
    pr=dict(m.prompts or {})
    dq = "Retrieval-query" if "Retrieval-query" in pr else "query"
    dd = "Retrieval-document" if "Retrieval-document" in pr else "document"
    # match the torch run's prompt choice exactly
    dq, dd = ("query" if "query" in pr else dq), ("document" if "document" in pr else dd)
    print(f"[{key}] {fn} dim={m.get_sentence_embedding_dimension()} qp={dq!r} dp={dd!r}", flush=True)
    json.dump({"dim":m.get_sentence_embedding_dimension(),"prompts":pr,"repo":REPO,
               "doc_prompt_name":dd,"query_prompt_name":dq,"onnx_file":fn},
              open(EMB/f"{key}.info.json","w"), indent=2)
    for corpus in ("questions",):
        out=EMB/f"{key}.{corpus}.npy"
        if out.exists(): continue
        recs=[json.loads(l) for l in open(CACHE/"corpus"/f"{corpus}.jsonl",encoding="utf-8")]
        m.max_seq_length = 512 if corpus=="questions" else 1024
        t=time.time()
        v=m.encode([r["text"] for r in recs],batch_size=32,normalize_embeddings=True,
                   convert_to_numpy=True,show_progress_bar=True,prompt_name=dd)
        assert not np.isnan(v).any(), f"{key}/{corpus}: NaN"
        np.save(out, v.astype(np.float32)); print(f"[{key}] {corpus} {v.shape} {time.time()-t:.0f}s",flush=True)
    recs=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
    test=[r for r in recs if r["split"]=="Test"]
    m.max_seq_length=512
    v=m.encode([r["text"] for r in test],batch_size=32,normalize_embeddings=True,convert_to_numpy=True,prompt_name=dq)
    assert not np.isnan(v).any(), f"{key}: NaN queries"
    np.save(EMB/f"{key}.testq.npy",v.astype(np.float32))
    np.save(EMB/f"{key}.handq.npy",
            m.encode(HQ,normalize_embeddings=True,convert_to_numpy=True,prompt_name=dq).astype(np.float32))
    print(f"[{key}] queries done",flush=True)
    del m; gc.collect()

for k in (sys.argv[1:] or VARIANTS):
    try: run(k)
    except Exception as e:
        import traceback; traceback.print_exc(); print(f"[{k}] FAILED: {e}",flush=True)
