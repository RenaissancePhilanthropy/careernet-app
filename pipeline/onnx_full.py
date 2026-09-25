import json, sys, time, numpy as np, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from paths import CACHE, EMB
VARIANTS={"gemma-onnxfull-q4f16":"onnx/model_q4f16.onnx","gemma-onnxfull-q4":"onnx/model_q4.onnx",
          "gemma-onnxfull-int8":"onnx/model_quantized.onnx"}
HQ=json.load(open(CACHE/"hand_queries.json"))
ref=SentenceTransformer("google/embeddinggemma-300m", device="cpu")
tail=list(ref)[2:]           # Dense, Dense, Normalize
print("grafting:",[type(x).__name__ for x in tail],flush=True)

def run(key,fn):
    on=SentenceTransformer("onnx-community/embeddinggemma-300m-ONNX", backend="onnx",
                           model_kwargs={"file_name":fn}, device="cpu")
    mods=list(on)+tail
    m=SentenceTransformer(modules=mods, device="cpu")
    m.prompts=dict(ref.prompts); m.max_seq_length=512
    recs=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
    t=time.time()
    v=m.encode([r["text"] for r in recs],batch_size=32,normalize_embeddings=True,
               convert_to_numpy=True,show_progress_bar=True,prompt_name="document")
    assert not np.isnan(v).any(), f"{key}: NaN"
    np.save(EMB/f"{key}.questions.npy",v.astype(np.float32))
    test=[r for r in recs if r["split"]=="Test"]
    qv=m.encode([r["text"] for r in test],batch_size=32,normalize_embeddings=True,
                convert_to_numpy=True,prompt_name="query")
    np.save(EMB/f"{key}.testq.npy",qv.astype(np.float32))
    np.save(EMB/f"{key}.handq.npy",m.encode(HQ,normalize_embeddings=True,convert_to_numpy=True,
                                            prompt_name="query").astype(np.float32))
    json.dump({"dim":768,"repo":"onnx-community/embeddinggemma-300m-ONNX","onnx_file":fn,
               "prompts":dict(ref.prompts),"doc_prompt_name":"document","query_prompt_name":"query",
               "note":"ONNX transformer + grafted Dense/Dense/Normalize from google repo"},
              open(EMB/f"{key}.info.json","w"),indent=2)
    b=np.load(EMB/"gemma-300m.questions.npy")
    print(f"[{key}] {v.shape} {time.time()-t:.0f}s  mean_cos_vs_bf16={float((b*v).sum(1).mean()):.4f}",flush=True)

for k in (sys.argv[1:] or VARIANTS): run(k, VARIANTS[k])
