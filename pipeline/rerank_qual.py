import json, numpy as np, torch
from pathlib import Path
from sentence_transformers import CrossEncoder
from paths import CACHE, EMB
Q=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
HQ=json.load(open(CACHE/"hand_queries.json"))
DEPTH=50
dv=np.load(EMB/"gemma-300m.questions.npy"); qv=np.load(EMB/"gemma-300m.handq.npy")
cand=np.argsort(-(qv@dv.T),axis=1)[:,:DEPTH]
ce=CrossEncoder("BAAI/bge-reranker-v2-m3",device="cuda",max_length=512,model_kwargs={"dtype":torch.float16})
for qi in (0,5,10):
    pairs=[(HQ[qi],Q[j]["text"]) for j in cand[qi]]
    s=ce.predict(pairs,batch_size=64,show_progress_bar=False)
    new=cand[qi][np.argsort(-s)]
    print("\n"+"="*95); print("QUERY:",HQ[qi]); print("="*95)
    for lbl,ids in (("dense top-3 (gemma)",cand[qi][:3]),("reranked top-3 (bge-v2-m3)",new[:3])):
        print(f"\n  {lbl}")
        for r,j in enumerate(ids,1):
            d=Q[int(j)]
            print(f"   {r}. {' '.join(d['text'].split())[:130]}")
            print(f"      soc:{'; '.join(d['soc_detailed']) or ('FLAG '+','.join(d['soc_flags']))}")
