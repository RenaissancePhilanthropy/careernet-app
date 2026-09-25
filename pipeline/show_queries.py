import json, sys, textwrap
from pathlib import Path
import numpy as np
from paths import CACHE, EMB
Q=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
A=[json.loads(l) for l in open(CACHE/"corpus"/"answers.jsonl",encoding="utf-8")]
HQ=json.load(open(CACHE/"hand_queries.json"))
CORPUS=sys.argv[1] if len(sys.argv)>1 else "questions"
TOP=int(sys.argv[2]) if len(sys.argv)>2 else 3
ONLY=[int(x) for x in sys.argv[3].split(",")] if len(sys.argv)>3 else range(len(HQ))
recs = Q if CORPUS=="questions" else A
MODELS=sorted(p.stem.replace(".info","") for p in EMB.glob("*.info.json")
              if (EMB/f"{p.stem.replace('.info','')}.{CORPUS}.npy").exists()
              and (EMB/f"{p.stem.replace('.info','')}.handq.npy").exists())

ranks={}
for k in MODELS:
    sim=np.load(EMB/f"{k}.handq.npy")@np.load(EMB/f"{k}.{CORPUS}.npy").T
    ranks[k]=np.argsort(-sim,axis=1)[:,:TOP]
import bm25s, Stemmer
st=Stemmer.Stemmer("english"); r=bm25s.BM25()
r.index(bm25s.tokenize([x["text"] for x in recs],stopwords="en",stemmer=st,show_progress=False))
res,_=r.retrieve(bm25s.tokenize(HQ,stopwords="en",stemmer=st,show_progress=False),k=TOP,show_progress=False)
ranks["bm25"]=res

def brief(i):
    d=recs[i]
    t = d["text"] if CORPUS=="questions" else f"[Q: {d['question_title']}] {d['text']}"
    t=" ".join(t.split())[:150]
    soc="; ".join(d["soc_detailed"]) or ("FLAG:"+",".join(d["soc_flags"]) if d["soc_flags"] else "-")
    return t, f"{d['domain']} | soc:{soc} | scen:{'; '.join(d['scenario'])[:60]}"

for qi in ONLY:
    print("\n"+"="*100); print(f"QUERY {qi}: {HQ[qi]}"); print("="*100)
    for k in list(ranks):
        print(f"\n--- {k} ---")
        for rank,i in enumerate(ranks[k][qi],1):
            t,meta=brief(int(i))
            print(f" {rank}. {t}")
            print(f"    {meta}")
