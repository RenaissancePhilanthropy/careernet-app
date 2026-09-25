import json, sys
from pathlib import Path
import numpy as np
from paths import CACHE, EMB

Q=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
A=[json.loads(l) for l in open(CACHE/"corpus"/"answers.jsonl",encoding="utf-8")]
qidx={r["id"]:i for i,r in enumerate(Q)}
TEST=[i for i,r in enumerate(Q) if r["split"]=="Test"]
TRAINQ=[i for i,r in enumerate(Q) if r["split"]=="Train"]
TRAINA=[i for i,r in enumerate(A) if r["split"]=="Train"]
K=10

def relfn(field):
    def f(q, d):
        a,b=set(q[field]), set(d[field])
        return 1.0 if (a and b and a&b) else 0.0
    return f

def dcg(rels): return sum(r/np.log2(i+2) for i,r in enumerate(rels))
def ndcg(rels, nrel):
    ideal=dcg([1.0]*min(nrel,len(rels))) if nrel else 0.0
    return dcg(rels)/ideal if ideal else 0.0

def score(rank_mat, qrecs, drecs, field):
    """rank_mat: (nq, K) indices into drecs"""
    P,N,R=[],[],[]
    # precompute doc label sets
    dl=[set(d[field]) for d in drecs]
    for qi,q in enumerate(qrecs):
        ql=set(q[field])
        if not ql: continue
        hits=[1.0 if dl[j]&ql else 0.0 for j in rank_mat[qi]]
        nrel=sum(1 for s in dl if s&ql)
        if nrel==0: continue
        P.append(np.mean(hits)); N.append(ndcg(hits, nrel))
        rr=next((1/(i+1) for i,h in enumerate(hits) if h), 0.0); R.append(rr)
    return dict(n=len(P), P_at_10=np.mean(P), nDCG_at_10=np.mean(N), MRR_at_10=np.mean(R))

def topk(sim, k=K):
    part=np.argpartition(-sim, k, axis=1)[:,:k]
    ordr=np.take_along_axis(sim, part, 1).argsort(axis=1)[:,::-1]
    return np.take_along_axis(part, ordr, 1)

def dense_ranks(key, corpus):
    qv=np.load(EMB/f"{key}.testq.npy")
    dv=np.load(EMB/f"{key}.{corpus}.npy")
    sub = TRAINQ if corpus=="questions" else TRAINA
    dv=dv[sub]
    sim=qv@dv.T
    return topk(sim), sub

def bm25_ranks(corpus):
    import bm25s, Stemmer
    st=Stemmer.Stemmer("english")
    recs = Q if corpus=="questions" else A
    sub = TRAINQ if corpus=="questions" else TRAINA
    docs=[recs[i]["text"] for i in sub]
    r=bm25s.BM25(); r.index(bm25s.tokenize(docs, stopwords="en", stemmer=st, show_progress=False))
    qtexts=[Q[i]["text"] for i in TEST]
    res,_=r.retrieve(bm25s.tokenize(qtexts,stopwords="en",stemmer=st,show_progress=False), k=K, show_progress=False)
    return res, sub

MODELS=[p.stem.replace(".info","") for p in sorted(EMB.glob("*.info.json"))]
rows=[]
qrecs=[Q[i] for i in TEST]
for corpus in ("questions","answers"):
    drecs_all = Q if corpus=="questions" else A
    runs={}
    for key in MODELS:
        if not (EMB/f"{key}.{corpus}.npy").exists() or not (EMB/f"{key}.testq.npy").exists(): continue
        runs[key]=dense_ranks(key,corpus)
    runs["bm25"]=bm25_ranks(corpus)
    for key,(ranks,sub) in runs.items():
        drecs=[drecs_all[i] for i in sub]
        for field in ("soc_detailed","scenario"):
            s=score(ranks, qrecs, drecs, field)
            rows.append(dict(corpus=corpus, model=key, label=field, **s))

import csv
w=csv.DictWriter(open(CACHE/"eval_results.csv","w",newline=""), fieldnames=list(rows[0]))
w.writeheader(); [w.writerow(r) for r in rows]

for corpus in ("questions","answers"):
    for field in ("soc_detailed","scenario"):
        sel=[r for r in rows if r["corpus"]==corpus and r["label"]==field]
        sel.sort(key=lambda r:-r["nDCG_at_10"])
        print(f"\n=== query: Test question  ->  corpus: Train {corpus}   |  relevance: shared {field}  (n={sel[0]['n']}) ===")
        print(f"{'model':<14}{'P@10':>8}{'nDCG@10':>10}{'MRR@10':>9}")
        for r in sel: print(f"{r['model']:<14}{r['P_at_10']:>8.3f}{r['nDCG_at_10']:>10.3f}{r['MRR_at_10']:>9.3f}")
