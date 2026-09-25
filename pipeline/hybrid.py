import json, sys
from pathlib import Path
import numpy as np
from paths import CACHE, EMB
Q=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
A=[json.loads(l) for l in open(CACHE/"corpus"/"answers.jsonl",encoding="utf-8")]
TEST=[i for i,r in enumerate(Q) if r["split"]=="Test"]
K=10; POOL=1000
MODEL=sys.argv[1] if len(sys.argv)>1 else "gemma-300m"

def dcg(r): return sum(x/np.log2(i+2) for i,x in enumerate(r))
def ndcg(hits,nrel):
    ideal=dcg([1.0]*min(nrel,len(hits)))
    return dcg(hits)/ideal if ideal else 0.0

def score(ranks, qrecs, drecs, field):
    dl=[set(d[field]) for d in drecs]; N=[]
    for qi,q in enumerate(qrecs):
        ql=set(q[field])
        if not ql: continue
        nrel=sum(1 for s in dl if s&ql)
        if not nrel: continue
        N.append(ndcg([1.0 if dl[j]&ql else 0.0 for j in ranks[qi][:K]], nrel))
    return float(np.mean(N))

def mm(x):  # per-row min-max
    lo=x.min(1,keepdims=True); hi=x.max(1,keepdims=True)
    return (x-lo)/np.maximum(hi-lo,1e-9)

out=[]
for corpus in ("questions","answers"):
    recs = Q if corpus=="questions" else A
    sub=[i for i,r in enumerate(recs) if r["split"]=="Train"]
    drecs=[recs[i] for i in sub]; qrecs=[Q[i] for i in TEST]
    # dense
    dv=np.load(EMB/f"{MODEL}.{corpus}.npy")[sub]; qv=np.load(EMB/f"{MODEL}.testq.npy")
    dense=qv@dv.T
    # bm25 -> dense-shaped sparse scores
    import bm25s, Stemmer
    st=Stemmer.Stemmer("english"); r=bm25s.BM25()
    r.index(bm25s.tokenize([d["text"] for d in drecs],stopwords="en",stemmer=st,show_progress=False))
    idx,sc=r.retrieve(bm25s.tokenize([q["text"] for q in qrecs],stopwords="en",stemmer=st,show_progress=False),
                      k=min(POOL,len(drecs)),show_progress=False)
    lex=np.zeros(dense.shape,dtype=np.float32)          # BM25 scores are >=0; unretrieved = 0
    np.put_along_axis(lex,idx,np.maximum(sc,0).astype(np.float32),axis=1)
    # rank arrays for RRF
    def ranks_of(mat):
        o=np.argsort(-mat,axis=1)
        rk=np.empty_like(o); np.put_along_axis(rk,o,np.arange(mat.shape[1])[None,:],axis=1)
        return rk
    rd, rl = ranks_of(dense), ranks_of(lex)
    # normalise BM25 over the retrieved pool only, so unretrieved docs sit at 0 rather than
    # dragging the min down and flattening the signal
    hi=lex.max(1,keepdims=True); dn, ln = mm(dense), lex/np.maximum(hi,1e-9)
    for field in ("soc_detailed","scenario"):
        row={"corpus":corpus,"label":field}
        row["dense_only"]=score(np.argsort(-dense,1),qrecs,drecs,field)
        row["bm25_only"]=score(np.argsort(-lex,1),qrecs,drecs,field)
        for w in (0.5,0.8,0.85,0.9,0.95,0.98,1.0):
            row[f"w{w}"]=score(np.argsort(-(w*dn+(1-w)*ln),1),qrecs,drecs,field)
        rrf=1.0/(60+rd)+1.0/(60+rl)
        row["rrf"]=score(np.argsort(-rrf,1),qrecs,drecs,field)
        out.append(row)

json.dump(out,open(CACHE/f"hybrid_{MODEL}.json","w"),indent=1)
hdr=[k for k in out[0] if k not in ("corpus","label")]
print(f"model={MODEL}   nDCG@10   (w = weight on dense; 1-w on BM25)\n")
print(f"{'corpus':<10}{'label':<14}"+"".join(f"{h:>12}" for h in hdr))
for r in out:
    best=max(hdr,key=lambda h:r[h])
    print(f"{r['corpus']:<10}{r['label']:<14}"+"".join(f"{r[h]:>12.3f}" for h in hdr)+f"   best={best}")
