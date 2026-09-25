import json, sys, time, gc
from pathlib import Path
import numpy as np, torch
from paths import CACHE, EMB
Q=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
A=[json.loads(l) for l in open(CACHE/"corpus"/"answers.jsonl",encoding="utf-8")]
TEST=[i for i,r in enumerate(Q) if r["split"]=="Test"]
BASE="gemma-300m"; DEPTH=50; K=10
NQ=int(sys.argv[1]) if len(sys.argv)>1 else 600
rng=np.random.default_rng(0)

def dcg(r): return sum(x/np.log2(i+2) for i,x in enumerate(r))
def ndcg(h,n):
    ideal=dcg([1.0]*min(n,len(h))); return dcg(h)/ideal if ideal else 0.0
def score(ranks,qrecs,drecs,field):
    dl=[set(d[field]) for d in drecs]; N=[]
    for qi,q in enumerate(qrecs):
        ql=set(q[field])
        if not ql: continue
        nrel=sum(1 for s in dl if s&ql)
        if not nrel: continue
        N.append(ndcg([1.0 if dl[j]&ql else 0.0 for j in ranks[qi][:K]],nrel))
    return float(np.mean(N)), len(N)

MAXLEN=512
def ce_scores(repo, pairs, bs):
    from sentence_transformers import CrossEncoder
    ce=CrossEncoder(repo, device="cuda", max_length=MAXLEN,
                    model_kwargs={"dtype":torch.float16}, trust_remote_code=True)
    s=ce.predict(pairs, batch_size=bs, show_progress_bar=True, convert_to_numpy=True)
    del ce; gc.collect(); torch.cuda.empty_cache()
    return np.asarray(s,dtype=np.float32).reshape(-1)

def qwen3_scores(repo, pairs, bs):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok=AutoTokenizer.from_pretrained(repo, padding_side="left")
    mdl=AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16).cuda().eval()
    yes=tok.convert_tokens_to_ids("yes"); no=tok.convert_tokens_to_ids("no")
    PRE=("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
         "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
         "<|im_end|>\n<|im_start|>user\n")
    POST="<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INS="Given a career-advice search query, retrieve relevant passages that answer the query"
    out=[]
    for i in range(0,len(pairs),bs):
        chunk=[f"{PRE}<Instruct>: {INS}\n<Query>: {q}\n<Document>: {d}{POST}" for q,d in pairs[i:i+bs]]
        enc=tok(chunk,return_tensors="pt",padding=True,truncation=True,max_length=MAXLEN).to("cuda")
        with torch.no_grad(): lg=mdl(**enc).logits[:,-1,:]
        st=torch.stack([lg[:,no],lg[:,yes]],dim=1).float().log_softmax(-1)[:,1]
        out.append(st.cpu().numpy())
        if i % (bs*40)==0: print(f"  qwen3-rr {i}/{len(pairs)}",flush=True)
    del mdl; gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(out).astype(np.float32)

RERANKERS={
 "bge-reranker-v2-m3": ("BAAI/bge-reranker-v2-m3", ce_scores, 64),
 "mxbai-rerank-xsmall": ("mixedbread-ai/mxbai-rerank-xsmall-v1", ce_scores, 128),
 "qwen3-reranker-0.6b": ("Qwen/Qwen3-Reranker-0.6B", qwen3_scores, 32),
}

results=[]
for corpus in ("questions","answers"):
    recs = Q if corpus=="questions" else A
    sub=[i for i,r in enumerate(recs) if r["split"]=="Train"]
    drecs=[recs[i] for i in sub]
    dv=np.load(EMB/f"{BASE}.{corpus}.npy")[sub]; qv=np.load(EMB/f"{BASE}.testq.npy")
    pick=rng.choice(len(TEST), size=min(NQ,len(TEST)), replace=False)
    qrecs=[Q[TEST[i]] for i in pick]; qv=qv[pick]
    sim=qv@dv.T
    cand=np.argsort(-sim,axis=1)[:,:DEPTH]
    base,_=score(cand,qrecs,drecs,"soc_detailed")
    pairs=[(qrecs[i]["text"], drecs[j]["text"]) for i in range(len(qrecs)) for j in cand[i]]
    print(f"\n### {corpus}: {len(qrecs)} queries x top-{DEPTH} = {len(pairs)} pairs",flush=True)
    for name,(repo,fn,bs) in RERANKERS.items():
        try:
            t=time.time(); s=fn(repo,pairs,bs).reshape(len(qrecs),DEPTH)
            el=time.time()-t
            order=np.argsort(-s,axis=1)
            newr=np.take_along_axis(cand,order,axis=1)
            for field in ("soc_detailed","scenario"):
                b,_=score(cand,qrecs,drecs,field); r,n=score(newr,qrecs,drecs,field)
                results.append(dict(corpus=corpus,reranker=name,label=field,base=b,reranked=r,
                                    delta=r-b,n=n,secs=round(el,1),
                                    ms_per_query=round(1000*el/len(qrecs),1)))
                print(f"  {name:22} {field:14} {b:.3f} -> {r:.3f}  ({r-b:+.3f})  {1000*el/len(qrecs):.0f} ms/q",flush=True)
        except Exception as e:
            import traceback; traceback.print_exc(); print(f"  {name} FAILED: {e}",flush=True)
json.dump(results, open(CACHE/"rerank_results.json","w"), indent=1)
print("\n=== summary (nDCG@10, base = gemma-300m dense top-50) ===")
print(f"{'corpus':<10}{'reranker':<22}{'label':<14}{'base':>7}{'rerank':>9}{'delta':>8}{'ms/q':>8}")
for r in results:
    print(f"{r['corpus']:<10}{r['reranker']:<22}{r['label']:<14}{r['base']:>7.3f}{r['reranked']:>9.3f}{r['delta']:>+8.3f}{r['ms_per_query']:>8.0f}")
