import json, sys, time, gc
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from paths import CACHE, EMB

TEST_QUERIES = [
 "How do I become a registered nurse without a bachelor's degree?",
 "Is a computer science degree worth it if I already know how to code?",
 "I'm a high school senior and have no idea what career to pick.",
 "What does a day in the life of a physical therapist look like?",
 "How much do software engineers make straight out of college?",
 "Can I switch from retail into a tech job at 30?",
 "What certifications should I get to work in cybersecurity?",
 "Do I need to go to medical school to work in healthcare?",
 "How do I pay for college if my family can't afford it?",
 "What's the difference between a data scientist and a data analyst?",
 "I'm interested in helping people but I don't like the sight of blood.",
 "How important are internships for getting a first job?",
]

def run(key):
    info=json.load(open(EMB/f"{key}.info.json"))
    qs=[json.loads(l) for l in open(CACHE/"corpus"/"questions.jsonl",encoding="utf-8")]
    test=[r for r in qs if r["split"]=="Test"]
    m=SentenceTransformer(info["repo"],device="cuda",model_kwargs={"dtype":getattr(torch,info.get("dtype","float16"))})
    m.max_seq_length=512
    kw={"prompt_name":info["query_prompt_name"]} if info["query_prompt_name"] else {}
    v=m.encode([r["text"] for r in test],batch_size=32,normalize_embeddings=True,
               convert_to_numpy=True,show_progress_bar=False,**kw)
    assert not np.isnan(v).any(), f"{key}: NaN in test queries"
    np.save(EMB/f"{key}.testq.npy", v.astype(np.float32))
    h=m.encode(TEST_QUERIES,batch_size=8,normalize_embeddings=True,convert_to_numpy=True,**kw)
    np.save(EMB/f"{key}.handq.npy", h.astype(np.float32))
    print(f"[{key}] testq {v.shape} handq {h.shape} qprompt={info['query_prompt_name']}",flush=True)
    del m; gc.collect(); torch.cuda.empty_cache()

json.dump(TEST_QUERIES, open(CACHE/"hand_queries.json","w"), indent=1)
for k in sys.argv[1:]: run(k)
