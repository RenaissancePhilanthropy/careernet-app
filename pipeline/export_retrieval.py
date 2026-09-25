"""Build the data files for the CareerNet search page.

Outputs into docs/search/data/:
  {corpus}.meta.json   text + labels + dates + cross-links, one entry per row
  {corpus}.vec.bin     int8 embeddings, row-major, n x 768
  dense_head.bin       float32 768x768 projection composed from the model's two Dense layers
  manifest.json        shapes, scales, label vocabularies, SOC->BLS join

Row order matches the embedding arrays exactly; nothing here may reorder anything.
"""
import json, re, csv
from pathlib import Path
import numpy as np

from paths import CACHE, SRC, STATIC
OUT = STATIC / "search" / "data"
OUT.mkdir(parents=True, exist_ok=True)
DASH = json.load(open(STATIC / "data.json", encoding="utf-8"))

GOAL_LABEL = {
    "explore_options": "Explore options", "take_action": "Take action",
    "understanding_purpose": "Understand purpose", "validation_support": "Validation / support",
    "find_resources": "Find resources", "navigate_constraints": "Navigate constraints",
    "compare_options": "Compare options", "unclear_goal": "Unclear goal",
}
GOAL_HELP = {
    "explore_options": "The asker is surveying possibilities rather than committing to one.",
    "take_action": "The asker wants concrete next steps they can carry out.",
    "understanding_purpose": "The asker is asking what a role or path is actually for.",
    "validation_support": "The asker is seeking reassurance about a choice or situation.",
    "find_resources": "The asker wants pointers to material, programmes or people.",
    "navigate_constraints": "The asker is working around a limit: money, location, grades, time.",
    "compare_options": "The asker is weighing two or more specific alternatives.",
    "unclear_goal": "No single goal could be identified from the question.",
}
QUALITY_HELP = {
    "correctness": "Is the advice factually right? Rated 1-4 by a human annotator.",
    "completeness": "Does it cover what the question actually asked? Rated 1-4.",
    "coherency": "Does it hold together and read clearly? Rated 1-4.",
}

csv.field_size_limit(10**9)
FLAGS = {"No career mentioned in question", "Career in question not aligned with a label"}
major_title, detail_title = {}, {}
for f in sorted(SRC.glob("*.csv")):
    for row in csv.DictReader(open(f, encoding="utf-8")):
        for e in row["soc_code"].split(";"):
            e = e.strip()
            if not e or e in FLAGS:
                continue
            m = re.match(r"(\d{2}-\d{4})\s+(.*)", e)
            if not m:
                continue
            code, title = m.group(1), m.group(2).strip()
            if code.endswith("-0000"):
                major_title[code[:2]] = title
            elif not code.endswith("0"):
                detail_title[code] = title


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.replace("\u2019", "'").lower()).strip()


lmi_by_norm = {norm(c["soc_title"]): c for c in DASH["charts"]["soc_coverage"]}
soc_lmi = {}
for code, title in detail_title.items():
    hit = lmi_by_norm.get(norm(title))
    if hit:
        soc_lmi[code] = {"title": title, "wage": hit["mean_wage"],
                         "employment": hit["total_employment"],
                         "questions": hit["question_count"], "ratio": hit["discussion_ratio"]}
print(f"SOC->BLS join: {len(soc_lmi)}/{len(detail_title)} detailed codes matched")


def q8(x):
    scale = float(np.abs(x).max()) / 127.0
    return np.round(x / scale).astype(np.int8), scale


manifest = {"corpora": {}, "soc_major": major_title, "soc_detail": detail_title,
            "soc_lmi": soc_lmi, "goal_labels": GOAL_LABEL, "goal_help": GOAL_HELP,
            "quality_help": QUALITY_HELP,
            "model": {"repo": "onnx-community/embeddinggemma-300m-ONNX", "dtype": "q8",
                      "query_prompt": "task: search result | query: ", "dim": 768}}

for corpus in ("questions", "answers"):
    recs = [json.loads(l) for l in open(CACHE / "corpus" / f"{corpus}.jsonl", encoding="utf-8")]
    V = np.load(CACHE / "emb" / f"gemma-300m.{corpus}.npy")
    assert len(recs) == V.shape[0], f"{corpus}: {len(recs)} records vs {V.shape[0]} vectors"

    meta = []
    for r in recs:
        e = {"t": r["text"], "d": r["domain"],
             "soc": r["soc_detailed"],
             "maj": sorted({c[:2] for c in r["soc_all"] if c.endswith("-0000")}),
             "sc": r["scenario"], "g": r["goal_cols"], "fl": r["soc_flags"],
             "y": r.get("year"), "w": r.get("when")}
        if corpus == "answers":
            e["rl"] = r["reasoning"]
            e["qt"] = r["question_title"]
            e["qi"] = r.get("q_idx")
            e["ql"] = [int(r[k].strip()[0]) if r.get(k, "").strip()[:1].isdigit() else None
                       for k in ("correctness", "completeness", "coherency")]
        else:
            e["ai"] = r.get("answer_idx", [])
            # the corpus text is title + newline + body; keep the title separate so the page
            # can set it as a heading instead of running both together in one block
            e["ti"] = r.get("title", "")
        meta.append(e)

    Vq, scale = q8(V)
    (OUT / f"{corpus}.vec.bin").write_bytes(Vq.tobytes())
    json.dump(meta, open(OUT / f"{corpus}.meta.json", "w", encoding="utf-8"),
              ensure_ascii=False, separators=(",", ":"))
    years = [m["y"] for m in meta if m["y"]]
    manifest["corpora"][corpus] = {"n": len(recs), "dim": int(V.shape[1]), "scale": scale,
                                   "year_min": min(years), "year_max": max(years)}
    print(f"  {corpus}: n={len(recs)} vec={Vq.nbytes/1e6:.1f}MB "
          f"meta={(OUT/f'{corpus}.meta.json').stat().st_size/1e6:.1f}MB "
          f"years {min(years)}-{max(years)}")

head = np.load(CACHE / "emb" / "gemma_dense_head.npy").astype(np.float32)
(OUT / "dense_head.bin").write_bytes(np.ascontiguousarray(head).tobytes())
manifest["dense_head"] = {"shape": list(head.shape)}
json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"  dense_head {head.shape} {head.nbytes/1e6:.1f}MB")

for stale in OUT.glob("*.umap.bin"):
    stale.unlink()
    print(f"  removed stale {stale.name}")
