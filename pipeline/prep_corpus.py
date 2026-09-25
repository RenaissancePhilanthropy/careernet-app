import csv, json, re, html, sys, os
from pathlib import Path
csv.field_size_limit(10**9)

from paths import SRC
from paths import CORPUS as OUT
FILES = {"general":"general_public_v1.1.csv",
         "health":"health_public_v1.1.csv",
         "technology":"technology_public_v1.1.csv"}

SOC_FLAGS = {"No career mentioned in question", "Career in question not aligned with a label"}
GOAL_COLS = ["explore_options","take_action","understanding_purpose","validation_support",
             "find_resources","navigate_constraints","compare_options","unclear_goal"]

tag_re = re.compile(r"<[^>]+>")
ws_re  = re.compile(r"[ \t]+")

def clean(t):
    if not t: return ""
    t = t.replace("<br>","\n").replace("<br/>","\n").replace("</p>","\n")
    t = tag_re.sub(" ", t)
    t = html.unescape(t)
    t = ws_re.sub(" ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def split_ml(v):
    return [x.strip() for x in (v or "").split(";") if x.strip()]

def soc_sets(v):
    """returns (detailed_codes, all_codes, flags)"""
    det, allc, flags = set(), set(), set()
    for e in split_ml(v):
        if e in SOC_FLAGS:
            flags.add(e); continue
        m = re.match(r"(\d{2}-\d{4})\s+(.*)", e)
        if not m: continue
        code, name = m.group(1), m.group(2)
        allc.add(code)
        # detailed = does not end in 000 (group/broad/minor levels end 0000/X000/XX0)
        if not code.endswith("0"):
            det.add(code)
    return det, allc, flags

answers, questions = [], {}
for domain, fn in FILES.items():
    with open(SRC/fn, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            det, allc, flags = soc_sets(row["soc_code"])
            goals = sorted({row[c].strip() for c in GOAL_COLS if row[c].strip()})
            goalcols = sorted({c for c in GOAL_COLS if row[c].strip()})
            scen = split_ml(row["ScenarioLabels"])
            qid = f"{domain}:{row['question_id']}"
            qtitle, qbody = clean(row["question_title"]), clean(row["question_body"])
            qtext = (qtitle + "\n" + qbody).strip()
            abody = clean(row["answer_body"])
            rec = dict(
                id=f"{domain}:a{row['answer_id']}", qid=qid, domain=domain,
                text=abody, split=row["Split"],
                soc_detailed=sorted(det), soc_all=sorted(allc), soc_flags=sorted(flags),
                scenario=scen, goals=goals, goal_cols=goalcols,
                reasoning=split_ml(row["reasoning_label"]),
                correctness=row["correctness"], completeness=row["completeness"],
                coherency=row["coherency"],
                question_title=qtitle, tags=row["question_tagnames"],
            )
            if abody: answers.append(rec)
            if qid not in questions and qtext:
                questions[qid] = dict(
                    id=qid, domain=domain, text=qtext, title=qtitle, split=row["Split"],
                    soc_detailed=sorted(det), soc_all=sorted(allc), soc_flags=sorted(flags),
                    scenario=scen, goals=goals, goal_cols=goalcols,
                    tags=row["question_tagnames"],
                )

def dump(name, recs):
    p = OUT/f"{name}.jsonl"
    with open(p,"w",encoding="utf-8") as fh:
        for r in recs: fh.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{name}: {len(recs)} -> {p}")

qs = list(questions.values())
dump("questions", qs)
dump("answers", answers)

import statistics as st
for nm, recs in (("questions",qs),("answers",answers)):
    wl = [len(r["text"].split()) for r in recs]
    tr = sum(1 for r in recs if r["split"]=="Train"); te=len(recs)-tr
    withsoc = sum(1 for r in recs if r["soc_detailed"])
    print(f"  {nm}: train={tr} test={te} median_words={st.median(wl):.0f} p90={sorted(wl)[int(.9*len(wl))]} with_detailed_soc={withsoc}")
