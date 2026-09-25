"""Add dates and question<->answer links to the existing corpus files.

Reads the corpus JSONL produced by prep_corpus.py and joins extra fields back from the
release CSVs by id. Deliberately does NOT rebuild the corpus: the embedding arrays are
positional, so any change to row order would silently misalign every vector. Row order and
text are asserted unchanged at the end.
"""
import csv, json, re, sys
from datetime import datetime
from pathlib import Path

csv.field_size_limit(10**9)
from paths import CACHE, SRC
FILES = {"general": "general_public_v1.1.csv",
         "health": "health_public_v1.1.csv",
         "technology": "technology_public_v1.1.csv"}


def parse_date(s):
    """'October 18, 2011, 5:39 PM' -> (year, 'Oct 2011'). None if unparseable."""
    s = (s or "").strip()
    if not s:
        return None, None
    for fmt in ("%B %d, %Y, %I:%M %p", "%B %d, %Y", "%b %d, %Y, %I:%M %p"):
        try:
            d = datetime.strptime(s, fmt)
            return d.year, d.strftime("%b %Y")
        except ValueError:
            continue
    m = re.search(r"\b(19|20)\d{2}\b", s)
    return (int(m.group(0)), m.group(0)) if m else (None, None)


# ---- harvest the extra fields, keyed by the same ids prep_corpus.py assigned ----------
q_extra, a_extra = {}, {}
for domain, fname in FILES.items():
    with open(SRC / fname, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            qid = f"{domain}:{row['question_id']}"
            aid = f"{domain}:a{row['answer_id']}"
            qy, qlabel = parse_date(row["question_added_at"])
            ay, alabel = parse_date(row["answer_added_at"])
            if qid not in q_extra:
                q_extra[qid] = {"year": qy, "when": qlabel,
                                "views": row["question_views"], "score": row["question_score"],
                                "tags": row["question_tagnames"]}
            a_extra[aid] = {"qid": qid, "year": ay, "when": alabel, "score": row["answer_score"]}


def load(name):
    return [json.loads(l) for l in open(CACHE / "corpus" / f"{name}.jsonl", encoding="utf-8")]


questions, answers = load("questions"), load("answers")
before_q = [r["text"] for r in questions]
before_a = [r["text"] for r in answers]

# question id -> its position, so answers can point at an index the front end can use directly
qpos = {r["id"]: i for i, r in enumerate(questions)}

missing_q = missing_a = orphan = 0
for r in questions:
    e = q_extra.get(r["id"])
    if not e:
        missing_q += 1
        continue
    r.update(year=e["year"], when=e["when"], views=e["views"], qscore=e["score"], tags=e["tags"])
    r["answer_idx"] = []

for i, r in enumerate(answers):
    e = a_extra.get(r["id"])
    if not e:
        missing_a += 1
        continue
    r.update(year=e["year"], when=e["when"], ascore=e["score"], qid=e["qid"])
    j = qpos.get(e["qid"])
    if j is None:
        orphan += 1
    else:
        r["q_idx"] = j
        questions[j]["answer_idx"].append(i)

assert [r["text"] for r in questions] == before_q, "question order or text changed"
assert [r["text"] for r in answers] == before_a, "answer order or text changed"

for name, recs in (("questions", questions), ("answers", answers)):
    with open(CACHE / "corpus" / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

n_ans = [len(r.get("answer_idx", [])) for r in questions]
years = [r.get("year") for r in questions if r.get("year")]
print(f"questions={len(questions)} answers={len(answers)}")
print(f"  unmatched: questions={missing_q} answers={missing_a} orphaned answers={orphan}")
print(f"  answers per question: min={min(n_ans)} median={sorted(n_ans)[len(n_ans)//2]} max={max(n_ans)}")
print(f"  questions with zero answers: {sum(1 for n in n_ans if n == 0)}")
print(f"  question years: {min(years)}-{max(years)}, missing on {len(questions)-len(years)}")
