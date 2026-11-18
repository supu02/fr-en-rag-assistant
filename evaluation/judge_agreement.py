"""How much can we trust the LLM judge? Compare it with a human on a sample.

Step 1 (writes a CSV with 30 random clean answers, the judge's grade hidden):
    python -m evaluation.judge_agreement sample
Step 2: open results/human_labels.csv and fill the column `human_correctness`
        with correct / partial / incorrect, reading the question, the reference
        and the answer (open the cited links if needed).
Step 3 (compares your labels with the judge's):
    python -m evaluation.judge_agreement score

Reports raw agreement and Cohen's kappa (agreement corrected for chance:
1 = perfect, 0 = no better than chance). Results go to results/judge_agreement.json.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter

from evaluation.retrieval_eval import load_jsonl
from rag_assistant.config import RESULTS_DIR

LABELS = ("correct", "partial", "incorrect")
CSV_PATH = RESULTS_DIR / "human_labels.csv"


def sample(n: int = 30, seed: int = 0) -> None:
    records = load_jsonl(RESULTS_DIR / "answers" / "clean.jsonl")
    picked = random.Random(seed).sample(records, min(n, len(records)))
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "question", "reference", "answer", "cited_urls", "human_correctness"])
        for r in picked:
            w.writerow([r["id"], r["question"], r["reference"], r["answer"],
                        " ".join(c["url"] for c in r["citations"]), ""])
    print(f"wrote {len(picked)} rows to {CSV_PATH}; fill the last column, then run `score`")


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score() -> None:
    judge = {r["id"]: r["judge"]["correctness"] for r in load_jsonl(RESULTS_DIR / "answers" / "clean.jsonl")}
    with CSV_PATH.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["human_correctness"].strip()]
    bad = [r["id"] for r in rows if r["human_correctness"].strip() not in LABELS]
    if bad:
        raise SystemExit(f"labels must be one of {LABELS}; check rows {bad}")
    human = [r["human_correctness"].strip() for r in rows]
    model = [judge[r["id"]] for r in rows]
    result = {
        "n": len(rows),
        "agreement": round(sum(h == m for h, m in zip(human, model, strict=True)) / len(rows), 3),
        "cohen_kappa": round(cohen_kappa(human, model), 3),
        "disagreements": [{"id": r["id"], "human": h, "judge": m}
                          for r, h, m in zip(rows, human, model, strict=True) if h != m],
    }
    (RESULTS_DIR / "judge_agreement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "sample"]()
