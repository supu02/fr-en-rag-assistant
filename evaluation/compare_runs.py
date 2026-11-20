"""Compare the answer evaluation before and after the fixes of 20 Nov 2026.

Before: results/before_fixes/ (first run, commit b12ef18). After: results/answers/.
The fixes came from reading the first run's wrong answers, on the same questions,
so the "after" numbers are optimistic.

Besides the rates, it counts paired changes: for each question answered in both
runs, did the judge's grade go up, down, or stay the same? With small samples this
says more than two rates side by side. Part of a change is run-to-run noise: the
model is not fully deterministic, even at temperature 0.

  python -m evaluation.compare_runs   # writes results/before_after.json
"""

from __future__ import annotations

import json
from math import comb

from evaluation.answer_eval import RAW_DIR, VARIANT_NAMES, summarise
from evaluation.retrieval_eval import load_jsonl
from rag_assistant.config import RESULTS_DIR

BEFORE_DIR = RESULTS_DIR / "before_fixes" / "answers"
METRICS = ("correct", "faithful", "false_abstention", "reference_numbers_in_answer", "out_of_scope_abstained",
           "injection_success", "injection_success_direct", "injection_success_system")


def paired(before: list[dict], after: list[dict]) -> dict:
    old = {r["id"]: r["judge"]["score"] for r in before if r["type"] != "unanswerable"}
    new = {r["id"]: r["judge"]["score"] for r in after if r["type"] != "unanswerable"}
    both = sorted(old.keys() & new.keys())
    better, worse = sum(new[i] > old[i] for i in both), sum(new[i] < old[i] for i in both)
    return {"n": len(both), "better": better, "worse": worse, "same": len(both) - better - worse,
            "sign_test_p": sign_test(better, worse)}


def sign_test(better: int, worse: int) -> float | None:
    """Two-sided sign test: how likely is a split at least this uneven if the fixes changed
    nothing and each changed grade went up or down by chance? Ties are left out."""
    n = better + worse
    if n == 0:
        return None
    tail = sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 3)


def compare() -> dict:
    out = {}
    for variant in VARIANT_NAMES:
        before_path, after_path = BEFORE_DIR / f"{variant}.jsonl", RAW_DIR / f"{variant}.jsonl"
        if not (before_path.exists() and after_path.exists()):
            continue
        before, after = load_jsonl(before_path), load_jsonl(after_path)
        out[variant] = {}
        for lang in ("fr", "en", "all"):
            b = [r for r in before if lang in ("all", r["lang"])]
            a = [r for r in after if lang in ("all", r["lang"])]
            sb, sa = summarise(b), summarise(a)
            out[variant][lang] = {
                "before": {m: sb[m] for m in METRICS if m in sb} | {"correctness_mean": sb["correctness_mean"]},
                "after": {m: sa[m] for m in METRICS if m in sa} | {"correctness_mean": sa["correctness_mean"]},
                "paired": paired(b, a),
            }
    path = RESULTS_DIR / "before_after.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}")
    return out


if __name__ == "__main__":
    for variant, langs in compare().items():
        for lang, v in langs.items():
            b, a, p = v["before"]["correct"], v["after"]["correct"], v["paired"]
            print(f"{variant:10s} {lang:3s} correct {b['k']}/{b['n']} -> {a['k']}/{a['n']}   "
                  f"paired: {p['better']} better, {p['worse']} worse, {p['same']} same, sign test p={p['sign_test_p']}")
