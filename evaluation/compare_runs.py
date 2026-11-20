"""Compare the answer evaluation before and after the fixes of 20 Nov 2026.

Before: results/before_fixes/ (first run, commit b12ef18). After: results/answers/.
The fixes came from reading the first run's wrong answers, on the same questions,
so the "after" numbers are optimistic.

Besides the rates, it counts paired changes: for each question answered in both
runs, did the judge's grade go up, down, or stay the same? With small samples this
says more than two rates side by side. Part of a change is run-to-run noise: the
model is not fully deterministic, even at temperature 0.

It also compares the main answer model with each other model run into
results/model_comparison/<name>/ (answer_eval --out), on the same questions.

  python -m evaluation.compare_runs   # writes results/before_after.json and .../vs_main.json
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from evaluation.answer_eval import RAW_DIR, VARIANT_NAMES, summarise, token_totals
from evaluation.retrieval_eval import load_jsonl
from rag_assistant.config import ANSWER_MODEL, RESULTS_DIR

BEFORE_DIR = RESULTS_DIR / "before_fixes" / "answers"
MODELS_DIR = RESULTS_DIR / "model_comparison"  # one folder per other answer model, made with answer_eval --out
METRICS = ("correct", "faithful", "false_abstention", "reference_numbers_in_answer", "out_of_scope_abstained",
           "injection_success", "injection_success_direct", "injection_success_system", "citation_hit", "no_citation",
           "date_questions_tool_used", "date_questions_correct")


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


def compare(before_dir: Path = BEFORE_DIR, after_dir: Path = RAW_DIR, with_tokens: bool = False,
            models: tuple[str, str] = (ANSWER_MODEL, ANSWER_MODEL)) -> dict:
    """models: the answer model of each side, for the cost estimate."""
    out = {}
    for variant in VARIANT_NAMES:
        before_path, after_path = before_dir / f"{variant}.jsonl", after_dir / f"{variant}.jsonl"
        if not (before_path.exists() and after_path.exists()):
            continue
        before, after = load_jsonl(before_path), load_jsonl(after_path)
        ids = {r["id"] for r in after}
        before = [r for r in before if r["id"] in ids]  # same questions on both sides
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
        if with_tokens:
            out[variant]["tokens"] = {"before": token_totals(before, models[0]), "after": token_totals(after, models[1])}
    return out


def save(data: dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"saved {path}")


def compare_models() -> dict:
    """The main answer model ("before") against each model in results/model_comparison/ ("after")."""
    out = {}
    for folder in sorted(p for p in MODELS_DIR.glob("*") if (p / "answer_eval.json").exists()):
        model = json.loads((folder / "answer_eval.json").read_text(encoding="utf-8"))["meta"]["answer_model"]
        out[folder.name] = {"answer_model": model, **compare(RAW_DIR, folder / "answers", with_tokens=True,
                                                             models=(ANSWER_MODEL, model))}
        save(out[folder.name], folder / "vs_main.json")
    return out


if __name__ == "__main__":
    results = {"fixes (before_fixes -> current)": compare()}
    save(results["fixes (before_fixes -> current)"], RESULTS_DIR / "before_after.json")
    for res in compare_models().values():
        results[f"main model -> {res.pop('answer_model')}"] = res
    for title, variants in results.items():
        print(title)
        for variant, langs in variants.items():
            for lang in ("fr", "en", "all"):
                b, a, p = langs[lang]["before"]["correct"], langs[lang]["after"]["correct"], langs[lang]["paired"]
                print(f"  {variant:10s} {lang:3s} correct {b['k']}/{b['n']} -> {a['k']}/{a['n']}   paired: "
                      f"{p['better']} better, {p['worse']} worse, {p['same']} same, sign test p={p['sign_test_p']}")
