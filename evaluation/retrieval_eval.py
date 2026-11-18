"""Retrieval evaluation: does the right page come back in the top k results?

No LLM is involved, so this is free, fast and deterministic. CI runs it on
every push and fails if quality drops below the thresholds in thresholds.json.

Metrics (computed on answerable questions only):
- hit@k:  share of questions where at least one of the top k chunks comes
          from a gold page.
- MRR@10: mean of 1/rank of the first gold chunk (0 if not in the top 10).
Gold pages were chosen by hand. Another page may also contain the answer, so
these numbers are a lower bound.

Run:  python -m evaluation.retrieval_eval
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path

from evaluation.perturb import VARIANTS
from rag_assistant.config import EMBED_MODEL, RESULTS_DIR
from rag_assistant.retrieve import METHODS, get_retriever

HERE = Path(__file__).parent
DEFAULT_METHOD = "auto"  # the method the assistant uses
KS = (1, 3, 5)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def first_gold_rank(results: list[dict], gold: list[str]) -> int | None:
    for rank, chunk in enumerate(results, start=1):
        if chunk["page_id"] in gold:
            return rank
    return None


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (Wilson score interval).
    With n = 45, the interval is roughly +/- 10 to 15 points: small samples
    cannot separate systems that differ by a few points."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 3), round(centre + half, 3))


def summarise(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {"n": n}
    for k in KS:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        out[f"hit@{k}"] = round(hits / n, 3)
    hits5 = sum(1 for r in ranks if r is not None and r <= 5)
    out["hit@5_ci95"] = wilson(hits5, n)
    out["mrr@10"] = round(sum(1 / r for r in ranks if r is not None) / n, 3)
    return out


def by_language(questions: list[dict], ranks: list[int | None]) -> dict:
    out = {"all": summarise(ranks)}
    for lang in ("fr", "en"):
        out[lang] = summarise([r for q, r in zip(questions, ranks, strict=True) if q["lang"] == lang])
    return out


def paired_language_check(questions: list[dict], ranks: list[int | None], k: int = 5) -> dict:
    """Same topic asked in FR and EN: count topics where only one language hits.
    This paired view is fairer than comparing two averages."""
    hit = {(q["topic"], q["lang"]): (r is not None and r <= k) for q, r in zip(questions, ranks, strict=True)}
    topics = sorted({q["topic"] for q in questions})
    only_fr = sum(1 for t in topics if hit[(t, "fr")] and not hit[(t, "en")])
    only_en = sum(1 for t in topics if hit[(t, "en")] and not hit[(t, "fr")])
    return {"topics": len(topics), f"only_fr_hit@{k}": only_fr, f"only_en_hit@{k}": only_en}


def flips(questions: list[dict], before: list, after: list, k: int = 5) -> dict:
    """How many questions changed outcome (hit@k) after the perturbation, per language.
    Averages can hide changes that cancel out: 2 lost + 2 gained = no change on average."""
    out = {}
    for lang in ("fr", "en"):
        pairs = [(b is not None and b <= k, a is not None and a <= k)
                 for q, b, a in zip(questions, before, after, strict=True) if q["lang"] == lang]
        out[lang] = {"lost": sum(b and not a for b, a in pairs), "gained": sum(a and not b for b, a in pairs)}
    return out


def run_ranks(retriever, questions: list[dict], method: str, text_of=lambda q: q["question"]) -> list:
    return [first_gold_rank(retriever.search(text_of(q), k=10, method=method), q["gold_pages"])
            for q in questions]


def main() -> dict:
    questions = [q for q in load_jsonl(HERE / "questions.jsonl") if q["answerable"]]
    paraphrases = load_jsonl(HERE / "paraphrases.jsonl")
    retriever = get_retriever()

    report: dict = {
        "meta": {
            "date": date.today().isoformat(),
            "embed_model": EMBED_MODEL,
            "chunks": len(retriever.chunks),
            "questions": len(questions),
            "note": "answerable questions only; hit@k counts a gold page anywhere in the top k chunks",
        },
        "methods": {},
        "robustness": {},
    }

    # 1) Compare retrieval methods on clean questions.
    for method in METHODS:
        ranks = run_ranks(retriever, questions, method)
        report["methods"][method] = by_language(questions, ranks)
        report["methods"][method]["paired"] = paired_language_check(questions, ranks)
        m = report["methods"][method]
        print(f"{method:14s} hit@5 all={m['all']['hit@5']:.3f} fr={m['fr']['hit@5']:.3f} en={m['en']['hit@5']:.3f}")

    # 2) Robustness of the default method to question perturbations.
    clean_ranks = None
    for name, make in VARIANTS.items():
        ranks = run_ranks(retriever, questions, DEFAULT_METHOD, text_of=make)
        report["robustness"][name] = by_language(questions, ranks)
        if name == "clean":
            clean_ranks = ranks
        else:
            report["robustness"][name]["flips"] = flips(questions, clean_ranks, ranks)
    ranks = run_ranks(retriever, paraphrases, DEFAULT_METHOD)
    report["robustness"]["paraphrase"] = by_language(paraphrases, ranks)
    report["meta"]["robustness_method"] = DEFAULT_METHOD
    for name, res in report["robustness"].items():
        print(f"{DEFAULT_METHOD}/{name:11s} hit@5 fr={res['fr']['hit@5']:.3f} en={res['en']['hit@5']:.3f}"
              f"  flips={res.get('flips', '')}")

    retriever.close()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "retrieval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out}")
    return report


def check_thresholds(report: dict) -> int:
    """Exit code 1 if any metric is below its threshold (used by CI)."""
    thresholds = json.loads((HERE / "thresholds.json").read_text())["retrieval"]
    failures = []
    for key, minimum in thresholds.items():
        method, subset, metric = key.split("/")
        # keys look like "dense_diverse/en/hit@5" or "robustness:typos/fr/hit@5"
        if method.startswith("robustness:"):
            value = report["robustness"][method.removeprefix("robustness:")][subset][metric]
        else:
            value = report["methods"][method][subset][metric]
        status = "ok " if value >= minimum else "FAIL"
        print(f"  {status} {key} = {value:.3f} (min {minimum})")
        if value < minimum:
            failures.append(key)
    return 1 if failures else 0


if __name__ == "__main__":
    rep = main()
    if "--check" in sys.argv:
        sys.exit(check_thresholds(rep))
