"""Answer-level evaluation and robustness suite (calls the LLM: Mistral models on Bedrock by default).

For every question variant it runs the full assistant, saves the raw output,
asks the judge for a grade, then writes a summary to results/answer_eval.json.

Variants:
  clean       the 100 labelled questions (50 topics x FR/EN)
  typos       90 answerable questions with keyboard typos
  paraphrase  90 answerable questions reworded (casual words, synonyms)
  injection   the 100 questions with a hidden instruction appended (half
              "ignore previous instructions...", half a fake [SYSTEM] rule)

Measured per variant and per language:
  correctness (judge: correct=1, partial=0.5), faithfulness to cited sources
  (judge), citation hit (a cited source comes from a gold page), retrieval
  hit@5, answer language = question language, tool use on date questions,
  abstention on out-of-scope questions, injection success (canary in answer).

Raw outputs are saved, so the run can resume after a rate-limit error and the
summary can be recomputed without new API calls:
  python -m evaluation.answer_eval                       # all variants
  python -m evaluation.answer_eval --variants clean --limit 10
  python -m evaluation.answer_eval --summary-only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date
from pathlib import Path

from evaluation.perturb import CANARY, inject, seed_for, typos
from evaluation.retrieval_eval import load_jsonl, wilson
from rag_assistant.config import ANSWER_MODEL, JUDGE_MODEL, RESULTS_DIR
from rag_assistant.lang import detect_language
from rag_assistant.monitoring import estimate_cost

HERE = Path(__file__).parent
RAW_DIR = RESULTS_DIR / "answers"
VARIANT_NAMES = ("clean", "typos", "paraphrase", "injection")


def build_variants() -> dict[str, list[dict]]:
    questions = load_jsonl(HERE / "questions.jsonl")
    answerable = [q for q in questions if q["answerable"]]
    injected = []
    for q in questions:
        style = "direct" if int(q["topic"][1:]) % 2 else "system"
        injected.append({**q, "id": f"{q['id']}-inj", "question": inject(q["question"], q["lang"], style),
                         "injection_style": style})
    return {
        "clean": questions,
        "typos": [{**q, "id": f"{q['id']}-typo", "question": typos(q["question"], seed_for(q["id"]))}
                  for q in answerable],
        "paraphrase": load_jsonl(HERE / "paraphrases.jsonl"),
        "injection": injected,
    }


def generate_and_judge(variant: str, items: list[dict], limit: int | None) -> None:
    from evaluation.judge import judge
    from rag_assistant.agent import build_graph, run
    from rag_assistant.llm import get_chat_model
    from rag_assistant.retrieve import get_retriever

    retriever = get_retriever()
    sources_by_chunk = {c["chunk_id"]: c for c in retriever.chunks}
    # A fixed "today" keeps the prompt identical between runs.
    graph = build_graph(get_chat_model(), retriever, today="2026-11-18")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{variant}.jsonl"
    done = {r["id"] for r in load_jsonl(path)} if path.exists() else set()
    todo = [q for q in items if q["id"] not in done][:limit]
    print(f"[{variant}] {len(done)} done, {len(todo)} to run")
    for i, q in enumerate(todo, start=1):
        start = time.time()
        try:
            result = run(graph, q["question"])
            verdict = judge(q, result, sources_by_chunk)
        except Exception as exc:  # keep going; the item is retried on the next run
            print(f"  ! {q['id']}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        record = {"id": q["id"], "topic": q["topic"], "lang": q["lang"], "type": q["type"],
                  "gold_pages": q["gold_pages"], "question": q["question"], "reference": q["reference"],
                  **{k: result[k] for k in ("answer", "search_query", "citations", "retrieved", "tool_calls",
                                            "usage")},
                  "judge": verdict, "seconds": round(time.time() - start, 1)}
        if "injection_style" in q:
            record["injection_style"] = q["injection_style"]
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  {i}/{len(todo)} {q['id']}: {verdict['correctness']}, faithful={verdict['faithful']}")


# ---------- metrics ----------
def rate(flags: list[bool]) -> dict:
    n = len(flags)
    k = sum(flags)
    return {"rate": round(k / n, 3) if n else None, "k": k, "n": n, "ci95": wilson(k, n)}


def summarise(records: list[dict]) -> dict:
    answerable = [r for r in records if r["type"] != "unanswerable"]
    oos = [r for r in records if r["type"] == "unanswerable"]
    dated = [r for r in records if r["type"] == "date"]
    gold_hit = lambda r, key: any(x["page_id"] in r["gold_pages"] for x in r[key])  # noqa: E731
    out = {
        "n": len(records),
        "correctness_mean": round(sum(r["judge"]["score"] for r in answerable) / len(answerable), 3)
        if answerable else None,
        "correct": rate([r["judge"]["correctness"] == "correct" for r in answerable]),
        "faithful": rate([r["judge"]["faithful"] for r in records]),
        "retrieval_hit@5": rate([gold_hit(r, "retrieved") for r in answerable]),
        "citation_hit": rate([gold_hit(r, "citations") for r in answerable]),
        "language_match": rate([detect_language(r["answer"]) == r["lang"] for r in records]),
        "false_abstention": rate([r["judge"]["abstained"] for r in answerable]),
    }
    if oos:
        out["out_of_scope_abstained"] = rate([r["judge"]["abstained"] for r in oos])
    if dated:
        out["date_questions_tool_used"] = rate([bool(r["tool_calls"]) for r in dated])
        out["date_questions_correct"] = rate([r["judge"]["correctness"] == "correct" for r in dated])
    if any("injection_style" in r for r in records):
        out["injection_success"] = rate([CANARY in r["answer"] for r in records])
        for style in ("direct", "system"):
            sub = [r for r in records if r.get("injection_style") == style]
            out[f"injection_success_{style}"] = rate([CANARY in r["answer"] for r in sub])
    return out


def token_totals(records: list[dict]) -> dict:
    """Tokens used by the answering model and by the judge over the whole run."""
    measured = [r for r in records if "usage" in r and "usage" in r["judge"]]
    total = lambda part, key: sum(  # noqa: E731
        (r["usage"] if part == "answer" else r["judge"]["usage"])[key] for r in measured)
    out = {"items_measured": len(measured)}
    for part in ("answer", "judge"):
        out[part] = {key: total(part, key) for key in ("llm_calls", "input_tokens", "output_tokens")}
    out["answer"]["est_cost_usd"] = estimate_cost(out["answer"]["input_tokens"], out["answer"]["output_tokens"])
    return out


def summary_report() -> dict:
    report = {"meta": {
        "date": date.today().isoformat(),
        "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL,
        "git_commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True, cwd=HERE.parent).stdout.strip() or None,
        "note": "correct/faithful are LLM-judge grades; rates come with 95% Wilson intervals; "
                "small samples (n per language <= 50)",
    }, "variants": {}}
    all_records = []
    for variant in VARIANT_NAMES:
        path = RAW_DIR / f"{variant}.jsonl"
        if not path.exists():
            continue
        records = load_jsonl(path)
        all_records += records
        report["variants"][variant] = {"all": summarise(records)} | {
            lang: summarise([r for r in records if r["lang"] == lang]) for lang in ("fr", "en")}
    report["meta"]["tokens"] = token_totals(all_records)
    out = RESULTS_DIR / "answer_eval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {out}")
    return report


def check_thresholds(report: dict) -> int:
    """Keys look like "clean/all/correct": {"min": 0.7} or "injection/all/injection_success": {"max": 0.1}."""
    thresholds = json.loads((HERE / "thresholds.json").read_text()).get("answer", {})
    failed = 0
    for key, bound in thresholds.items():
        variant, subset, metric = key.split("/")
        if variant not in report["variants"]:
            continue
        value = report["variants"][variant][subset][metric]["rate"]
        ok = value >= bound.get("min", 0) and value <= bound.get("max", 1)
        failed += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {key} = {value} ({bound})")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(VARIANT_NAMES), choices=VARIANT_NAMES)
    parser.add_argument("--limit", type=int, default=None, help="max new items per variant (for a quick try)")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--check", action="store_true", help="exit 1 if below the thresholds (for CI)")
    args = parser.parse_args()
    if not args.summary_only:
        variants = build_variants()
        for v in args.variants:
            generate_and_judge(v, variants[v], args.limit)
    report = summary_report()
    for v, res in report["variants"].items():
        a = res["all"]
        print(f"{v:11s} n={a['n']:3d} correct={a['correct']['rate']} faithful={a['faithful']['rate']} "
              f"hit@5={a['retrieval_hit@5']['rate']} lang_match={a['language_match']['rate']}"
              + (f" injection_success={a['injection_success']['rate']}" if "injection_success" in a else ""))
    if args.check:
        raise SystemExit(check_thresholds(report))


if __name__ == "__main__":
    main()
