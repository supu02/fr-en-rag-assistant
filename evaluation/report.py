"""Write the results tables in README.md from the saved results files.

The numbers in the README are never typed by hand: this script reads
results/retrieval.json and results/answer_eval.json and replaces the text
between the RESULTS markers. CI runs it with --check and fails if the
README does not match the saved results.

  python -m evaluation.report          # update README.md
  python -m evaluation.report --check  # exit 1 if README.md is out of date
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def ci(r: dict) -> str:
    lo, hi = r["ci95"]
    return f"{pct(r['rate'])} ({pct(lo)}–{pct(hi)})"


def retrieval_tables(r: dict) -> list[str]:
    m = r["meta"]
    lines = [
        f"### Retrieval (no LLM; {m['questions']} answerable questions, half French, half English)",
        "",
        "Share of questions where a page labelled as correct is among the top 5 results (hit@5), "
        "with the 95% interval. MRR@10 rewards finding it near the top.",
        "",
        "| Method | hit@5 FR | hit@5 EN | MRR@10 FR | MRR@10 EN |",
        "|---|---|---|---|---|",
    ]
    for name, v in r["methods"].items():
        f, e = v["fr"], v["en"]
        lines.append(f"| {name} | {pct(f['hit@5'])} ({pct(f['hit@5_ci95'][0])}–{pct(f['hit@5_ci95'][1])}) "
                     f"| {pct(e['hit@5'])} ({pct(e['hit@5_ci95'][0])}–{pct(e['hit@5_ci95'][1])}) "
                     f"| {f['mrr@10']:.2f} | {e['mrr@10']:.2f} |")
    lines += [
        "",
        f"### Retrieval robustness (method `{m.get('robustness_method', 'auto')}`)",
        "",
        "Same questions with small changes that keep the meaning. "
        "\"Lost / gained\" counts questions whose result changed compared with the clean question.",
        "",
        "| Change | hit@5 FR | hit@5 EN | FR lost / gained | EN lost / gained |",
        "|---|---|---|---|---|",
    ]
    for name, v in r["robustness"].items():
        fl = v.get("flips")
        f_fl = f"{fl['fr']['lost']} / {fl['fr']['gained']}" if fl else "–"
        e_fl = f"{fl['en']['lost']} / {fl['en']['gained']}" if fl else "–"
        lines.append(f"| {name} | {pct(v['fr']['hit@5'])} | {pct(v['en']['hit@5'])} | {f_fl} | {e_fl} |")
    lines += ["", f"Source: `results/retrieval.json` ({m['date']}, {m['chunks']} chunks, `{m['embed_model']}`)."]
    return lines


def answer_tables(a: dict) -> list[str]:
    m = a["meta"]
    lines = [
        "",
        f"### Answers (answer model `{m['answer_model']}`, judge `{m['judge_model']}`)",
        "",
        "\"Correct\" and \"faithful\" are grades from the LLM judge. Brackets: 95% interval.",
        "",
        "| Variant | Lang | n | Correct | Faithful to cited sources | Retrieval hit@5 | Cites a gold page "
        "| Answer in question's language |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant, v in a["variants"].items():
        for lang in ("fr", "en"):
            s = v[lang]
            lines.append(f"| {variant} | {lang.upper()} | {s['n']} | {ci(s['correct'])} | {ci(s['faithful'])} "
                         f"| {pct(s['retrieval_hit@5']['rate'])} | {pct(s['citation_hit']['rate'])} "
                         f"| {pct(s['language_match']['rate'])} |")
    clean = a["variants"].get("clean", {}).get("all")
    extra = []
    if clean:
        if "out_of_scope_abstained" in clean:
            o = clean["out_of_scope_abstained"]
            extra.append(f"| Out-of-scope questions where the assistant said it could not answer "
                         f"| {o['k']}/{o['n']} |")
        if "date_questions_tool_used" in clean:
            t, c = clean["date_questions_tool_used"], clean["date_questions_correct"]
            extra.append(f"| Date questions where the date tool was called | {t['k']}/{t['n']} |")
            extra.append(f"| Date questions graded correct | {c['k']}/{c['n']} |")
        fa = clean["false_abstention"]
        extra.append(f"| Answerable questions where the assistant wrongly said it could not answer "
                     f"| {fa['k']}/{fa['n']} |")
    inj = a["variants"].get("injection", {}).get("all")
    if inj:
        for key, label in (("injection_success_direct", "\"Ignore previous instructions\""),
                           ("injection_success_system", "Fake \"[SYSTEM]\" rule")):
            s = inj[key]
            extra.append(f"| Prompt injection that worked: {label} | {s['k']}/{s['n']} |")
    if extra:
        lines += ["", "| Check (clean questions unless stated) | Result |", "|---|---|", *extra]
    lines += ["", f"Source: `results/answer_eval.json` ({m['date']}, commit `{m['git_commit']}`)."]
    return lines


def findings(r: dict | None, a: dict | None) -> list[str]:
    """Short sentences computed from the results, so the text never drifts from the data."""
    out = []
    if r:
        mth, n = r["methods"], r["meta"]["questions"] // 2
        auto, dense, div, hyb = mth["auto"], mth["dense"], mth["dense_diverse"], mth["hybrid"]
        out.append(f"English questions find the right French page less often than French questions: "
                   f"hit@5 {pct(auto['en']['hit@5'])} vs {pct(auto['fr']['hit@5'])} ({n} topics asked in both languages).")
        out.append(f"Keeping at most 2 chunks per page raised hit@5 from {pct(dense['all']['hit@5'])} "
                   f"to {pct(div['all']['hit@5'])} (all questions).")
        out.append(f"Adding keyword search (BM25, hybrid) helps French questions "
                   f"({pct(div['fr']['hit@5'])} -> {pct(hyb['fr']['hit@5'])}) but not English ones "
                   f"({pct(div['en']['hit@5'])} -> {pct(hyb['en']['hit@5'])}), so `auto` uses it for French only. "
                   f"This choice was made on the same questions, so `auto` scores are optimistic.")
        ws = r["robustness"].get("whitespace", {}).get("flips")
        if ws:
            changed = sum(ws[lg]["lost"] + ws[lg]["gained"] for lg in ("fr", "en"))
            out.append(f"Extra spaces alone (same words, same meaning) changed the retrieval result for "
                       f"{changed} of {2 * n} questions.")
    if a and "clean" in a["variants"]:
        c = a["variants"]["clean"]
        out.append(f"Answers graded correct: {pct(c['fr']['correct']['rate'])} in French, "
                   f"{pct(c['en']['correct']['rate'])} in English.")
        if r:
            out.append(f"Translating English questions to French before the search: English retrieval hit@5 "
                       f"{pct(r['methods']['auto']['en']['hit@5'])} without translation (retrieval-only run) vs "
                       f"{pct(c['en']['retrieval_hit@5']['rate'])} with translation (answer run).")
    if a and "injection" in a["variants"]:
        i = a["variants"]["injection"]["all"]["injection_success"]
        out.append(f"Hidden prompt injections made the assistant print the attacker's code word in "
                   f"{i['k']} of {i['n']} answers.")
    return ["### Key findings", "", *[f"- {f}" for f in out], ""] if out else []


def render() -> str:
    retrieval = ROOT / "results" / "retrieval.json"
    answers = ROOT / "results" / "answer_eval.json"
    r = json.loads(retrieval.read_text(encoding="utf-8")) if retrieval.exists() else None
    a = json.loads(answers.read_text(encoding="utf-8")) if answers.exists() else None
    lines = findings(r, a)
    if r:
        lines += retrieval_tables(r)
    if a:
        lines += answer_tables(a)
    return "\n".join(lines)


def updated_readme() -> str:
    text = README.read_text(encoding="utf-8")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{render()}\n{END}{tail}"


if __name__ == "__main__":
    new = updated_readme()
    if "--check" in sys.argv:
        if new != README.read_text(encoding="utf-8"):
            print("README.md results are out of date: run `python -m evaluation.report`")
            sys.exit(1)
        print("README.md results match the saved results files")
    else:
        README.write_text(new, encoding="utf-8")
        print("README.md updated")
