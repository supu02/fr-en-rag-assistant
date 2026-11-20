"""Does the text given to the model contain the facts it needs? (no LLM, free to run)

A page-level hit (retrieval_eval.py) can hide a miss: the right page is found, but
not the section with the number. This check takes the French answerable questions
whose reference answer contains numbers with a unit ("3 mois", "10 %", "1 200 €")
and counts the questions where every one of those numbers is in the retrieved text.
It is a rough proxy (a number can be written another way), used to compare settings.

  python -m evaluation.context_coverage
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.retrieval_eval import load_jsonl
from rag_assistant.config import RESULTS_DIR

HERE = Path(__file__).parent
WORDS = {"un": "1", "une": "1", "deux": "2", "trois": "3", "quatre": "4", "cinq": "5", "six": "6",
         "huit": "8", "dix": "10", "douze": "12",
         "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "eight": "8", "ten": "10", "twelve": "12"}
UNITS = {"%": "%", "€": "€", "euro": "€", "euros": "€", "an": "year", "ans": "year", "year": "year",
         "years": "year", "mois": "month", "month": "month", "months": "month", "jour": "day", "jours": "day",
         "day": "day", "days": "day", "semaine": "week", "semaines": "week", "week": "week", "weeks": "week",
         "heure": "hour", "heures": "hour", "hour": "hour", "hours": "hour"}
NUMBER_WITH_UNIT = re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(%|€|" + "|".join(sorted(
    (u for u in UNITS if u not in "%€"), key=len, reverse=True)) + r")(?!\w)")


def normalise(text: str) -> str:
    text = text.lower().replace(" ", " ").replace(" ", " ")
    return re.sub(r"\b(" + "|".join(WORDS) + r")\b", lambda m: WORDS[m.group(1)], text)


def facts(text: str) -> set[tuple[str, str]]:
    """Numbers with a unit, French or English: "3 mois" and "three months" both give ("3", "month")."""
    return {(n.replace(",", "."), UNITS[u]) for n, u in NUMBER_WITH_UNIT.findall(normalise(text))}


def all_present(needed: set[tuple[str, str]], text: str) -> bool:
    return needed <= facts(text)


def main() -> None:
    from rag_assistant.retrieve import get_retriever

    retriever = get_retriever()
    questions = [q for q in load_jsonl(HERE / "questions.jsonl") if q["answerable"] and q["lang"] == "fr"]
    items = [(q, facts(q["reference"])) for q in questions]
    items = [(q, f) for q, f in items if f]
    settings = {
        "5 chunks": lambda q: retriever.search(q, k=5),
        "10 chunks": lambda q: retriever.search(q, k=10),
        "5 chunks + neighbours (used by the assistant)": lambda q: retriever.add_context(retriever.search(q, k=5)),
    }
    out = {"questions": len(items), "settings": {}}
    for name, fetch in settings.items():
        found, chars = 0, 0
        for q, needed in items:
            text = "\n".join(s["text"] for s in fetch(q["question"]))
            found += all_present(needed, text)
            chars += len(text)
        out["settings"][name] = {"all_numbers_in_context": found, "avg_context_chars": chars // len(items)}
        print(f"{name:48s} {found}/{len(items)} questions, {chars // len(items):,} characters on average")
    (RESULTS_DIR / "context_coverage.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
