"""LLM-as-judge for answer correctness and faithfulness.

A larger model than the answering model grades each answer against the
human-checked reference answer and against the sources the answer cited.

Known limits (written in the README too):
- The judge is from the same model family as the answering model, which can
  favour its own style. evaluation/judge_agreement.py measures how often the
  judge agrees with a human on a sample.
- Faithfulness is checked against the cited chunks only, so an answer that
  states facts without citing anything is marked unfaithful.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rag_assistant.config import JUDGE_MODEL
from rag_assistant.llm import get_judge_model
from rag_assistant.monitoring import TokenCounter

JUDGE_PROMPT = """You grade an assistant's answer to a question about French administrative rules.

Question:
{question}

Reference answer (checked by a human against the official page):
{reference}

Assistant's answer:
{answer}

Text of the sources the assistant cited (in French):
{cited}

Grade three things:
1. correctness:
   - "correct": the answer gives the key facts of the reference (numbers, yes/no, dates). Extra correct details are fine.
   - "partial": a key fact is missing, or a correct fact is mixed with a wrong one.
   - "incorrect": the key fact is wrong or absent.
   If the reference says the sources do not cover the question, "correct" means the assistant clearly says
   it cannot find the answer, and "incorrect" means it gives an answer anyway.
2. faithful: true if every factual claim in the answer is supported by the cited source text above.
   False if any claim is not in that text, or if the answer states facts without citing any source.
   An answer that only says it cannot find the information is faithful.
3. abstained: true if the assistant says it cannot find or cannot answer the question from its sources.
Ignore the language of the answer (French or English are both fine)."""


class Verdict(BaseModel):
    correctness: Literal["correct", "partial", "incorrect"]
    faithful: bool
    abstained: bool
    reason: str = Field(description="one short sentence explaining the grade")


SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def cited_text(result: dict, sources_by_chunk: dict[str, dict]) -> str:
    cited_n = {c["n"] for c in result["citations"]}
    parts = [f"[{r['n']}] {sources_by_chunk[r['chunk_id']]['text']}"
             for r in result["retrieved"] if r["n"] in cited_n]
    return "\n\n".join(parts) if parts else "(no source cited)"


def judge(question: dict, result: dict, sources_by_chunk: dict[str, dict]) -> dict:
    grader = get_judge_model().with_structured_output(Verdict)
    tokens = TokenCounter()
    verdict: Verdict = grader.invoke(JUDGE_PROMPT.format(
        question=question["question"], reference=question["reference"],
        answer=result["answer"], cited=cited_text(result, sources_by_chunk)),
        config={"callbacks": [tokens]})
    return verdict.model_dump() | {"score": SCORE[verdict.correctness], "judge_model": JUDGE_MODEL,
                                   "usage": tokens.as_dict()}
