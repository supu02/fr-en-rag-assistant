from evaluation.perturb import CANARY, inject, strip_accents, typos, whitespace
from evaluation.retrieval_eval import first_gold_rank, summarise, wilson
from rag_assistant.lang import detect_language


def test_perturbations_are_deterministic():
    q = "Combien de jours de congés payés par mois de travail ?"
    assert typos(q, seed=7) == typos(q, seed=7)
    assert whitespace(q, seed=7) == whitespace(q, seed=7)
    assert typos(q, seed=7) != q  # something changed


def test_whitespace_keeps_the_words():
    q = "How many days of paid holiday?"
    assert whitespace(q, seed=3).split() == q.split()


def test_strip_accents():
    assert strip_accents("préavis à l'échéance") == "preavis a l'echeance"


def test_injection_contains_canary():
    assert CANARY in inject("Question ?", "fr")
    assert CANARY in inject("Question?", "en", style="system")


def test_language_detection():
    assert detect_language("Combien de temps dure le préavis ?") == "fr"
    assert detect_language("How long is the notice period?") == "en"


def test_first_gold_rank_and_summary():
    results = [{"page_id": "F1"}, {"page_id": "F2"}, {"page_id": "F3"}]
    assert first_gold_rank(results, ["F2"]) == 2
    assert first_gold_rank(results, ["F9"]) is None
    s = summarise([1, 3, None, 6])
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.5
    assert s["mrr@10"] == round((1 + 1 / 3 + 1 / 6) / 4, 3)


def test_wilson_interval_is_wide_for_small_samples():
    low, high = wilson(40, 45)
    assert low < 40 / 45 < high
    assert high - low > 0.15


def test_cohen_kappa():
    from evaluation.judge_agreement import cohen_kappa

    assert cohen_kappa(["correct", "incorrect"] * 5, ["correct", "incorrect"] * 5) == 1.0
    # agreement no better than chance -> kappa near 0
    k = cohen_kappa(["correct", "correct", "incorrect", "incorrect"], ["correct", "incorrect", "correct", "incorrect"])
    assert abs(k) < 1e-9
