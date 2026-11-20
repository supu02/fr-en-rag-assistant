"""Tests for the LangGraph agent and the API, with a fake LLM and a fake
retriever: no network, no API key, no vector index needed."""

import json
import logging

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from rag_assistant.agent import build_graph, format_sources, run

SOURCES = [
    {"chunk_id": "F1168#5", "page_id": "F1168", "page_title": "Préavis du locataire",
     "section": "Logement meublé", "url": "https://example.org/F1168",
     "text": "Préavis du locataire > Logement meublé\nLe préavis est de 1 mois."},
    {"chunk_id": "F31269#2", "page_id": "F31269", "page_title": "Dépôt de garantie",
     "section": "Logement vide", "url": "https://example.org/F31269",
     "text": "Dépôt de garantie > Logement vide\nAu maximum 1 mois de loyer."},
]


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, k=5, method="auto"):
        self.queries.append(query)
        return [{**s, "score": 0.9} for s in SOURCES[:k]]

    def add_context(self, hits):
        return hits


class FakeToolLLM(FakeMessagesListChatModel):
    """Returns the scripted messages in order; accepts bind_tools like a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_format_sources_numbers_each_chunk():
    text = format_sources(SOURCES)
    assert text.startswith("[1] Préavis du locataire > Logement meublé\nLe préavis est de 1 mois.")
    assert "[2] Dépôt de garantie > Logement vide" in text


def test_translation_is_cleaned_before_search():
    """Additions seen in the first evaluation's translated queries."""
    from rag_assistant.agent import clean_translation

    q = "What is the maximum deposit for an unfurnished flat?"
    assert clean_translation("Quel est le **dépôt de garantie maximal** pour un logement vide ? (en France, en **2024**)",
                             q) == "Quel est le dépôt de garantie maximal pour un logement vide ?"
    assert clean_translation("À la fin de mon CDD avec [nom de l'employeur si connu], ai-je droit à une prime ?\n"
                             "Note : traduction littérale.", q) == "À la fin de mon CDD avec, ai-je droit à une prime ?"
    assert clean_translation("Traduction : Puis-je avoir un chat ?", q) == "Puis-je avoir un chat ?"
    # Brackets the question itself had are kept; an empty reply falls back to the question.
    assert clean_translation("Ai-je droit à l'APL (aide au logement) ?", "Can I get APL (housing aid)?") == \
        "Ai-je droit à l'APL (aide au logement) ?"
    assert clean_translation("**  **", q) == q


def test_neighbouring_chunks_are_added_once_and_stay_on_the_page():
    from rag_assistant.retrieve import with_neighbours

    def chunk(cid, section, body):
        page = cid.split("#")[0]
        return {"chunk_id": cid, "page_id": page, "page_title": page, "section": section, "url": "",
                "text": f"{page} > {section}\n{body}"}

    chunks = [chunk("A#0", "Logement vide", "Cas général."), chunk("A#1", "Logement vide", "Le préavis..."),
              chunk("A#2", "Logement vide", "...est de 3 mois."), chunk("A#3", "Logement meublé", "1 mois."),
              chunk("B#0", "Intro", "Autre page.")]
    position = {c["chunk_id"]: i for i, c in enumerate(chunks)}
    hits = [{**chunks[1], "score": 0.9}, {**chunks[2], "score": 0.8}, {**chunks[3], "score": 0.7}]
    out = with_neighbours(hits, chunks, position)

    assert [s["chunk_ids"] for s in out] == [["A#0", "A#1", "A#2"], ["A#3"]]  # A#2 was already shown
    assert out[0]["text"] == "A > Logement vide\nCas général.\n\nLe préavis...\n\n...est de 3 mois."
    assert out[0]["score"] == 0.9 and "B#0" not in out[1]["chunk_ids"]  # no text from another page
    hits = [{**chunks[2], "score": 0.9}]
    assert "(Logement meublé)\n1 mois." in with_neighbours(hits, chunks, position)[0]["text"]


def test_english_question_is_translated_for_search_and_tool_is_called():
    llm = FakeToolLLM(responses=[
        AIMessage(content="Combien de préavis pour un meublé si le propriétaire a reçu la lettre le 30 janvier 2027 ?"),
        AIMessage(content="", tool_calls=[{"name": "add_to_date", "id": "call_1",
                                           "args": {"start_date": "2027-01-30", "amount": 1, "unit": "months"}}]),
        AIMessage(content="Your notice is 1 month [1], so it ends on 28 February 2027 [1]."),
    ])
    retriever = FakeRetriever()
    graph = build_graph(llm, retriever, today="2026-11-18")
    result = run(graph, "Furnished flat, my landlord got my notice on 30 January 2027: when does it end?")

    assert result["language"] == "en"
    assert retriever.queries[0].startswith("Combien de préavis")  # searched in French
    assert result["tool_calls"] == [{"name": "add_to_date",
                                     "args": {"start_date": "2027-01-30", "amount": 1, "unit": "months"}}]
    assert [c["page_id"] for c in result["citations"]] == ["F1168"]
    assert "28 February 2027" in result["answer"]


def test_bad_tool_arguments_go_back_to_the_model():
    """The model can fix a wrong tool call instead of the request failing (seen in the evaluation)."""
    from langchain_core.messages import ToolMessage

    seen = []

    class RecordingLLM(FakeToolLLM):
        def invoke(self, messages, *args, **kwargs):
            seen.append(messages)
            return super().invoke(messages, *args, **kwargs)

    llm = RecordingLLM(responses=[
        AIMessage(content="", tool_calls=[{"name": "add_to_date", "id": "call_1",
                                           "args": {"start_date": "2027-01-30", "amount": 2, "unit": "fortnights"}}]),
        AIMessage(content="", tool_calls=[{"name": "add_to_date", "id": "call_2",
                                           "args": {"start_date": "2027-01-30", "amount": 4, "unit": "weeks"}}]),
        AIMessage(content="Le préavis se termine le 27 février 2027 [1]."),
    ])
    result = run(build_graph(llm, FakeRetriever(), today="2026-11-18"), "Quand se termine mon préavis ?")
    assert result["answer"].startswith("Le préavis se termine")
    errors = [m for m in seen[1] if isinstance(m, ToolMessage)]
    assert errors and "unit must be days, weeks, months or years" in errors[0].content


def test_french_question_is_not_translated_and_bad_citations_are_ignored():
    llm = FakeToolLLM(responses=[AIMessage(content="Le préavis est de 1 mois [1][7].")])
    retriever = FakeRetriever()
    result = run(build_graph(llm, retriever), "Quel préavis pour un logement meublé ?")
    assert result["language"] == "fr"
    assert retriever.queries == ["Quel préavis pour un logement meublé ?"]
    assert [c["n"] for c in result["citations"]] == [1]  # [7] does not exist, so it is dropped


def _no_llm(monkeypatch):
    """Act as if no LLM credentials exist (no Mistral key, no AWS sign-in)."""
    from rag_assistant import api, llm

    monkeypatch.setattr(llm, "llm_configured", lambda: False)
    monkeypatch.setattr(api, "llm_configured", lambda: False)


def test_api_search_and_health(monkeypatch):
    from fastapi.testclient import TestClient

    from rag_assistant import api

    fake = FakeRetriever()
    fake.chunks = SOURCES
    monkeypatch.setattr(api, "get_retriever", lambda: fake)
    monkeypatch.setattr(api, "embed_query", lambda text: None)  # no model download in tests
    _no_llm(monkeypatch)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok", "llm_configured": False, "chunks": 2}
        r = client.post("/search", json={"question": "préavis meublé", "k": 1})
        assert r.status_code == 200 and r.json()["results"][0]["page_id"] == "F1168"
        assert client.post("/search", json={"question": "préavis", "method": "magic"}).status_code == 422
        assert client.post("/ask", json={"question": "no key configured"}).status_code == 503
        assert "Ask about renting" in client.get("/").text


def test_run_counts_tokens_of_every_llm_call():
    usage = {"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050}
    llm = FakeToolLLM(responses=[
        AIMessage(content="Quel préavis pour un meublé ?", usage_metadata=usage),  # translation
        AIMessage(content="One month [1].", usage_metadata=usage),  # answer
    ])
    result = run(build_graph(llm, FakeRetriever()), "What notice for a furnished flat?")
    assert result["usage"] == {"llm_calls": 2, "input_tokens": 2000, "output_tokens": 100}
    assert result["retrieved"][0]["score"] == 0.9


def _logs(caplog, event="request"):
    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "rag_assistant.requests"]
    return [line for line in lines if line["event"] == event]


def _request_logs(caplog):
    return _logs(caplog, "request")


def test_api_writes_one_log_line_per_request(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    from rag_assistant import api

    caplog.set_level(logging.INFO, logger="rag_assistant.requests")
    fake = FakeRetriever()
    fake.chunks = SOURCES
    monkeypatch.setattr(api, "get_retriever", lambda: fake)
    monkeypatch.setattr(api, "embed_query", lambda text: None)  # no model download in tests
    usage = {"input_tokens": 3000, "output_tokens": 100, "total_tokens": 3100}
    llm = FakeToolLLM(responses=[AIMessage(content="Je ne trouve pas cette information.", usage_metadata=usage)])
    monkeypatch.setattr(api, "get_graph", lambda: build_graph(llm, fake))

    with TestClient(api.app) as client:
        client.get("/health")
        client.post("/search", json={"question": "préavis meublé", "k": 2})
        r = client.post("/ask", json={"question": "Quel est le prix d'un passeport ?"})

    assert r.headers["X-Request-ID"]
    (startup,) = _logs(caplog, "startup")  # one line when the app starts (a cold start on Lambda)
    assert startup["startup_ms"] >= startup["model_ms"] >= 0 and startup["index_ms"] >= 0
    search_log, ask_log = _request_logs(caplog)  # /health is not logged
    assert search_log["endpoint"] == "/search" and search_log["status"] == 200
    assert search_log["lang"] == "fr" and search_log["retrieval_method"] == "hybrid"
    assert search_log["n_results"] == 2 and search_log["top_score"] == 0.9
    assert ask_log["endpoint"] == "/ask" and ask_log["no_citation"] is True
    assert ask_log["input_tokens"] == 3000 and ask_log["llm_calls"] == 1
    assert ask_log["est_cost_usd"] > 0 and ask_log["latency_ms"] >= 0
    assert "question" not in ask_log and "answer" not in ask_log  # no user text in the logs


def test_api_logs_failures(monkeypatch, caplog):
    from fastapi.testclient import TestClient

    from rag_assistant import api

    caplog.set_level(logging.INFO, logger="rag_assistant.requests")

    class BrokenRetriever(FakeRetriever):
        def search(self, query, k=5, method="auto"):
            raise RuntimeError("index missing")

    broken = BrokenRetriever()
    broken.chunks = SOURCES
    monkeypatch.setattr(api, "get_retriever", lambda: broken)
    monkeypatch.setattr(api, "embed_query", lambda text: None)  # no model download in tests
    _no_llm(monkeypatch)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        assert client.post("/search", json={"question": "préavis"}).status_code == 500
        assert client.post("/ask", json={"question": "no key configured"}).status_code == 503
    crash, no_key = _request_logs(caplog)
    assert crash["status"] == 500 and crash["error"] == "RuntimeError"
    assert no_key["status"] == 503 and no_key["error"] == "LLMNotConfigured"


def test_bedrock_model_settings(monkeypatch):
    """The default provider builds a Bedrock Converse model, deterministic and capped."""
    from langchain_aws import ChatBedrockConverse

    from rag_assistant import config, llm

    monkeypatch.setattr(config, "LLM_PROVIDER", "bedrock")
    monkeypatch.setattr(llm, "llm_configured", lambda: True)
    llm.get_chat_model.cache_clear()
    try:
        model = llm.get_chat_model("mistral.ministral-3-14b-instruct")
        assert isinstance(model, ChatBedrockConverse)
        assert model.region_name == config.BEDROCK_REGION and model.temperature == 0.0
        assert model.max_tokens == 1024
        judge = llm.get_judge_model()
        assert judge.model_id == config.JUDGE_MODEL and judge.region_name == config.BEDROCK_JUDGE_REGION
    finally:
        llm.get_chat_model.cache_clear()


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    import pytest

    from rag_assistant import llm

    monkeypatch.setattr(llm, "llm_configured", lambda: False)
    llm.get_chat_model.cache_clear()
    with pytest.raises(llm.LLMNotConfigured, match="aws login|MISTRAL_API_KEY"):
        llm.get_chat_model()
