"""Tests for the LangGraph agent and the API, with a fake LLM and a fake
retriever: no network, no API key, no vector index needed."""

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


class FakeToolLLM(FakeMessagesListChatModel):
    """Returns the scripted messages in order; accepts bind_tools like a real model."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_format_sources_numbers_each_chunk():
    text = format_sources(SOURCES)
    assert text.startswith("[1] Préavis du locataire > Logement meublé\nLe préavis est de 1 mois.")
    assert "[2] Dépôt de garantie > Logement vide" in text


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


def test_french_question_is_not_translated_and_bad_citations_are_ignored():
    llm = FakeToolLLM(responses=[AIMessage(content="Le préavis est de 1 mois [1][7].")])
    retriever = FakeRetriever()
    result = run(build_graph(llm, retriever), "Quel préavis pour un logement meublé ?")
    assert result["language"] == "fr"
    assert retriever.queries == ["Quel préavis pour un logement meublé ?"]
    assert [c["n"] for c in result["citations"]] == [1]  # [7] does not exist, so it is dropped


def test_api_search_and_health(monkeypatch):
    from fastapi.testclient import TestClient

    from rag_assistant import api

    fake = FakeRetriever()
    fake.chunks = SOURCES
    monkeypatch.setattr(api, "get_retriever", lambda: fake)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").json() == {"status": "ok", "llm_configured": False, "chunks": 2}
        r = client.post("/search", json={"question": "préavis meublé", "k": 1})
        assert r.status_code == 200 and r.json()["results"][0]["page_id"] == "F1168"
        assert client.post("/search", json={"question": "préavis", "method": "magic"}).status_code == 422
        assert client.post("/ask", json={"question": "no key configured"}).status_code == 503
        assert "Ask about renting" in client.get("/").text
