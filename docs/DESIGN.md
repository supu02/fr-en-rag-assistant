# Design decisions

Each choice below has a reason and an alternative that was considered.
Numbers come from `results/`; see the README for the full tables.

## Corpus

- **Official XML, not web scraping.** DILA publishes every page as XML on data.gouv.fr under the Licence Ouverte 2.0, so reuse is allowed (with attribution) and the structure is clean.
- **362 pages on renting and work.** These topics have many deadlines (notice periods, deposits, withdrawal periods), which gives the date tool real work. Out-of-scope topics (passports, taxes, cars) are used as "should refuse" questions.
- **The processed chunks are committed** (`data/corpus/chunks.jsonl`), because the dataset is updated daily. Without a fixed snapshot, results could change from one run to the next.

## Chunking

- **Cut along the document's own sections, and repeat the heading path at the top of each chunk.** Many pages give different rules for different cases (furnished / unfurnished, private / public sector). A chunk cut every N characters can lose the case it belongs to; a chunk that starts with "Deposit > Furnished flat > Inspection matches" cannot.
- **Max ~1,200 characters per chunk, no overlap.** The e5 model reads up to 512 tokens (~1,800 French characters). Overlap is less useful when every chunk repeats its heading path.
- Trade-off: many cases repeat the same text, so the corpus has 8,652 chunks but only ~5,700 distinct bodies.

## Embeddings and vector store

- **`intfloat/multilingual-e5-small` through fastembed (ONNX).** Multilingual (French and English share one vector space), small enough for CPU, MIT licence. ONNX instead of PyTorch keeps the Docker image much smaller.
- The e5 models need the prefixes `query: ` and `passage: `.
- **Qdrant in local mode.** No server to run. Moving to a Qdrant server or Qdrant Cloud means changing one line (`QdrantClient(url=...)`).
- Alternatives: Chroma or FAISS would work as well at this size (8.6k vectors). A larger model (e5-large) may retrieve better but makes the image about 2 GB larger; not measured yet.

## Retrieval

- **At most 2 chunks per page** (`dense_diverse`). Without this, one page often filled all 5 slots with near-identical chunks.
- **Hybrid (dense + BM25 with Reciprocal Rank Fusion) for French only.** BM25 needs shared words between question and page. It helped French questions and hurt English ones, so `auto` picks the method from the detected language.
- **Translate English questions to French for the search only.** The answer is still written in English. This costs one extra LLM call per English question.
- **Honesty note:** these choices were made after looking at results on the evaluation questions themselves (no separate development set), so scores for the chosen setup are optimistic.

## Agent

- **LangGraph with explicit nodes** (translate → retrieve → agent ⇄ tools) instead of one big prompt or a generic ReAct agent. Each step can be tested and measured on its own, and retrieval always runs, so every answer has sources to cite.
- **Date tool.** LLMs make mistakes with calendar arithmetic (for example, "1 month after 30 January"). The LLM picks the numbers from the sources; Python does the maths. Month arithmetic clamps to the end of the month, like the French rule for deadlines in months.
- **Citations as `[n]`** referring to the numbered sources in the prompt. The code keeps only numbers that exist, and the API returns the page links.
- **Prompt-injection defence by instruction only:** the system prompt says the user message is only a question. The robustness suite measures how well that works; there is no separate classifier.

## Evaluation

- **Paired questions:** every topic is asked in French and in English, so the language comparison is not affected by different topics.
- **Retrieval metrics without an LLM** (hit@k, MRR) are free and deterministic, so they run in CI as a regression test.
- **LLM judge for answers,** using a larger model than the answering model, with structured output (correct / partial / incorrect, faithful, abstained). The judge can be wrong: `judge_agreement.py` compares it with human labels.
- **Confidence intervals (Wilson) and "lost / gained" counts.** With 45 questions per language, one question is 2.2 points. Averages can hide changes that cancel out, so the robustness tables also count how many questions changed result.
- **Deterministic perturbations** (seeded from the question id), so two runs see exactly the same typos.

## Serving and deployment

- **FastAPI**: `/search` works without an API key (useful for a public demo and for CI smoke tests); `/ask` returns 503 if the key is missing.
- **The index is built inside the Docker image**, so the container starts fast and search needs no network. Only `/ask` calls the Mistral API.
- **Secrets** come from environment variables (`.env` locally, platform secrets in the cloud); `.env` is in both `.gitignore` and `.dockerignore`.
