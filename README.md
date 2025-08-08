# LangChain Analytics & RAG Playbook

Production‑leaning reference for building a governed analytics assistant with LangChain: data loading, retrieval (mock & real), hybrid search, lightweight agents, storytelling, cost/logging, and evaluation utilities.

## Core Components
| Area | Location | Summary |
|------|----------|---------|
| Data Loading | `src/data_loading.py` | Superstore CSV loader & sampling |
| Retrieval | `src/retrieval.py` | Mock hash embeddings & optional OpenAI + Chroma |
| Agent | `src/agents.py` | Simple tool selection scaffold |
| Metrics & Logging | `src/metrics.py` | Token estimate, interaction log, recall metric |
| Storytelling | `src/storytelling.py` | KPI narrative formatter |
| Env Check | `scripts/env_check.py` | Verifies key libs installed |
| Vector Build | `scripts/build_vector_store.py` | Build/persist vector store |
| Tests | `tests/` | Token + retrieval smoke tests |
| Play Notebooks | `genai_langchain_practice.ipynb`, `notebooks/langchain_core_rag_agents.ipynb` | Hands‑on exploration |
| Reference Docs | `genai_answers_guide.md`, `genai_interview_questions.md`, `stories/`, `scenarios/` | Q&A + narrative + blueprints |

## Quick Start
1. (Optionally) set `OPENAI_API_KEY` in a `.env` file (see `.env.example`).
2. Install dependencies:
	```powershell
	pip install -r requirements.txt
	```
3. Run environment check:
	```powershell
	python scripts/env_check.py
	```
4. (Optional) Register isolated kernel:
	```powershell
	python scripts/setup_env.py
	```
5. Open a notebook (`genai_langchain_practice.ipynb` or `notebooks/langchain_core_rag_agents.ipynb`) and run setup cells.
6. For mock embeddings (no API key) similarity & hybrid retrieval still function.
7. Switch to real embeddings by providing the key and persisting Chroma: 
	```powershell
	python scripts/build_vector_store.py --inputs metric_defs.txt --persist ./.chroma --real
	```
8. Run smoke script headlessly:
	```powershell
	python scripts/smoke_check.py
	```
9. Use CLI for quick tasks:
	```powershell
	python scripts/cli.py tokens "Churn reduction via better onboarding"
	python scripts/cli.py story '{"revenue_growth_pct":11.2,"aov":57.9,"orders":350}'
	```

## Minimal Workflow
1. Load data → `load_superstore_csv()`
2. Build / load vector store → `build_vector_store()`
3. Retrieve context → `hybrid_search()`
4. Generate or reason via agent → `SimpleAgent.run()`
5. Log & evaluate → `log_interaction()`, `retrieval_recall()`
6. Summarize KPIs → `kpi_story()`

## Tests
Install pytest if not already (added to requirements) and run:
```powershell
pytest -q
```

## Governance & Safety Highlights
- No PII in prompts (developer responsibility)
- Deterministic mock path without external calls for offline dev
- Cost estimation & hashed prompt IDs for audit trail extension
- Simple recall metric encourages measurable retrieval quality
- Guarded imports: falls back to mock embeddings if real ecosystem packages missing
- Heuristic + tokenizer-based token counting (tiktoken when available)

## Extending Ideas
- Add structured output enforcement with Pydantic models
- Integrate moderation / sensitive term filter
- Add feedback loop store (thumbs up/down JSONL)
- Multi-vector (dense + BM25) retriever fusion

## Data
`data/superstore_sample.csv` retained as canonical demo dataset. The smaller sample dataset has been archived to reduce clutter.

## License
Add a LICENSE file if distributing externally.

---
Maintained as a lean, extensible LangChain analytics template.
