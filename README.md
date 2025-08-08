# LangChain Analytics & RAG Playbook 🚀

<p align="center">
  <em>Practical, production‑leaning blueprint for an AI‑assisted analytics stack: Retrieval, Lightweight Agents, KPI Storytelling, Governance, & Cost Awareness.</em>
</p>

<p align="center">
  <a href="https://github.com/And3m/Langchain_playbook/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/And3m/Langchain_playbook/actions/workflows/ci.yml/badge.svg"/></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue"/>
  <img alt="License" src="https://img.shields.io/badge/License-TBD-lightgrey"/>
  <img alt="Status" src="https://img.shields.io/badge/Status-Alpha-orange"/>
</p>

---

## 🔎 Why This Exists
Most LangChain examples stop at “hello world.” This playbook goes further: **offline‑safe mock modes**, **graceful fallbacks**, **retrieval evaluation**, **cost/token estimation**, and **clear pathways to production hardening**—while staying lean and readable for analysts and engineers alike.

## 📑 Table of Contents
1. [Features](#-features)
2. [Architecture](#-architecture-overview)
3. [Core Components](#-core-components)
4. [Quick Start](#-quick-start)
5. [Usage Examples](#-usage-examples)
6. [Workflow Cheat‑Sheet](#-workflow-cheat-sheet)
7. [Testing & CI](#-testing--ci)
8. [Governance & Safety](#-governance--safety)
9. [Roadmap](#-roadmap)
10. [Contributing](#-contributing)
11. [FAQ](#-faq)
12. [License](#-license)

## ✨ Features
- 🔁 **Dual Mode Retrieval**: Real embeddings (OpenAI + Chroma) or deterministic hash mock (offline / cost‑free).
- 🧠 **Lightweight Agent Scaffold**: Minimal tool selection pattern ready for extension.
- 🧾 **KPI Storytelling**: Structured narrative builder for exec summaries.
- 📊 **Token & Cost Estimation**: Heuristic + optional `tiktoken` precision fallback.
- 🧪 **Retrieval Evaluation**: Precision/recall proxy utilities for measurable iteration.
- 🛡️ **Guarded Imports**: Fallback paths avoid hard crashes when optional deps missing.
- 🛠️ **CLI + Smoke Check**: Instant verification & scripted flows (`scripts/cli.py`, `scripts/smoke_check.py`).
- 🧱 **Composable Modules**: Each concern isolated (`metrics`, `retrieval`, `storytelling`, etc.).
- 📓 **Hands‑On Notebooks**: Fundamentals & advanced RAG/agent flows.

## 🏗 Architecture Overview
```
		  +---------------------------+
		  |        Notebooks          |  ← Exploration / Prototyping
		  +-------------+-------------+
							 |
		  +-------------v-------------+
		  |         CLI / Scripts     |  smoke_check | build_vector_store | env_check
		  +-------------+-------------+
							 |
		  +-------------v-------------+
		  |          src/ Modules     |
		  |  retrieval | metrics | ...|
		  +------+------+------+------+
					|      |
		  Mock Embeds   |   Token/Cost Est.
					|      |
		  +------+------v------+
		  |   External APIs    |  (Optional OpenAI Embeddings / LLM)
		  +--------------------+
```

## 🧩 Core Components
| Area | Location | Summary |
|------|----------|---------|
| Data Loading | `src/data_loading.py` | Superstore CSV + sampling helpers |
| Retrieval | `src/retrieval.py` | Mock hash embeddings + optional OpenAI + Chroma |
| Agent | `src/agents.py` | Simple tool selection scaffold |
| Metrics & Logging | `src/metrics.py` | Token estimate, interaction log, recall metric |
| Storytelling | `src/storytelling.py` | KPI narrative formatter |
| Env Check | `scripts/env_check.py` | Verifies key libs installed |
| Vector Build | `scripts/build_vector_store.py` | Build/persist Chroma store |
| Smoke & CLI | `scripts/smoke_check.py`, `scripts/cli.py` | Headless validation & quick ops |
| Tests | `tests/` | Retrieval + token + guardrail checks |
| Notebooks | `genai_langchain_practice.ipynb`, `notebooks/langchain_core_rag_agents.ipynb` | Learning & advanced patterns |
| Reference Docs | `genai_answers_guide.md`, `stories/`, `scenarios/` | Q&A + scenario narratives |

## ⚡ Quick Start
1. (Optional) Create `.env` with `OPENAI_API_KEY=...`
2. Install deps:
	```powershell
	pip install -r requirements.txt
	```
3. Environment check:
	```powershell
	python scripts/env_check.py
	```
4. (Optional) Isolated kernel:
	```powershell
	python scripts/setup_env.py
	```
5. Explore notebooks or run smoke test:
	```powershell
	python scripts/smoke_check.py
	```
6. Build real vector store (needs key):
	```powershell
	python scripts/build_vector_store.py --inputs metric_defs.txt --persist ./.chroma --real
	```

## 🛠 Usage Examples
CLI:
```powershell
python scripts/cli.py tokens "Churn reduction via better onboarding"
python scripts/cli.py story '{"revenue_growth_pct":9.4,"aov":61.2,"orders":312}'
```
Python (library style):
```python
from src.retrieval import build_vector_store, hybrid_search
from src.storytelling import kpi_story

texts = ["Churn reflects customer loss", "LTV captures lifetime profit"]
store = build_vector_store(texts, use_real=False)
print([r.text for r in hybrid_search(store, "What is churn?", k=1)])
print(kpi_story({"revenue_growth_pct": 12.1, "aov": 57.8, "orders": 420}))
```

## 🧾 Workflow Cheat‑Sheet
1. Ingest & prepare: `load_superstore_csv()`
2. Build store: `build_vector_store()` (mock or real)
3. Retrieve context: `hybrid_search()`
4. Reason / chain: (extend agent or notebook chains)
5. Story & summarize: `kpi_story()`
6. Log & estimate cost: `log_interaction()`, `estimate_tokens()`
7. Evaluate retrieval: `retrieval_recall()`

## ✅ Testing & CI
Run tests:
```powershell
pytest -q
```
CI (GitHub Actions) runs: dependency install → smoke check → tests → token spot‑check.

## 🔐 Governance & Safety
| Control | Description |
|---------|-------------|
| Mock Mode | No external calls; deterministic dev path |
| Token Estimation | Upfront cost awareness; encourages budgeting |
| Fallback Imports | Avoids runtime crashes when embeddings libs absent |
| Simple Recall Metric | Drives measurable retrieval iteration |
| (Pluggable) Filters | Add content moderation / PII stripping before real deployment |

## 🗺 Roadmap
- [ ] Structured Pydantic output models
- [ ] Feedback loop store (user ratings)
- [ ] Advanced multi-vector fusion (semantic + BM25)
- [ ] Docker / FastAPI service layer
- [ ] Richer evaluation harness (nDCG / MRR)
- [ ] Observability: latency histogram + JSONL trace log

## 🤝 Contributing
Lightweight process:
1. Fork & branch (`feat/<short-topic>`)
2. Add/adjust tests
3. Run `pytest -q` + `python scripts/smoke_check.py`
4. PR with concise summary & rationale

## ❓ FAQ
**Q: Do I need an API key to start?**  
No—mock mode works out of the box.

**Q: Where do I add more documents?**  
Adapt `build_vector_store.py` or load them in a notebook and rebuild.

**Q: How do I add a new embedding provider?**  
Wrap provider in a function and branch in `build_vector_store` similar to existing pattern.

## 📄 License
TBD – add a LICENSE file (MIT / Apache 2 recommended for openness).

---

<p align="center"><strong>Crafted to be clear first, powerful second. Extend responsibly.</strong></p>
