# MARC-Clinical: Decentralized Multi-Agent RAG Framework
## for Multi-Disease Clinical Decision Support

![RAG](https://img.shields.io/badge/RAG-Enabled-blue)
![LLMs](https://img.shields.io/badge/LLMs-Gemini%20%7C%20Groq-green)
![Multi--Agent](https://img.shields.io/badge/Multi--Agent-4%20Agents-purple)
![IEEE](https://img.shields.io/badge/Base%20Paper-IEEE%20Access%202025-orange)

## Project Overview
MARC-Clinical is a research framework with four specialist RAG agents (Cardiology, Nephrology,
Diabetology, Pharmacology) and five orchestration modules for decentralized multi-agent coordination:

| Module | Full Name | Novel Contribution |
|---|---|---|
| **SRAL** | Shared Retrieval Awareness Layer | Retrieval deduplication with cross-agent interpretation sharing |
| **IAKB** | Inter-Agent Knowledge Bus | Mid-reasoning knowledge transfer via publish-subscribe |
| **SCDP** | Semantic Conflict Detection Protocol | Evidence-weighted conflict resolution |
| **DCWO** | Decentralized Consensus Without Orchestrator | Gossip-style consensus over agent confidences and treatment positions |
| **AFL** | Adaptive Feedback Loop | Re-queries low-confidence agents with expanded, case-anchored retrieval queries |

Supporting components: BioBERT embeddings + FAISS for local RAG, live PubMed retrieval,
web evidence retrieval (`marc_framework/web_evidence.py`), and an LLM fusion step in the benchmark path.

## Benchmark Results (50 questions per dataset)

| Dataset | Backbone | MARC | Single-agent baseline | Base paper |
|---|---|---|---|---|
| MedQA | Gemini 3.6 Flash | 96.0% (48/50) | **re-run pending** (see below) | 94% |
| PubMedQA | Gemini 3.6 Flash | 86.0% (43/50) | 88.0% (44/50) | 88% |
| MedBullets | Groq gpt-oss-120b | 74.0% (37/50) | 70.0% (35/50) | 84% |

Notes:
- With n = 50, 95% confidence intervals are about +/-10 points. Run `python evaluation/paper_stats.py`
  for exact Wilson intervals and paired McNemar tests.
- `evaluation/medqa_results.json` was produced before the "Unavailable -> A" parsing fix, so its
  baseline column (56.0%) is not trusted. Re-run the baseline with
  `python evaluation/medqa_baseline_rerun.py`; it writes `evaluation/medqa_baseline_rerun.json`
  and does not overwrite the original file.
- The earlier single-case demo numbers (76.2% vs 49.5%) came from `main.py` on one test case and
  are not benchmark results.
- PubMedQA is scored by exact match on yes/no/maybe, not cosine similarity as in the base paper.

## Multi-Model Comparison
`evaluation/model_comparison.py` runs the four agents on the same 20 MedQA questions across three
backbones (Gemini 3.6 Flash, Groq gpt-oss-120b, Groq gpt-oss-20b) and records per-agent accuracy,
per-agent response time and a confidence-vote ensemble. It saves progress after every question and
resumes on rerun; a failed API call is never scored as a wrong answer.

## Base Papers
- **Primary (SCIE):** IEEE Access 2025 — DOI: 10.1109/ACCESS.2025.3613340
- **Supporting (Scopus):** MediHive IEEE ICHI 2026 — arXiv: 2603.27150

## Project Structure
```
MARC_Clinical/
├── agents/                    # 4 specialist RAG agents (Gemini or Groq via llm_provider)
├── marc_framework/            # SRAL, IAKB, SCDP, DCWO, AFL + web_evidence
├── knowledge_bases/           # PDF guidelines (not in repo)
├── vectorstores_biobert/      # cached FAISS indexes
├── evaluation/
│   ├── medqa_benchmark.py
│   ├── pubmedqa_benchmark.py
│   ├── medbullets_benchmark.py
│   ├── medqa_baseline_rerun.py   # clean baseline re-run for MedQA
│   ├── model_comparison.py       # 3 backbones x 4 agents
│   ├── paper_stats.py            # CIs, McNemar, comparison tables
│   └── metrics.py
├── frontend/                  # React web UI
├── main.py                    # Demo pipeline
└── requirements.txt
```

## Setup & Installation

### Backend (Python)
```bash
python -m venv marc_env
marc_env\Scripts\activate   # Windows
source marc_env/bin/activate # Linux/Mac

pip install -r requirements.txt

# Create .env (never commit real keys)
GOOGLE_API_KEY=your_key_here
GROQ_API_KEY=your_key_here

python main.py
```

### Running the evaluation
```bash
python evaluation/medqa_baseline_rerun.py   # clean MedQA baseline
python evaluation/model_comparison.py       # rerun daily until all 3 models finish
python evaluation/paper_stats.py            # tables for the paper
```

### Frontend (React)
```bash
cd frontend
npm install
npm start
# Opens at http://localhost:3000
```

## Knowledge Bases Required
Download and place in respective folders:
| Agent | File | Source |
|---|---|---|
| Cardiology | `knowledge_bases/cardiology/aha_heart_failure.pdf` | ahajournals.org |
| Nephrology | `knowledge_bases/nephrology/kdigo_ckd_2024.pdf` | kdigo.org |
| Diabetology | `knowledge_bases/diabetology/ada_standards_2025.pdf` | wafp.org |
| Pharmacology | `knowledge_bases/pharmacology/who_essential_medicines.pdf` | who.int |

## SDG Alignment
- **SDG 3** — Good Health and Well-Being
- **SDG 9** — Industry, Innovation and Infrastructure

## Department
**Department of AI & Data Science**  
Kongu Engineering College (Autonomous)  
Course: 22ADP72 — Project Work II Phase I  
Academic Year: 2025-26
