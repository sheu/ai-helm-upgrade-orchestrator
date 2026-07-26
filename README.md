# AI-Assisted GitOps Orchestration for Safe Helm Upgrades
## Capstone Project 7 — Integrative Industry Synthesis

**Industry:** Information Technology — Platform Engineering, Kubernetes, GitOps  
**Student:** Udacity AI Mastery Capstone  

---

## Project Overview

This project demonstrates an AI-assisted system that automates the analysis, risk scoring, validation, and GitOps change generation phases of Helm chart upgrades across multi-cluster Kubernetes environments. A platform engineer submits an upgrade request; the system researches release notes, calculates risk, validates Helm charts, evaluates INT health, and proposes production changes — all while preserving human approval for production deployments.

## Prior Projects Integrated

| Prior Project | Domain | Contribution |
|---|---|---|
| Project 2 — Statistical Data Analysis | Data & Statistics | Pandas EDA methods for inventory quality analysis |
| Project 3 — Applied ML (LSTM) | Machine Learning | Feature-weight risk scoring framework |
| Project 5 — Generative AI (VAE) | Generative AI | LLM synthesis of release notes into structured findings |
| Project 6a — Agentic AI Capstone | Agentic AI | Tool-registry discipline and audit-logging pattern for UpgradeResearchAgent |
| Project 6b — Agentic AI Beaver Choice | Agentic AI | Multi-agent sequential orchestration pattern |

## Repository Structure

```
ai-helm-upgrade-orchestrator/
├── Integrated_Helm_Upgrade_Orchestrator.ipynb  ← Main artifact
├── requirements.txt
├── README.md
├── src/
│   ├── models.py         # Pydantic data models
│   ├── inventory.py      # Cluster inventory analysis (P2)
│   ├── risk.py           # Risk scoring model (P3)
│   ├── knowledge.py      # Release note retrieval
│   ├── helm_tools.py     # Helm lint/template wrappers
│   ├── monitoring.py     # Health gate evaluation
│   ├── gitops.py         # GitOps change generation
│   ├── agents.py         # Research agent with LLM synthesis (P5, P6a)
│   ├── orchestrator.py   # Multi-agent coordinator (P6b)
│   └── reporting.py      # Audit logging
├── config/
│   ├── quality_gates.yaml
│   ├── risk_rules.yaml
│   └── components.yaml
├── data/
│   ├── cluster_inventory.csv
│   ├── upgrade_history.csv
│   ├── component_dependencies.csv
│   ├── release_notes/
│   └── runbooks/
├── environments/          # Helm values per cluster/environment
├── charts/                # Helm chart definitions (kafka-connect, prometheus, grafana, loki)
├── scenarios/             # 5 upgrade scenario YAML files
├── outputs/               # Generated upgrade plans, diffs, audit logs, reports
├── diagrams/              # Architecture diagram
└── paper/
    └── Reflective_Synthesis_Paper.pdf
```

## Prerequisites

- Python 3.9+
- Helm 3.x or 4.x (`helm` on PATH)
- OpenAI API key (for LLM research synthesis; system degrades gracefully without it)

## Setup

```bash
cd ai-helm-upgrade-orchestrator
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
# Or add to a .env file in the parent directory
```

## Running the Notebook

```bash
source venv/bin/activate
jupyter notebook Integrated_Helm_Upgrade_Orchestrator.ipynb
```

Run all cells from top to bottom. All outputs must remain visible for submission.

To execute non-interactively:
```bash
jupyter nbconvert --to notebook --execute --inplace \
    Integrated_Helm_Upgrade_Orchestrator.ipynb \
    --ExecutePreprocessor.timeout=300
```

## Upgrade Scenarios Demonstrated

| # | Scenario | Expected Outcome |
|---|---|---|
| 1 | Patch upgrade (0.18.2 → 0.18.3) | Low risk, promote recommended |
| 2 | Major upgrade with breaking value change | High risk, review required |
| 3 | Kubernetes version incompatibility | BLOCKED before deployment |
| 4 | Failed INT deployment (high restarts/errors) | Rollback recommended |
| 5 | Missing observability evidence | UNKNOWN state, human investigation |

## System Architecture

```
Upgrade Request
      │
      ▼
[Inventory Analysis]───→ InventoryReport
      │                    (Pandas EDA — P2)
      ▼
[Research Agent]────────→ ResearchReport
      │                    (tool-constrained retrieval + LLM synthesis — P5, P6a)
      ▼
[Planning/Risk]─────────→ UpgradePlan
      │                    (Risk model — P3)
      ▼
[Validation]────────────→ ValidationResult
      │                    (helm lint/template)
      ▼
[Health Evaluation]─────→ HealthEvaluationResult
      │                    (Quality gates)
      ▼
[GitOps Generator]──────→ outputs/proposed_changes/
      │
      ▼
Human Approval ◄───────── REQUIRED for PROD
      │
      ▼
   PROD Rollout
```

## Key Safety Properties

- Production deployment always requires human approval (hard constraint, not soft recommendation)
- `GateResult.UNKNOWN` is never treated as `PASS` — missing evidence blocks promotion
- LLM output never directly determines gate results — Python decides
- Agents cannot execute arbitrary shell commands or access Kubernetes secrets
- All reasoning steps are recorded in a JSONL audit log

## Ethical Considerations

See `paper/Reflective_Synthesis_Paper.pdf` Section 5 for full ethical analysis covering:
automation bias, hallucinated claims, prompt injection through documentation, secret exposure, and accountability.

## Reproducibility

```bash
source venv/bin/activate
pip freeze > requirements.txt
```

All outputs in `outputs/` are generated by running the notebook. No pre-generated results are included.

## Submission Checklist

- [x] Integrated notebook (`Integrated_Helm_Upgrade_Orchestrator.ipynb`) — all cells executed
- [x] Reflective Synthesis Paper (`paper/Reflective_Synthesis_Paper.pdf`) — ~1,750 words
- [x] Architecture diagram (`diagrams/architecture.png` or described in notebook)
- [x] `requirements.txt`
- [x] All supporting source files (`src/`, `config/`, `data/`, `charts/`, `environments/`, `scenarios/`)
- [x] `outputs/` directory with generated artefacts
