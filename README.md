# writing-coach-eval

Evaluation framework for **Writing Coach V2** — a conversational, LangGraph-based writing assistant. The framework builds a labelled dataset from public writing/editing corpora, augments it with LLM-synthesised domain data, runs the Writing Coach against every example in batch, and captures routing decisions and responses for downstream evaluation.

---

## Overview

Writing Coach V2 classifies every incoming user query into one of four **routes** and processes it through a matching graph path:

| Route | Description |
|---|---|
| `REVISE_SIMPLE` | Style, grammar, tone, clarity edits that do not require external evidence |
| `REVISE_RESEARCH` | Text improvements that require finding supporting evidence or references |
| `RESPOND` | Summarise, compare, analyse, or answer questions about provided content |
| `RESEARCH` | Find papers, explore literature, validate claims against published work |

The evaluation pipeline **so far** has three stages:

1. **Dataset construction** — merge public datasets and label rows by route.
2. **Data synthesis** — use Azure OpenAI to expand the dataset with domain-adapted examples.
3. **Batch inference** — run Writing Coach V2 on every row and store outputs for analysis.

---

## Repository structure

```
writing-coach-eval/
├── data/                              # CSV datasets and query lists
│   ├── data_routes_original.csv       # Merged public dataset (52 rows)
│   ├── data_routes_processed.csv      # Processed version used as synthesis input
│   ├── data_routes_synthetic.csv      # Synthesised rows only
│   ├── data_routes_expanded.csv       # Original + synthetic (208 rows, 52 per route)
│   └── queries/                       # Distinct query lists extracted during exploration
├── src/
│   ├── store_output.py                # Batch inference runner
│   ├── evaluation/
│   │   ├── merge_public_datasets.py   # Build data_routes_original.csv from public HF datasets
│   │   ├── synthesize_domain_data.py  # Synthesise domain-adapted rows via Azure OpenAI
│   │   └── notebooks/
│   │       └── merge_public_datasets.ipynb  # Interactive version of the merge script
│   └── wc_src/                        # Writing Coach V2 source (graph, nodes, prompts)
```

---

## Public datasets used

| Dataset | Source | Route(s) |
|---|---|---|
| [OpenRewriteEval](https://huggingface.co/datasets/gabrielmbmb/OpenRewriteEval) | HuggingFace | REVISE_SIMPLE, REVISE_RESEARCH, RESPOND |
| [CoEdiT](https://huggingface.co/datasets/grammarly/coedit) | HuggingFace / Grammarly | REVISE_SIMPLE |
| [Kiwi](https://huggingface.co/datasets/fangyuan/kiwi) | HuggingFace | REVISE_SIMPLE, REVISE_RESEARCH, RESPOND, RESEARCH |

Each dataset is filtered to queries that map cleanly onto a Writing Coach route, producing `(input, query, target)` triplets.

---

## Setup

### Prerequisites

- Python 3.10+
- Access to Writing Coach V2 source under `src/wc_src/` (not included here)
- An Azure OpenAI deployment for data synthesis

### Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the repo root for the synthesis and batch inference steps:

```
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

---

## Usage

### 1. Build the merged public dataset

```bash
python src/evaluation/merge_public_datasets.py
```

Reads the three public datasets from Hugging Face, filters and labels rows by route, and writes `data/data_routes_original.csv` (52 rows, ~13 per route).

The interactive notebook at `src/evaluation/notebooks/merge_public_datasets.ipynb` provides the same pipeline with exploratory outputs.

### 2. Synthesise domain-adapted data

```bash
python src/evaluation/synthesize_domain_data.py
```

Takes `data/data_routes_processed.csv` as input and uses Azure OpenAI to generate:
- **84 domain-adapted rows** — 28 CS rows × 3 domains (Physics, Biochemistry, Humanities & Social Sciences)
- **72 generic rows** — 36 REVISE_SIMPLE + 24 REVISE_RESEARCH + 12 RESPOND

Combined with the original 52 rows this produces **208 rows (52 per route)**, saved to:
- `data/data_routes_synthetic.csv` — new rows only
- `data/data_routes_expanded.csv` — full expanded dataset

Synthesis is resumable: a checkpoint file (`data/.synthesis_checkpoint.json`) tracks completed rows.

### 3. Run batch inference

```bash
# Process all routes
python -m src.store_output data/data_routes_expanded.csv

# Process a single route only
python -m src.store_output data/data_routes_expanded.csv --route RESEARCH

# Custom output directory
python -m src.store_output data/data_routes_expanded.csv --output my_results
```

Results are written incrementally to `batch_outputs/` after each row so runs can be safely interrupted and resumed:

- `<stem>_results.csv` — summary per row (route, intent, response length, tool counts)
- `<stem>_details.jsonl` — full output per row (response text, suggestions, references, papers)
---

## Data schema

The dataset CSVs use the following columns:

| Column | Description |
|---|---|
| `query` | The user instruction to Writing Coach |
| `input` | The document or text the query acts on |
| `target` | Expected/reference output (from the source dataset) |
| `dataset` | Source dataset name (`OpenRewriteEval`, `CoEdiT`, `Kiwi`) |
| `route` | Ground-truth route label (`REVISE_SIMPLE`, `REVISE_RESEARCH`, `RESPOND`, `RESEARCH`) |

---

## Dependencies

See [requirements.txt](requirements.txt). Key packages:

- `langgraph` — Writing Coach V2 graph execution
- `openai` — Azure OpenAI client (data synthesis)
- `pandas`, `pyarrow` — data manipulation and Parquet support
- `huggingface_hub`, `datasets`, `fsspec` — public dataset access
- `fuzzywuzzy`, `Levenshtein` — fuzzy query matching during dataset merging
- `python-dotenv` — environment variable loading

