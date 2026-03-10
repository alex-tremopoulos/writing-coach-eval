# writing-coach-eval

Evaluation workspace for Writing Coach V2. The repository is mainly built to prepare evaluation inputs, run the system in batch, normalize outputs into a reviewable dataset, and score those outputs with a dynamic rubrics pipeline.

## What This Repository Covers

The evaluation flow has four practical stages:

1. Build or extend route-labeled evaluation data.
2. Run Writing Coach V2 on each row and capture routing plus model outputs.
3. Consolidate those outputs into a single evaluation dataset with intended-route labels and final returned text.
4. Score the results with a two-stage LLM evaluation pipeline.

Writing Coach routes each example into one of four task types:

| Route | Purpose |
|---|---|
| `REVISE_SIMPLE` | Edit text without external research |
| `REVISE_RESEARCH` | Revise text and add evidence or citations |
| `RESPOND` | Answer, summarize, compare, or discuss based on given context |
| `RESEARCH` | Search literature and return evidence-grounded findings |

## Evaluation Pipeline

The core evaluation logic lives in `src/evaluation/eval_pipeline.py`.

It uses a two-stage asynchronous process:

1. **Rubrics generation**: given a row's `query`, `input`, and intended `route`, the model generates task-specific rubric criteria.
2. **Rubrics judging**: a second call scores the system output against those generated criteria.

The pipeline is designed for batch work rather than one-off inspection:

- Async concurrency with a semaphore limit
- Incremental CSV and JSONL writing for resume support
- Route filtering and row limits for targeted runs
- Generator-only mode for inspecting rubric quality without running the judge
- Enriched outputs that merge evaluation results back onto the original dataset rows

The default evaluation input is `final_data/all_results_with_final_text.csv`, which contains the normalized system output text that should be judged.

## Prompt Structure And Strategy

Prompt assembly is handled by `src/evaluation/prompt_loader.py`.

The evaluation prompts are intentionally split into two layers:

- `src/prompts/rubrics_prompt.txt`: generates dynamic rubrics for the specific task instance
- `src/prompts/rubrics_judge_prompt.txt`: scores the returned output against those rubrics

Each prompt file uses a simple block structure:

```text
{% block system %}
...
{% endblock %}

{% block prompt %}
...
{% endblock %}
```

That separation keeps the evaluation pipeline maintainable: stable evaluator behavior stays in the system block, while row-specific inputs are injected into the prompt block.

The rubric-generation strategy combines four sources of context:

- The user command
- The original input text
- Route-specific behavior guidance from `src/constants/route_prompts.py`
- Shared evaluation dimensions from `src/constants/metrics_definitions.py`

Current metrics are deliberately narrow: the generated rubrics focus mainly on output relevancy and completeness. This keeps scoring aligned to the user request and route behavior rather than drifting into unrelated quality dimensions.

The route-constraint hook exists in `src/constants/route_constraints.py`. It is currently minimal, but the structure is already in place for tightening route-specific rubric rules later.

## Important Dataset Handling

Most dataset work lives in `src/dataset_handling/`. The important pieces for evaluation are:

- `merge_public_datasets.py`: builds the initial labeled base set from public sources
- `synthesize_domain_data.py`: expands coverage with synthetic domain-adapted examples
- `generate_eval_dataset.py`: creates the focused 21-row evaluation set used for Writing Coach examples
- `reassign_data.py`: consolidates batch outputs from multiple route folders into a single `all_results` dataset and applies intended-route overrides where needed

For evaluation, `reassign_data.py` matters because it turns scattered run outputs into a single table with:

- A new global `row_id`
- The original source folder and prior row id
- `route_orch` and `route_intended`
- A nested `output` payload with response data, suggestions, references, and tool usage
- `dataset_source` mapping back to the original dataset when possible

After consolidation, `src/build_final_text.py` computes `returned_final_text`. For `RESPOND` and `RESEARCH`, this is the response text directly. For revision routes, it applies all proposed suggestions to the original input so the evaluator scores the effective final text rather than the raw suggestion list.

## Running The Flow

### Setup

Requirements:

- Python 3.12
- Azure OpenAI credentials for synthesis and evaluation

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Environment variables:

```env
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_API_VERSION=2025-01-01-preview
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
```

### 1. Run batch inference

```bash
python -m src.store_output data/data_routes_expanded.csv
```

This runs Writing Coach V2 over the dataset and stores routing decisions plus structured outputs.

### 2. Consolidate outputs for evaluation

```bash
python src/dataset_handling/reassign_data.py
python src/build_final_text.py
```

This produces:

- `final_data/all_results.csv`
- `final_data/all_results.jsonl`
- `final_data/all_results_with_final_text.csv`
- `final_data/all_results_with_final_text.jsonl`

### 3. Run the evaluation pipeline

```bash
python -m src.evaluation.eval_pipeline --input final_data/all_results_with_final_text.csv
```

Useful options:

- `--routes RESEARCH RESPOND`
- `--limit 10`
- `--concurrency 3`
- `--generator-only`

Evaluation outputs are written to `data_outputs/eval/` as:

- `*_results.csv`: per-row summary scores and packed JSON fields
- `*_details.jsonl`: full generator and judge outputs
- `*_all_results_enriched.csv`
- `*_all_results_enriched.jsonl`

## Repository Pointers

- `src/store_output.py`: batch inference runner for Writing Coach V2
- `src/dataset_handling/`: dataset construction and consolidation scripts
- `src/build_final_text.py`: derives final text for evaluation
- `src/evaluation/eval_pipeline.py`: async rubrics evaluation pipeline
- `src/evaluation/prompt_loader.py`: prompt parsing and slot injection
- `src/prompts/`: evaluation prompt templates
- `src/constants/`: route prompts, route constraints, and metric definitions

## Dependencies

Key packages:

- `langgraph` for Writing Coach graph execution
- `langchain` and `langchain-openai` for async evaluation calls
- `openai` for Azure OpenAI access
- `pandas` and `pyarrow` for dataset handling
- `datasets` and `huggingface_hub` for public data ingestion
- `python-dotenv` for environment loading

