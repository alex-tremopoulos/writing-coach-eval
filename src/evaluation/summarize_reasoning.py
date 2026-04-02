"""Send per-metric, per-score reasoning JSONs to an LLM for pattern summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

DATA_DIR = Path("eval_data/wcv2_one_prompt/route_intended/0327_1342/metrics_scores_combinations/item_level")
ALL_METRIC_NAMES = ["output_relevancy", "completeness", "correctness"]
SCORES = [0, 1, 2]
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat-2025-10-03")

SYSTEM_PROMPT = "You are an evaluation analyst. You will receive reasoning texts from an LLM-as-judge evaluation of a writing-coach system. Identify common patterns, recurring themes, and notable observations. Be concise and specific."

USER_PROMPTS = {
    0: (
        "Below are the reasoning explanations for all cases that received a score of 0 (worst) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns in these cases. Focus primarily on recurring issues "
        "and input characteristics that led to failure, but also note any positive aspects "
        "the system still exhibited despite the low score.\n\n"
        "Format your response as bullet points, grouped under '### Weaknesses' and '### Strengths' headings. "
        "Within each group, order bullet points by frequency — most common patterns first."
    ),
    1: (
        "Below are the reasoning explanations for all cases that received a score of 1 (partial) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns in these partially-successful cases. Cover both "
        "the recurring gaps that prevented a perfect score and the positive behaviors "
        "the system demonstrated.\n\n"
        "Format your response as bullet points, grouped under '### Weaknesses' and '### Strengths' headings. "
        "Within each group, order bullet points by frequency — most common patterns first."
    ),
    2: (
        "Below are the reasoning explanations for all cases that received a score of 2 (perfect) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns in these successful cases. Focus primarily on "
        "the good behaviors the system consistently exhibits, but also note any minor "
        "weaknesses or areas for improvement mentioned despite the perfect score.\n\n"
        "Format your response as bullet points, grouped under '### Strengths' and '### Weaknesses' headings. "
        "Within each group, order bullet points by frequency — most common patterns first."
    ),
}


def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def summarize(client: AzureOpenAI, metric: str, score: int, reasonings: str) -> str:
    metric_display = metric.replace("_", " ").title()
    user_msg = USER_PROMPTS[score].format(metric=metric_display, reasonings=reasonings)
    resp = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    return resp.choices[0].message.content


def infer_metrics(data_dir: Path) -> list[str]:
    """Infer which metrics have extracted reasoning files in the target directory."""
    return [
        metric
        for metric in ALL_METRIC_NAMES
        if any((data_dir / f"{metric}_items_score_{score}.json").exists() for score in SCORES)
    ]


def main(data_dir: Path | None = None, metrics: list[str] | None = None):
    resolved_dir = data_dir or DATA_DIR
    client = get_client()
    all_summaries = {}
    selected_metrics = metrics or infer_metrics(resolved_dir)

    for metric in selected_metrics:
        for score in SCORES:
            path = resolved_dir / f"{metric}_items_score_{score}.json"
            if not path.exists():
                print(f"SKIP (not found): {path}")
                continue

            entries = json.loads(path.read_text(encoding="utf-8"))
            if not entries:
                print(f"SKIP (empty): {path}")
                continue

            reasonings = "\n---\n".join(e["reasoning"] for e in entries)
            print(f"Summarizing {metric} score={score} ({len(entries)} entries)...")
            summary = summarize(client, metric, score, reasonings)
            all_summaries[f"{metric}_score_{score}"] = summary
            print(f"  Done.\n")

    out_path = resolved_dir / "reasoning_summaries.json"
    out_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"Wrote summaries -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize per-metric, per-score reasoning via LLM.")
    parser.add_argument("--data-dir", type=Path, default=None, help=f"Directory with score JSON files (default: {DATA_DIR})")
    parser.add_argument(
        "--metrics",
        "--metric",
        nargs="+",
        default=None,
        choices=ALL_METRIC_NAMES,
        metavar="METRIC",
        help=(
            "Subset of metrics to summarize (default: infer from files in data-dir). "
            f"Choices: {', '.join(ALL_METRIC_NAMES)}."
        ),
    )
    args = parser.parse_args()
    main(data_dir=args.data_dir, metrics=args.metrics)
