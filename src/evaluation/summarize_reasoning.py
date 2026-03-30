"""Send per-metric, per-score reasoning JSONs to an LLM for pattern summaries."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

DATA_DIR = Path("eval_data/wcv2_one_prompt/route_intended/0327_1342")
METRICS = ["output_relevancy", "completeness", "correctness"]
SCORES = [0, 1, 2]
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-chat-2025-10-03")

SYSTEM_PROMPT = "You are an evaluation analyst. You will receive reasoning texts from an LLM-as-judge evaluation of a writing-coach system. Identify common patterns, recurring themes, and notable observations. Be concise and specific."

USER_PROMPTS = {
    0: (
        "Below are the reasoning explanations for all cases that received a score of 0 (worst) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns explaining why these cases scored poorly. "
        "What recurring issues or input characteristics led to failure?"
    ),
    1: (
        "Below are the reasoning explanations for all cases that received a score of 1 (partial) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns among these partially-successful cases. "
        "What recurring gaps or input characteristics prevented a perfect score?"
    ),
    2: (
        "Below are the reasoning explanations for all cases that received a score of 2 (perfect) "
        "on the '{metric}' metric.\n\n{reasonings}\n\n"
        "Summarize the common patterns in these successful cases. "
        "What good behaviors does the system consistently exhibit?"
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


def main():
    client = get_client()
    all_summaries = {}

    for metric in METRICS:
        for score in SCORES:
            path = DATA_DIR / f"{metric}_score_{score}.json"
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

    out_path = DATA_DIR / "reasoning_summaries.json"
    out_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(f"Wrote summaries -> {out_path}")


if __name__ == "__main__":
    main()
