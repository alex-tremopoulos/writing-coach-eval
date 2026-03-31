"""Extract per-metric, per-score reasoning from enriched eval CSV into JSON files."""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

INPUT_CSV = Path("eval_data/wcv2_one_prompt/route_intended/0327_1342/eval_20260327_134212_all_results_enriched.csv")
OUTPUT_DIR = INPUT_CSV.parent


def main():
    df = pd.read_csv(INPUT_CSV)

    # --- Overall metric-level buckets ---
    buckets = defaultdict(list)
    # --- Item-level buckets ---
    item_buckets = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row["eval_verdicts_json"]):
            continue
        verdicts = json.loads(row["eval_verdicts_json"])
        for v in verdicts:
            metric_key = v["metric_name"].lower().replace(" ", "_")
            buckets[(metric_key, v["score"])].append({
                "row_id": row["row_id"],
                "score": v["score"],
                "reasoning": v["reasoning"],
            })
            for item in v.get("evaluation_items", []):
                item_buckets[(metric_key, item["score"])].append({
                    "row_id": row["row_id"],
                    "item_name": item["item_name"],
                    "score": item["score"],
                    "reasoning": item["reasoning"],
                })

    # Write overall files
    for (metric, score), entries in buckets.items():
        filename = f"{metric}_score_{score}.json"
        out_path = OUTPUT_DIR / filename
        out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Wrote {len(entries):>3} entries -> {out_path}")

    # Write item-level files
    items_dir = OUTPUT_DIR / "item_level"
    items_dir.mkdir(exist_ok=True)
    for (metric, score), entries in item_buckets.items():
        filename = f"{metric}_items_score_{score}.json"
        out_path = items_dir / filename
        out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Wrote {len(entries):>3} item entries -> {out_path}")


if __name__ == "__main__":
    main()
