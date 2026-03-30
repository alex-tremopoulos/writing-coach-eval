"""Extract per-metric, per-score reasoning from enriched eval CSV into JSON files."""

import json
import pandas as pd
from collections import defaultdict
from pathlib import Path

INPUT_CSV = Path("eval_data/wcv2_one_prompt/route_intended/0327_1342/eval_20260327_134212_all_results_enriched.csv")
OUTPUT_DIR = INPUT_CSV.parent


def main():
    df = pd.read_csv(INPUT_CSV)
    # bucket: (metric_name, score) -> list of {row_id, score, reasoning}
    buckets = defaultdict(list)

    for _, row in df.iterrows():
        if pd.isna(row["eval_verdicts_json"]):
            continue
        verdicts = json.loads(row["eval_verdicts_json"])
        for v in verdicts:
            buckets[(v["metric_name"], v["score"])].append({
                "row_id": row["row_id"],
                "score": v["score"],
                "reasoning": v["reasoning"],
            })

    for (metric, score), entries in buckets.items():
        filename = f"{metric.lower().replace(' ', '_')}_score_{score}.json"
        out_path = OUTPUT_DIR / filename
        out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Wrote {len(entries):>3} entries -> {out_path}")


if __name__ == "__main__":
    main()
