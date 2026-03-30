"""
Append edge case results to final_data/all_results.csv

Transforms data_outputs/round4/cor_edge_cases/manual_edge_cases_results.csv
into the all_results.csv format and appends it.
"""

import json
from pathlib import Path
import pandas as pd

# Paths
ROOT = Path(__file__).resolve().parent
EDGE_CASES_CSV = ROOT / "data_outputs/round4/cor_edge_cases/manual_edge_cases_results.csv"
EDGE_CASES_JSONL = ROOT / "data_outputs/round4/cor_edge_cases/manual_edge_cases_details.jsonl"
ALL_RESULTS_CSV = ROOT / "final_data/all_results.csv"
ALL_RESULTS_JSONL = ROOT / "final_data/all_results.jsonl"
EDGE_CASES_SEPARATE_CSV = ROOT / "final_data/cor_edge_cases_results.csv"
EDGE_CASES_SEPARATE_JSONL = ROOT / "final_data/cor_edge_cases_results.jsonl"

# Configuration
DATASET_SOURCE = "Synthetic_cor_edge_cases"
ROUTE_COLUMN = "route"  # will become both route_orch and route_intended


def read_existing_all_results():
    """Read existing all_results.csv to get the next row_id."""
    if not ALL_RESULTS_CSV.exists():
        return None, 0
    df = pd.read_csv(ALL_RESULTS_CSV, encoding="utf-8-sig")
    max_id = df["row_id"].max() if "row_id" in df.columns else 0
    return df, max_id


def read_edge_cases_source():
    """Read edge cases CSV and JSONL."""
    csv_df = pd.read_csv(EDGE_CASES_CSV, encoding="utf-8-sig")
    
    # Read JSONL for full output details
    details = {}
    if EDGE_CASES_JSONL.exists():
        with open(EDGE_CASES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    row_id = record.get("row_id_previous_folder", record.get("row_id"))
                    details[row_id] = record
    
    return csv_df, details


def build_output_field(row, details_record=None):
    """Build the nested 'output' JSON field from a row.

    The details JSONL stores fields directly (route, intent, reasoning, response,
    suggestions, references, research_papers, segments_count, tools_used) without
    a nested 'output' key. Read them directly and only override scalar metadata
    fields (route, intent, reasoning, segments_count) from the CSV if available.
    """
    output = {}

    # Read all content and metadata fields directly from the details record
    if details_record:
        for field in [
            "route", "intent", "reasoning",
            "response", "suggestions", "references", "research_papers",
            "segments_count", "tools_used",
        ]:
            if field in details_record:
                output[field] = details_record[field]

    # Override scalar metadata fields from CSV row (tools_used stays as list from JSONL)
    for field in ["route", "intent", "reasoning", "segments_count"]:
        val = row.get(field)
        if val is not None and pd.notna(val):
            output[field] = val

    return output


def transform_to_all_results_format(edge_csv_df, details_dict, start_id):
    """Transform edge cases into all_results format."""
    rows = []
    
    for idx, (_, row) in enumerate(edge_csv_df.iterrows()):
        new_id = start_id + idx + 1
        
        # Get details record for this row if available
        prev_id = int(row.get("row_id", idx + 1))
        details = details_dict.get(prev_id, {})
        
        output = build_output_field(row, details)
        
        transformed_row = {
            "row_id": new_id,
            "row_id_previous_folder": prev_id,
            "folder_source": "cor_edge_cases",
            "dataset_source": DATASET_SOURCE,
            "query": str(row.get("query", "")),
            "input_preview": str(details.get("input_preview", row.get("input_preview", "")))[:100],
            "input": str(row.get("input", "")),
            "route_orch": str(row.get(ROUTE_COLUMN, "UNKNOWN")),
            "route_intended": str(row.get(ROUTE_COLUMN, "UNKNOWN")),
            "output": json.dumps(output, ensure_ascii=False),
            "intent": str(details.get("intent", row.get("intent", ""))),
            "reasoning": str(details.get("reasoning", row.get("reasoning", ""))),
            "segments_count": details.get("segments_count", row.get("segments_count")),
            "tools_used": str(details.get("tools_used", [])),
            "response": details.get("response", ""),
            "suggestions": str(details.get("suggestions", [])),
            "references": str(details.get("references", [])),
            "research_papers": str(details.get("research_papers", [])),
        }
        rows.append(transformed_row)
    
    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("APPENDING EDGE CASES TO all_results.csv")
    print("=" * 80)
    
    # Read existing data
    existing_df, max_id = read_existing_all_results()
    print(f"\n[1] Existing all_results.csv: {max_id} rows, next row_id: {max_id + 1}")
    
    # Read edge cases
    edge_csv, details_dict = read_edge_cases_source()
    print(f"[2] Edge cases source: {len(edge_csv)} rows from {EDGE_CASES_CSV.relative_to(ROOT)}")
    
    # Transform
    transformed_df = transform_to_all_results_format(edge_csv, details_dict, max_id)
    print(f"[3] Transformed {len(transformed_df)} rows in all_results format")
    print(f"    Row IDs assigned: {transformed_df['row_id'].min()} - {transformed_df['row_id'].max()}")
    
    # Save separate edge cases copy (like kiwi_extra)
    transformed_df.to_csv(EDGE_CASES_SEPARATE_CSV, index=False, encoding="utf-8")
    print(f"[4] Saved separate copy: {EDGE_CASES_SEPARATE_CSV.relative_to(ROOT)}")
    
    # Save as JSONL too (restore list fields to native Python types)
    import ast
    list_columns = ["tools_used", "suggestions", "references", "research_papers"]
    with open(EDGE_CASES_SEPARATE_JSONL, "w", encoding="utf-8") as f:
        for _, row in transformed_df.iterrows():
            record = row.to_dict()
            for col in list_columns:
                val = record.get(col)
                if isinstance(val, str):
                    try:
                        record[col] = ast.literal_eval(val)
                    except Exception:
                        pass
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"    Saved JSONL: {EDGE_CASES_SEPARATE_JSONL.relative_to(ROOT)}")
    
    # Append to all_results.csv
    if existing_df is not None:
        # Align columns: add any new columns from transformed_df, fill missing with NaN
        for col in transformed_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        combined_df = pd.concat([existing_df, transformed_df[existing_df.columns]], ignore_index=True)
        combined_df.to_csv(ALL_RESULTS_CSV, index=False, encoding="utf-8")
        print(f"\n[5] Appended to all_results.csv: {len(combined_df)} total rows")
    else:
        transformed_df.to_csv(ALL_RESULTS_CSV, index=False, encoding="utf-8")
        print(f"\n[5] Created all_results.csv: {len(transformed_df)} rows")

    # Append to all_results.jsonl (native list types for list columns)
    existing_jsonl_rows: list[dict] = []
    if ALL_RESULTS_JSONL.exists():
        with open(ALL_RESULTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_jsonl_rows.append(json.loads(line))
    with open(ALL_RESULTS_JSONL, "w", encoding="utf-8") as f:
        for jsonl_row in existing_jsonl_rows:
            f.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")
        for _, row in transformed_df.iterrows():
            record = row.to_dict()
            for col in list_columns:
                val = record.get(col)
                if isinstance(val, str):
                    try:
                        record[col] = ast.literal_eval(val)
                    except Exception:
                        pass
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    total_jsonl = len(existing_jsonl_rows) + len(transformed_df)
    print(f"[6] Appended to all_results.jsonl: {total_jsonl} total rows")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print(f"""
1. Run evaluation on the new rows (uses resume support to skip existing rows):
   
   python -m src.evaluation.eval_pipeline \\
     --input final_data/all_results.csv \\
     --output-dir data_outputs/eval \\
     --save-local

2. Check the enriched results in:
   - data_outputs/eval/eval_*_all_results_enriched.csv
   - data_outputs/eval/eval_*_all_results_enriched.jsonl

3. Compute final metrics across all results (see compute_metrics.py)

The evaluation pipeline will:
- Skip rows already evaluated (from row IDs in existing JSONL)
- Only process the {len(transformed_df)} new rows
- Append results to existing output files
- Produce enriched CSV/JSONL with all rows merged
""")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
