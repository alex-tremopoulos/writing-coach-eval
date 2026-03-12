"""
reassign_data.py

Reads all CSV and JSONL files from the 6 source folders under data_outputs/whole_input
and produces a single combined CSV and JSONL file at data_outputs/final_data/all_results.

Each row contains:
  - row_id                 : new 1-based id across the combined file
  - row_id_previous_folder : original row_id from the source file
  - folder_source          : source folder name
  - query, input_preview, input
  - route_orch             : route value returned by the orchestrator (from 'route' column)
  - route_intended         : intended route — same as route_orch except for manual overrides
  - output                 : nested JSON / dict with the model output fields

Manual overrides for route_intended
------------------------------------
  extra10           : row_id 3  → RESPOND,  row_id 5 → REVISE_RESEARCH
  revise_research_only : row_ids 31,131,133,120,121 → REVISE_RESEARCH
  respond_only      : row_ids 25,110,145 → RESPOND
                      all rows where route_orch == RESEARCH → RESPOND
  new21, research_only, revise_simple_only: no overrides
"""

import argparse
import ast
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INPUT_BASE = ROOT / "data_outputs" / "whole_input"
OUTPUT_DIR = ROOT / "final_data"
REF_CSV = ROOT / "data" / "data_routes_expanded.csv"

SOURCE_FOLDERS = [
    "extra10",
    "new21",
    "research_only",
    "respond_only",
    "revise_research_only",
    "revise_simple_only",
]

ROUTES = ["RESPOND", "RESEARCH", "REVISE_RESEARCH", "REVISE_SIMPLE"]

# Columns bundled into the nested 'output' field
OUTPUT_FIELDS = [
    "route",
    "intent",
    "reasoning",
    "response_length",
    "suggestions_count",
    "references_count",
    "papers_count",
    "segments_count",
    "tools_used",
    "response",
    "suggestions",
    "references",
    "research_papers",
]

# Manual overrides: folder → {row_id → intended_route}
# For respond_only, rows with route_orch == RESEARCH are also overridden to RESPOND.
INTENDED_OVERRIDES: dict[str, dict[int, str]] = {
    "extra10": {
        3: "RESPOND",
        5: "REVISE_RESEARCH",
    },
    "revise_research_only": {
        31: "REVISE_RESEARCH",
        131: "REVISE_RESEARCH",
        133: "REVISE_RESEARCH",
        120: "REVISE_RESEARCH",
        121: "REVISE_RESEARCH",
    },
    "respond_only": {
        25: "RESPOND",
        110: "RESPOND",
        145: "RESPOND",
    },
}

# respond_only rows with these prev_ids are the 9 extra respond cases
# added after the original 208-row dataset; they don't exist in
# data_routes_expanded.csv.
EXTRA_RESPOND_PREV_IDS = set(range(1, 10))   # prev_id 1–9

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl_files(folder: Path) -> pd.DataFrame:
    frames = []
    for jsonl_file in sorted(f for f in folder.glob("*.jsonl") if not f.stem.endswith("_extra")):
        records = []
        with open(jsonl_file, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        df = pd.DataFrame(records)
        frames.append(df)
        print(f"  [JSONL] {jsonl_file.relative_to(ROOT)}  →  {len(df)} rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_intended_route(row: pd.Series) -> str:
    """Return the intended route for a row, applying manual overrides."""
    folder = row["folder_source"]
    row_id = row["row_id_previous_folder"]
    route_orch = row["route_orch"]

    # respond_only: all RESEARCH rows → RESPOND
    if folder == "respond_only" and route_orch == "RESEARCH":
        return "RESPOND"

    overrides = INTENDED_OVERRIDES.get(folder, {})
    return overrides.get(row_id, route_orch)


def _parse_value(v):
    """Convert a value to a JSON-serialisable form.

    String values that look like Python lists/dicts (single-quoted, as written
    by pandas when saving complex objects to CSV) are first tried with
    ``json.loads`` and then ``ast.literal_eval`` so that they end up as proper
    nested objects in the final JSON rather than raw strings.
    """
    if pd.isna(v) if not isinstance(v, (list, dict)) else False:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s and s[0] in ("[", "{"):
            try:
                return json.loads(s)
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                return ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
    return v


def build_output_dict(row: pd.Series) -> dict:
    """Collect OUTPUT_FIELDS from a row into a dict, omitting NaN values."""
    result = {}
    for field in OUTPUT_FIELDS:
        if field not in row.index:
            continue
        v = row.get(field)
        if v is None:
            continue
        parsed = _parse_value(v)
        result[field] = parsed
    return result


def lookup_dataset_source(row: pd.Series, ref_df: pd.DataFrame) -> str | None:
    """Determine the dataset_source for a row.

    Priority:
    1. Folder-based overrides (extra10, new21, extra respond cases).
    2. Match against ``data_routes_expanded.csv`` by *query* (exact first,
       then prefix) and *input* text (first 100 chars) to disambiguate
       duplicate queries.
    """
    folder = row["folder_source"]
    prev_id = row["row_id_previous_folder"]

    # ----- folder-based overrides ------------------------------------------
    if folder == "extra10":
        return "extra_10_manu"
    if folder == "new21":
        return "extra_21_alex"
    if folder == "extra200kiwi":
        return "Kiwi"
    if folder == "respond_only" and prev_id in EXTRA_RESPOND_PREV_IDS:
        return "extra_respond_alex"

    # ----- match against ref ------------------------------------------------
    query = str(row.get("query", ""))
    full_input = str(row.get("input", ""))

    # Strategy 1: exact query match
    candidates = ref_df[ref_df["query"] == query]

    # Strategy 2: prefix match (handles truncated queries in ref)
    if len(candidates) == 0:
        candidates = ref_df[
            ref_df["query"].apply(
                lambda rq: query.startswith(str(rq)) or str(rq).startswith(query)
            )
        ]

    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates.iloc[0]["dataset"]

    # Multiple candidates — disambiguate by input prefix
    input_prefix = full_input[:100].strip()
    for _, cand in candidates.iterrows():
        ref_prefix = str(cand["input"])[:100].strip()
        if input_prefix == ref_prefix:
            return cand["dataset"]

    return None


def write_csv(df: pd.DataFrame, path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    header = not (append and path.exists())
    df.to_csv(path, index=False, mode=mode, header=header)
    action = "appended" if mode == "a" else "wrote"
    print(f"  → {action} {len(df)} rows  to  {path.relative_to(ROOT)}")


def write_jsonl(records: list[dict], path: Path, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append and path.exists() else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    action = "appended" if mode == "a" else "wrote"
    print(f"  → {action} {len(records)} records to  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reassign and combine writing-coach output data."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        metavar="FOLDER",
        help=(
            "One or more source folder names under data_outputs/whole_input to process. "
            "Defaults to all known folders. Example: --folders extra200kiwi"
        ),
    )
    parser.add_argument(
        "--ref-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to the reference CSV used for dataset_source lookup. "
            "Relative paths are resolved from the repo root. "
            f"Default: {REF_CSV.relative_to(ROOT)}"
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output CSV/JSONL files instead of overwriting them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    folders_to_process = args.folders if args.folders is not None else SOURCE_FOLDERS
    ref_csv_path = Path(args.ref_csv) if args.ref_csv else REF_CSV
    if not ref_csv_path.is_absolute():
        ref_csv_path = ROOT / ref_csv_path

    # ---- 1. Read all source folders ----------------------------------------
    print("=" * 60)
    print("Reading source folders …")
    print(f"  folders  : {folders_to_process}")
    print(f"  ref-csv  : {ref_csv_path.relative_to(ROOT)}")
    print(f"  append   : {args.append}")
    print("=" * 60)

    all_frames: list[pd.DataFrame] = []

    for folder_name in folders_to_process:
        folder = INPUT_BASE / folder_name
        if not folder.exists():
            print(f"[WARN] folder not found, skipping: {folder}")
            continue

        print(f"\n{folder_name}/")
        df = read_jsonl_files(folder)
        if not df.empty:
            df["folder_source"] = folder_name
            all_frames.append(df)

    if not all_frames:
        print("\n[ERROR] No CSV data found. Aborting.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows collected: {len(combined)}")

    # ---- 2. Validate --------------------------------------------------------
    if "route" not in combined.columns:
        print("[ERROR] 'route' column not found. Aborting.")
        return

    # ---- 3. Normalise input column (new21 uses 'input_text') ---------------
    if "input_text" in combined.columns:
        # Ensure 'input' exists before combining; some sources may only have 'input_text'
        if "input" not in combined.columns:
            combined["input"] = pd.NA
        combined["input"] = combined["input"].combine_first(combined["input_text"])
        combined.drop(columns=["input_text"], inplace=True)

    # ---- 4. Build route_orch and route_intended -----------------------------
    combined.rename(columns={"row_id": "row_id_previous_folder"}, inplace=True)
    combined["route_orch"] = combined["route"]

    combined["route_intended"] = combined.apply(compute_intended_route, axis=1)

    override_count = (combined["route_intended"] != combined["route_orch"]).sum()
    print(f"route_intended overrides applied: {override_count}")

    # ---- 4b. Build dataset_source -------------------------------------------
    print("\nLooking up dataset_source …")
    ref_df = pd.read_csv(ref_csv_path)
    combined["dataset_source"] = combined.apply(
        lookup_dataset_source, axis=1, ref_df=ref_df
    )
    ds_matched = combined["dataset_source"].notna().sum()
    print(f"  dataset_source assigned: {ds_matched} / {len(combined)}")
    if ds_matched < len(combined):
        missing = combined[combined["dataset_source"].isna()]
        print("  [WARN] Unmatched rows:")
        for _, r in missing.iterrows():
            print(f"    prev_id={r['row_id_previous_folder']}, "
                  f"folder={r['folder_source']}, "
                  f"query={str(r['query'])[:60]}")

    # ---- 5. Build 'output' column ------------------------------------------
    combined["output"] = combined.apply(
        lambda row: json.dumps(build_output_dict(row), ensure_ascii=False),
        axis=1,
    )

    # ---- 6. Select and order final columns ----------------------------------
    keep_cols = [
        "row_id_previous_folder",
        "folder_source",
        "dataset_source",
        "query",
        "input_preview",
        "input",
        "route_orch",
        "route_intended",
        "output",
    ]
    # All OUTPUT_FIELDS columns stay; just make sure 'route' (the raw column)
    # doesn't clash — we already have route_orch for that.
    final = combined.drop(columns=["route"], errors="ignore")

    # Place keep_cols first, then the output fields, then anything remaining
    output_fields_present = [c for c in OUTPUT_FIELDS if c in final.columns and c not in keep_cols]
    extra_cols = [c for c in final.columns if c not in keep_cols and c not in output_fields_present]
    final = final[keep_cols + output_fields_present + extra_cols]

    # New 1-based row_id (continue from existing max when appending)
    start_id = 1
    if args.append and (OUTPUT_DIR / "all_results.csv").exists():
        try:
            existing_ids = pd.read_csv(OUTPUT_DIR / "all_results.csv", usecols=["row_id"])
            start_id = int(existing_ids["row_id"].max()) + 1
        except Exception:
            pass
    final.insert(0, "row_id", range(start_id, start_id + len(final)))
    final.reset_index(drop=True, inplace=True)

    # ---- 7. Write CSV -------------------------------------------------------
    print("\n" + "=" * 60)
    print("Writing combined output …")
    print("=" * 60)

    csv_path = OUTPUT_DIR / "all_results.csv"
    write_csv(final, csv_path, append=args.append)

    # ---- 8. Write JSONL (output field as nested dict, not string) ----------
    jsonl_path = OUTPUT_DIR / "all_results.jsonl"
    records = []
    for _, row in final.iterrows():
        rec = row.to_dict()
        rec["output"] = json.loads(rec["output"])   # un-stringify for JSONL
        records.append(rec)
    write_jsonl(records, jsonl_path, append=args.append)

    # ---- 9. Summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total rows : {len(final)}")
    print(f"\n  route_orch distribution:")
    for route, count in final["route_orch"].value_counts().items():
        print(f"    {route:<20} {count:>5}")
    print(f"\n  route_intended distribution:")
    for route, count in final["route_intended"].value_counts().items():
        print(f"    {route:<20} {count:>5}")
    print(f"\n  dataset_source distribution:")
    for ds, count in final["dataset_source"].value_counts(dropna=False).items():
        print(f"    {str(ds):<40} {count:>5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
