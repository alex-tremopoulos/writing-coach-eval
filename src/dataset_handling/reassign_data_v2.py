"""
reassign_data_v2.py

Reads all JSONL files from source folders under a configurable input base
and produces a single combined CSV and JSONL file at final_data/.

Gold-standard mapping
---------------------
This version loads route_intended from an existing gold-standard file
(default: final_data/all_results.csv) keyed by (row_id_previous_folder,
folder_source).  Rows not found in the gold standard default to route_orch.

The gold-standard key uses row_id, row_id_previous_folder and folder_source
to guarantee correct matching.

Output filenames include the current month and day (MMDD).
"""

import argparse
import ast
import json
from datetime import date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_BASE = ROOT / "data_outputs" / "round4"
OUTPUT_DIR = ROOT / "final_data"
REF_CSV = ROOT / "data" / "data_routes_expanded.csv"
GOLD_STANDARD = ROOT / "final_data" / "all_results.csv"
DOMAINS_CSV = ROOT / "final_data" / "all_results_v2_domains.csv"

DEFAULT_SOURCE_FOLDERS = [
    "extra10",
    "new21",
    "research_only",
    "respond_only",
    "revise_research_only",
    "revise_simple_only",
    "extra167kiwi",
    "cor_edge_cases",
]

ROUTES = ["RESPOND", "RESEARCH", "REVISE_RESEARCH", "REVISE_SIMPLE"]

# Folders that always map to a fixed dataset_source value regardless of query matching.
# This mirrors the folder-based priority logic from the original reassign_data.py.
FOLDER_DATASET_SOURCE: dict[str, str] = {
    "extra10": "extra_10_manu",
    "new21": "extra_21_alex",
    "extra167kiwi": "Kiwi",
    "cor_edge_cases": "Synthetic_cor_edge_cases",
}

# respond_only rows whose row_id_previous_folder falls in this range are the 9
# extra respond cases added after the original dataset; they don't exist in
# data_routes_expanded.csv.
EXTRA_RESPOND_PREV_IDS = set(range(1, 10))  # prev_id 1–9

# Rows that accept two routes: if route_orch is one of the listed values,
# route_intended is set to match route_orch (both are considered correct).
# Applied after the gold-standard lookup.
DUAL_ACCEPTABLE_ROUTES: dict[tuple[int, str], frozenset] = {
    (110, "respond_only"):        frozenset({"RESPOND", "RESEARCH"}),
    (111, "respond_only"):        frozenset({"RESPOND", "RESEARCH"}),
    (31,  "revise_research_only"): frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
    (132, "revise_research_only"): frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
    (133, "revise_research_only"): frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
    (15,  "extra167kiwi"):         frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
    (60,  "extra167kiwi"):         frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
    (62,  "extra167kiwi"):         frozenset({"REVISE_SIMPLE", "REVISE_RESEARCH"}),
}

# Hard manual overrides: these take the highest priority, overriding the gold
# standard and dual-acceptable logic.  The route_intended is always set to the
# specified value regardless of what the gold standard or route_orch says.
MANUAL_OVERRIDES: dict[tuple[int, str], str] = {
    (141, "respond_only"): "RESPOND",
    (30,  "extra167kiwi"): "RESPOND",
    (31,  "extra167kiwi"): "RESPOND",
    (63,  "extra167kiwi"): "RESPOND",
}

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

# ---------------------------------------------------------------------------
# Gold-standard mapping
# ---------------------------------------------------------------------------

def load_gold_standard(path: Path) -> dict[tuple[int, str], str]:
    """Load gold-standard mapping: (row_id_previous_folder, folder_source) → route_intended.

    Uses row_id, row_id_previous_folder, and folder_source from the gold-standard
    file to guarantee correct matching.  The lookup key is
    (row_id_previous_folder, folder_source).
    """
    if not path.exists():
        print(f"[WARN] Gold standard not found at {path}. "
              "route_intended will default to route_orch.")
        return {}

    df = pd.read_csv(path)
    required = {"row_id", "row_id_previous_folder", "folder_source", "route_intended"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Gold standard missing columns: {missing}. "
              "route_intended will default to route_orch.")
        return {}

    mapping: dict[tuple[int, str], str] = {}
    for _, row in df.iterrows():
        key = (int(row["row_id_previous_folder"]), str(row["folder_source"]))
        mapping[key] = str(row["route_intended"])

    print(f"  Gold standard loaded: {len(mapping)} entries from {path}")
    return mapping


def compute_intended_route(
    row: pd.Series,
    gold_mapping: dict[tuple[int, str], str],
) -> str:
    """Return the intended route, applying overrides in priority order:

    1. MANUAL_OVERRIDES — hard-coded corrections, always win.
    2. DUAL_ACCEPTABLE_ROUTES — if route_orch is one of the two acceptable
       values for this row, set route_intended = route_orch so they match.
    3. Gold-standard mapping keyed by (row_id_previous_folder, folder_source).
    4. Fallback: route_orch itself (for rows absent from the gold standard).
    """
    key = (int(row["row_id_previous_folder"]), str(row["folder_source"]))

    # 1. Hard manual overrides
    if key in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[key]

    # 2. Dual-acceptable routes
    acceptable = DUAL_ACCEPTABLE_ROUTES.get(key)
    if acceptable and str(row["route_orch"]) in acceptable:
        return str(row["route_orch"])

    # 3. Gold standard, 4. fallback
    return gold_mapping.get(key, row["route_orch"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_jsonl_files(folder: Path) -> pd.DataFrame:
    frames = []
    for jsonl_file in sorted(folder.glob("*.jsonl")):
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


def _parse_value(v):
    """Convert a value to a JSON-serialisable form."""
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
    1. Folder-based fixed mappings (extra10, new21, extra167kiwi, cor_edge_cases).
    2. respond_only rows with prev_id 1-9 → extra_respond_alex.
    3. Match against data_routes_expanded.csv by query and input prefix.
    """
    folder = row["folder_source"]
    prev_id = int(row["row_id_previous_folder"])

    # Fixed folder mappings
    if folder in FOLDER_DATASET_SOURCE:
        return FOLDER_DATASET_SOURCE[folder]

    # respond_only extra rows (not in the reference CSV)
    if folder == "respond_only" and prev_id in EXTRA_RESPOND_PREV_IDS:
        return "extra_respond_alex"

    query = str(row.get("query", ""))
    full_input = str(row.get("input", ""))

    # Strategy 1: exact query match
    candidates = ref_df[ref_df["query"] == query]

    # Strategy 2: prefix match
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
        description="Reassign and combine writing-coach output data (v2)."
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        metavar="FOLDER",
        help=(
            "Source folder names to process (looked up under --input-base). "
            "Defaults to: " + ", ".join(DEFAULT_SOURCE_FOLDERS)
        ),
    )
    parser.add_argument(
        "--input-base",
        default=None,
        metavar="PATH",
        help=(
            "Base directory containing the source folders. "
            f"Default: {DEFAULT_INPUT_BASE.relative_to(ROOT)}"
        ),
    )
    parser.add_argument(
        "--gold-standard",
        default=None,
        metavar="PATH",
        help=(
            "Path to the gold-standard CSV with route_intended values. "
            f"Default: {GOLD_STANDARD.relative_to(ROOT)}"
        ),
    )
    parser.add_argument(
        "--ref-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to the reference CSV used for dataset_source lookup. "
            f"Default: {REF_CSV.relative_to(ROOT)}"
        ),
    )
    parser.add_argument(
        "--domains-csv",
        default=None,
        metavar="PATH",
        help=(
            "Path to a CSV containing query_domain and input_domain columns, "
            "joined by (row_id_previous_folder, folder_source). "
            f"Default: {DOMAINS_CSV.relative_to(ROOT)}"
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

    folders_to_process = args.folders if args.folders is not None else DEFAULT_SOURCE_FOLDERS
    input_base = Path(args.input_base) if args.input_base else DEFAULT_INPUT_BASE
    if not input_base.is_absolute():
        input_base = ROOT / input_base
    gold_standard_path = Path(args.gold_standard) if args.gold_standard else GOLD_STANDARD
    if not gold_standard_path.is_absolute():
        gold_standard_path = ROOT / gold_standard_path
    ref_csv_path = Path(args.ref_csv) if args.ref_csv else REF_CSV
    if not ref_csv_path.is_absolute():
        ref_csv_path = ROOT / ref_csv_path
    domains_csv_path = Path(args.domains_csv) if args.domains_csv else DOMAINS_CSV
    if not domains_csv_path.is_absolute():
        domains_csv_path = ROOT / domains_csv_path

    date_suffix = date.today().strftime("%m%d")

    # ---- 1. Load gold standard ---------------------------------------------
    print("=" * 60)
    print("Configuration")
    print(f"  folders        : {folders_to_process}")
    print(f"  input-base     : {input_base}")
    print(f"  gold-standard  : {gold_standard_path}")
    print(f"  ref-csv        : {ref_csv_path}")
    print(f"  domains-csv    : {domains_csv_path}")
    print(f"  append         : {args.append}")
    print(f"  date suffix    : {date_suffix}")
    print("=" * 60)

    gold_mapping = load_gold_standard(gold_standard_path)

    # ---- 2. Read all source folders ----------------------------------------
    print("\nReading source folders …")
    all_frames: list[pd.DataFrame] = []

    for folder_name in folders_to_process:
        folder = input_base / folder_name
        if not folder.exists():
            print(f"[WARN] folder not found, skipping: {folder}")
            continue

        print(f"\n{folder_name}/")
        df = read_jsonl_files(folder)
        if not df.empty:
            df["folder_source"] = folder_name
            all_frames.append(df)

    if not all_frames:
        print("\n[ERROR] No data found. Aborting.")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nTotal rows collected: {len(combined)}")

    # ---- 3. Validate --------------------------------------------------------
    if "route" not in combined.columns:
        print("[ERROR] 'route' column not found. Aborting.")
        return

    # ---- 4. Normalise input column (some sources use 'input_text') ----------
    if "input_text" in combined.columns:
        if "input" not in combined.columns:
            combined["input"] = pd.NA
        combined["input"] = combined["input"].combine_first(combined["input_text"])
        combined.drop(columns=["input_text"], inplace=True)

    # ---- 5. Build route_orch and route_intended -----------------------------
    combined.rename(columns={"row_id": "row_id_previous_folder"}, inplace=True)
    combined["route_orch"] = combined["route"]

    combined["route_intended"] = combined.apply(
        compute_intended_route, axis=1, gold_mapping=gold_mapping,
    )

    override_count = (combined["route_intended"] != combined["route_orch"]).sum()
    gold_matched = sum(
        1 for _, row in combined.iterrows()
        if (int(row["row_id_previous_folder"]), str(row["folder_source"])) in gold_mapping
    )
    gold_unmatched = len(combined) - gold_matched
    print(f"\nGold standard matches : {gold_matched} / {len(combined)}")
    if gold_unmatched:
        print(f"  Rows not in gold standard (using route_orch): {gold_unmatched}")
    print(f"route_intended != route_orch : {override_count}")

    # ---- 6. Build dataset_source --------------------------------------------
    print("\nLooking up dataset_source …")
    ref_df = pd.read_csv(ref_csv_path)
    combined["dataset_source"] = combined.apply(
        lookup_dataset_source, axis=1, ref_df=ref_df,
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

    # ---- 7. Join query_domain and input_domain from domains CSV --------------
    if domains_csv_path.exists():
        domains_df = pd.read_csv(
            domains_csv_path,
            usecols=["row_id_previous_folder", "folder_source", "query_domain", "input_domain"],
        )
        combined = combined.merge(domains_df, on=["row_id_previous_folder", "folder_source"], how="left")
        matched = combined["query_domain"].notna().sum()
        print(f"\n  domain columns joined: {matched} / {len(combined)} rows matched")
        if matched < len(combined):
            print("  [WARN] Some rows have no domain data — query_domain/input_domain will be NaN.")
    else:
        print(f"\n[WARN] domains CSV not found at {domains_csv_path}. "
              "query_domain and input_domain will be absent.")
        combined["query_domain"] = None
        combined["input_domain"] = None

    # ---- 8. Build 'output' column -------------------------------------------
    combined["output"] = combined.apply(
        lambda row: json.dumps(build_output_dict(row), ensure_ascii=False),
        axis=1,
    )

    # ---- 9. Select and order final columns ----------------------------------
    keep_cols = [
        "row_id_previous_folder",
        "folder_source",
        "dataset_source",
        "query",
        "query_domain",
        "input_preview",
        "input",
        "input_domain",
        "route_orch",
        "route_intended",
        "output",
    ]
    final = combined.drop(columns=["route"], errors="ignore")

    output_fields_present = [
        c for c in OUTPUT_FIELDS if c in final.columns and c not in keep_cols
    ]
    extra_cols = [
        c for c in final.columns
        if c not in keep_cols and c not in output_fields_present
    ]
    final = final[keep_cols + output_fields_present + extra_cols]

    # New 1-based row_id
    csv_path = OUTPUT_DIR / f"all_results_{date_suffix}.csv"
    # (step numbering follows the added join step above)
    start_id = 1
    if args.append and csv_path.exists():
        try:
            existing_ids = pd.read_csv(csv_path, usecols=["row_id"])
            start_id = int(existing_ids["row_id"].max()) + 1
        except Exception:
            pass
    final.insert(0, "row_id", range(start_id, start_id + len(final)))
    final.reset_index(drop=True, inplace=True)

    # ---- 10. Write CSV ------------------------------------------------------
    print("\n" + "=" * 60)
    print("Writing combined output …")
    print("=" * 60)

    write_csv(final, csv_path, append=args.append)

    # ---- 11. Write JSONL (output field as nested dict, not string) ----------
    jsonl_path = OUTPUT_DIR / f"all_results_{date_suffix}.jsonl"
    records = []
    for _, row in final.iterrows():
        rec = row.to_dict()
        rec["output"] = json.loads(rec["output"])
        records.append(rec)
    write_jsonl(records, jsonl_path, append=args.append)

    # ---- 12. Summary --------------------------------------------------------
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
