"""
data_pipeline.py

End-to-end evaluation data pipeline:

  1. Runs store_output.py for each configured input job, writing JSONL outputs
     into subfolders of a versioned round directory (data_outputs/round_DDMM).

  2. Runs reassign_data_v2.py against that round directory to produce the
     combined final_data/all_results_MMDD.csv and .jsonl files.

Default jobs
------------
  Input CSV                              Subfolder         Route filter
  data/alex-9-extra-responds.csv         respond_only      (none)
  data/data_routes_expanded.csv          research_only     RESEARCH
  data/data_routes_expanded.csv          respond_only      RESPOND
  data/data_routes_expanded.csv          revise_research_only  REVISE_RESEARCH
  data/data_routes_expanded.csv          revise_simple_only    REVISE_SIMPLE
  data/manu-10-extra-queries.csv         extra10           (none)
  data/manual_edge_cases_cor.csv         cor_edge_cases    (none)
  data/wc2_eval_21.csv                   new21             (none)
  data/data_routes_kiwi_selected167.csv  extra167kiwi      (none)

Custom jobs
-----------
Pass --jobs as JSON (list of objects) to override the default list:

  --jobs '[{"csv":"data/my.csv","folder":"my_folder"},
           {"csv":"data/other.csv","folder":"other","route":"RESEARCH"}]'

  Each object must have "csv" and "folder"; "route" is optional.
  Paths for "csv" are resolved relative to the repo root if not absolute.

Usage
-----
  python -m src.dataset_handling.data_pipeline
  python -m src.dataset_handling.data_pipeline --round-dir data_outputs/round_custom
  python -m src.dataset_handling.data_pipeline --jobs '[...]'
  python -m src.dataset_handling.data_pipeline --skip-store   # only run reassign step
  python -m src.dataset_handling.data_pipeline --skip-reassign  # only run store steps
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root so WC_APP_SRC (and other vars) are available
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ---------------------------------------------------------------------------
# Repo root (two levels up from this file: src/dataset_handling/ → repo root)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Default job definitions
# ---------------------------------------------------------------------------
DEFAULT_JOBS = [
    {"csv": "data/alex-9-extra-responds.csv",         "folder": "respond_only"},
    {"csv": "data/data_routes_expanded.csv",           "folder": "research_only",        "route": "RESEARCH"},
    {"csv": "data/data_routes_expanded.csv",           "folder": "respond_only",         "route": "RESPOND"},
    {"csv": "data/data_routes_expanded.csv",           "folder": "revise_research_only", "route": "REVISE_RESEARCH"},
    {"csv": "data/data_routes_expanded.csv",           "folder": "revise_simple_only",   "route": "REVISE_SIMPLE"},
    {"csv": "data/manu-10-extra-queries.csv",          "folder": "extra10"},
    {"csv": "data/manual_edge_cases_cor.csv",          "folder": "cor_edge_cases"},
    {"csv": "data/wc2_eval_21.csv",                    "folder": "new21"},
    {"csv": "data/data_routes_kiwi_selected167.csv",   "folder": "extra167kiwi"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_csv(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def _run(cmd: list[str], label: str, extra_env: dict[str, str] | None = None) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' exited with code {result.returncode}. Aborting.")
        sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def run_store_output(jobs: list[dict], round_dir: Path, wc_app_src: str | None) -> None:
    """Run store_output.py once per job, placing outputs in round_dir/<folder>/."""
    extra_env = {"WC_APP_SRC": wc_app_src} if wc_app_src else None

    for job in jobs:
        csv_path = _resolve_csv(job["csv"])
        folder = job["folder"]
        route = job.get("route")
        output_subdir = round_dir / folder

        label_parts = [f"store_output: {csv_path.name} → {folder}"]
        if route:
            label_parts.append(f"(route={route})")
        label = " ".join(label_parts)

        store_script = ROOT / "src" / "scripts" / "store_output.py"
        cmd = [
            sys.executable, str(store_script),
            str(csv_path),
            "--output", str(output_subdir),
        ]
        if route:
            cmd += ["--route", route]

        _run(cmd, label, extra_env=extra_env)


def run_reassign(round_dir: Path, folders: list[str]) -> None:
    """Run reassign_data_v2.py against the round directory."""
    label = f"reassign_data_v2: {round_dir.relative_to(ROOT)}"
    cmd = [
        sys.executable, "-m", "src.dataset_handling.reassign_data_v2",
        "--input-base", str(round_dir),
        "--folders", *folders,
    ]
    _run(cmd, label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    today = date.today()
    default_round = f"data_outputs/round_{today.strftime('%d%m')}"

    parser = argparse.ArgumentParser(
        description="Full evaluation data pipeline: store_output → reassign_data_v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.dataset_handling.data_pipeline\n"
            "  python -m src.dataset_handling.data_pipeline --round-dir data_outputs/round_test\n"
            "  python -m src.dataset_handling.data_pipeline --skip-store\n"
            "  python -m src.dataset_handling.data_pipeline --jobs "
            "'[{\"csv\":\"data/my.csv\",\"folder\":\"my_folder\"}]'\n"
        ),
    )
    parser.add_argument(
        "--round-dir",
        default=default_round,
        metavar="PATH",
        help=(
            "Output base directory for this pipeline run. "
            f"Default: {default_round}"
        ),
    )
    parser.add_argument(
        "--jobs",
        default=None,
        metavar="JSON",
        help=(
            "JSON array of job objects to override the default job list. "
            'Each object must have "csv" and "folder"; "route" is optional. '
            "Example: '[{\"csv\":\"data/my.csv\",\"folder\":\"my_folder\",\"route\":\"RESEARCH\"}]'"
        ),
    )
    parser.add_argument(
        "--skip-store",
        action="store_true",
        help="Skip the store_output step (reassign step still runs).",
    )
    parser.add_argument(
        "--skip-reassign",
        action="store_true",
        help="Skip the reassign_data_v2 step (store steps still run).",
    )
    parser.add_argument(
        "--wc-app-src",
        default=(os.getenv("WC_APP_SRC") or "").strip('"').strip("'") or None,
        metavar="PATH",
        dest="wc_app_src",
        help=(
            "Path to the Writing Coach app codebase root. "
            "Passed as WC_APP_SRC to store_output subprocesses. "
            "Defaults to the WC_APP_SRC environment variable if set."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve round dir
    round_dir = Path(args.round_dir)
    if not round_dir.is_absolute():
        round_dir = ROOT / round_dir

    # Parse jobs
    if args.jobs is not None:
        try:
            jobs = json.loads(args.jobs)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] --jobs is not valid JSON: {exc}")
            sys.exit(1)
        if not isinstance(jobs, list) or not all(
            isinstance(j, dict) and "csv" in j and "folder" in j for j in jobs
        ):
            print('[ERROR] --jobs must be a JSON array of objects with "csv" and "folder" keys.')
            sys.exit(1)
    else:
        jobs = DEFAULT_JOBS

    folders = list(dict.fromkeys(j["folder"] for j in jobs))  # deduplicated, ordered

    print("=" * 70)
    print("DATA PIPELINE")
    print(f"  round-dir    : {round_dir}")
    print(f"  jobs         : {len(jobs)}")
    print(f"  skip-store   : {args.skip_store}")
    print(f"  skip-reassign: {args.skip_reassign}")
    print(f"  wc-app-src   : {args.wc_app_src or '(not set — WC_APP_SRC must be in env)'}")
    print("=" * 70)

    if not args.skip_store:
        if not args.wc_app_src:
            print(
                "[ERROR] WC_APP_SRC is not set. "
                "Pass --wc-app-src or set the WC_APP_SRC environment variable "
                "to the root of the Writing Coach app codebase."
            )
            sys.exit(1)
        run_store_output(jobs, round_dir, args.wc_app_src)

    if not args.skip_reassign:
        run_reassign(round_dir, folders)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
