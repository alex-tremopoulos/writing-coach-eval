"""Universal batch runner for Writing Coach outputs across multiple versions."""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.scripts.writing_coach_interfaces import create_writing_coach
from src.scripts.writing_coach_interfaces.utils import install_wc_app_path


CSV_FIELDNAMES = [
    'row_id', 'query', 'input_preview', 'input', 'route_orch', 'intent', 'reasoning',
    'segments_count', 'tools_used', 'response', 'suggestions', 'references',
    'research_papers', 'output',
    'folder_source', 'dataset_source', 'route_intended', 'metadata',
]

ROW_DELAY_SECONDS = 5  # Delay between rows to avoid HTTP 429 rate limit errors


def _load_environment() -> None:
    """Load env vars from repo .env so direct script runs get required config."""
    repo_env = Path(__file__).resolve().parents[2] / '.env'
    if repo_env.exists():
        load_dotenv(repo_env)
    else:
        load_dotenv()


def _already_processed(details_jsonl: Path) -> set:
    """Return set of row_ids already written to the JSONL output file."""
    processed = set()
    if details_jsonl.exists():
        with open(details_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        processed.add(json.loads(line)['row_id'])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return processed


def _build_output_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    """Build nested output dict with agent state fields only (no input dataset fields)."""
    return {
        'route': result.get('route', ''),
        'intent': result.get('intent', ''),
        'reasoning': result.get('reasoning', ''),
        'response': result.get('response', ''),
        'segments_count': result.get('segments_count', 0),
        'tools_used': result.get('tools_used', []),
        'suggestions': result.get('suggestions', []),
        'references': result.get('references', []),
        'research_papers': result.get('research_papers', []),
    }


def _serialize_csv_json(value: Any) -> str:
    """Serialize structured values for CSV columns using JSON strings."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _result_to_csv_row(result: Dict[str, Any]) -> Dict[str, Any]:
    """Project a full result payload into the flat CSV schema."""
    response = result.get('response', '') or ''
    route_orch = result.get('route_orch', result.get('route', ''))
    tools_used = result.get('tools_used') or []
    if isinstance(tools_used, str):
        tools_used_value = tools_used
    else:
        tools_used_value = ','.join(str(tool) for tool in tools_used)

    return {
        'row_id': result['row_id'],
        'query': result['query'],
        'input_preview': result['input_preview'],
        'input': result.get('input', ''),
        'route_orch': route_orch,
        'intent': result['intent'],
        'reasoning': result['reasoning'],
        'segments_count': result.get('segments_count', 0),
        'tools_used': tools_used_value,
        'response': response,
        'suggestions': _serialize_csv_json(result.get('suggestions', [])),
        'references': _serialize_csv_json(result.get('references', [])),
        'research_papers': _serialize_csv_json(result.get('research_papers', [])),
        'output': json.dumps(_build_output_dict(result), ensure_ascii=False),
        'folder_source': result.get('folder_source', ''),
        'dataset_source': result.get('dataset_source', ''),
        'route_intended': result.get('route_intended', ''),
        'metadata': result.get('metadata', ''),
    }


def _write_csv_row(csv_writer, result: Dict[str, Any]) -> None:
    """Write a single result row to the CSV."""
    csv_writer.writerow(_result_to_csv_row(result))


def _load_jsonl_results(details_jsonl: Path) -> list[Dict[str, Any]]:
    """Read result payloads from JSONL, ignoring malformed lines."""
    results = []
    if not details_jsonl.exists():
        return results

    with open(details_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def _ensure_results_csv_schema(results_csv: Path, details_jsonl: Path) -> None:
    """Upgrade legacy results CSVs so resumed runs include the response column."""
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return

    with open(results_csv, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader, [])

    if header == CSV_FIELDNAMES:
        return

    source_results = _load_jsonl_results(details_jsonl)
    if not source_results:
        with open(results_csv, 'r', encoding='utf-8-sig', newline='') as f:
            source_results = list(csv.DictReader(f))

    tmp_path = results_csv.with_suffix(results_csv.suffix + '.tmp')
    with open(tmp_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for result in source_results:
            _write_csv_row(writer, result)

    tmp_path.replace(results_csv)


def process_csv(
    input_csv: str,
    version: str = 'v3',
    output_dir: str = 'batch_outputs',
    filter_route: Optional[str] = None,
    limit: Optional[int] = None,
    results_csv_override: Optional[str] = None,
    details_jsonl_override: Optional[str] = None,
    wc_app_src: Optional[str] = None,
) -> None:
    """Process rows in CSV, writing results incrementally after each row.

    Supports resume: rows already present in the JSONL output are skipped.

    Args:
        input_csv: Path to input CSV with 'query', 'input', and optionally 'route' columns.
        version: Writing Coach version key (v2 or v3).
        output_dir: Directory for output files.
        filter_route: If set, only process rows whose 'route' column matches this value
                      (case-insensitive). Pass None to process all rows.
        limit: If set, only consider the first N selected rows after route filtering.
    """
    if limit is not None and limit < 0:
        raise ValueError('limit must be >= 0')

    _load_environment()
    install_wc_app_path(wc_app_src or os.getenv('WC_APP_SRC'))
    coach = create_writing_coach(version)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fixed filenames based on input stem so resume works across restarts
    # Allow caller to override output paths (e.g. to append into an existing file)
    stem = Path(input_csv).stem
    folder_source = Path(input_csv).resolve().parent.name
    dataset_source = stem
    route_suffix = f'_{filter_route.upper()}' if filter_route else ''
    if results_csv_override:
        results_csv = Path(results_csv_override)
        results_csv.parent.mkdir(parents=True, exist_ok=True)
    else:
        results_csv = output_path / f'{stem}{route_suffix}_results.csv'

    if details_jsonl_override:
        details_jsonl = Path(details_jsonl_override)
        details_jsonl.parent.mkdir(parents=True, exist_ok=True)
    else:
        details_jsonl = output_path / f'{stem}{route_suffix}_details.jsonl'

    _ensure_results_csv_schema(results_csv, details_jsonl)

    # Resume: skip rows already in the JSONL output
    processed_ids = _already_processed(details_jsonl)
    if processed_ids:
        print(f"Resuming — {len(processed_ids)} rows already processed, skipping them.")

    with open(input_csv, 'r', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    # Apply route filter if requested
    if filter_route:
        rows_to_run = [
            (i + 1, row) for i, row in enumerate(rows)
            if row.get('route', '').strip().upper() == filter_route.upper()
        ]
        print(
            f"\n[{coach.version}] Route filter: '{filter_route.upper()}' — "
            f"{len(rows_to_run)} matching rows out of {len(rows)} total"
        )
    else:
        rows_to_run = [(i + 1, row) for i, row in enumerate(rows)]
        print(f"\n[{coach.version}] Processing all {len(rows)} rows from {input_csv}")

    if limit is not None:
        original_count = len(rows_to_run)
        rows_to_run = rows_to_run[:limit]
        print(
            f"[{coach.version}] Row limit: processing first {len(rows_to_run)} "
            f"of {original_count} selected rows"
        )

    print("=" * 80)

    # Open output files in append mode so partial progress is never lost
    csv_is_new = not results_csv.exists() or results_csv.stat().st_size == 0
    csv_file   = open(results_csv,   'a', newline='', encoding='utf-8')
    jsonl_file = open(details_jsonl, 'a', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if csv_is_new:
        csv_writer.writeheader()

    completed = len(processed_ids)
    total = len(rows_to_run)

    if total == 0:
        csv_file.close()
        jsonl_file.close()
        print("\nNo rows selected for processing.")
    else:
        # Warm up with dummy query to consume cold start (first row CONVERSATION fallback)
        print("\n" + "=" * 80)
        print(f"WARMUP [{coach.version.upper()}]: Running dummy query to initialize graph...")
        try:
            dummy_result = coach.run_query(
                row_id=0,
                query="What is machine learning?",
                document_text="Machine learning is a subset of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on tasks through experience, without being explicitly programmed."
            )
            print(f"  Dummy query completed - route: {dummy_result['route']}")
        except Exception as e:
            print(f"  Dummy query failed (continuing anyway): {e}")
        print("=" * 80 + "\n")

        try:
            for idx, (row_id, row) in enumerate(rows_to_run):
                if row_id in processed_ids:
                    print(f"Row {row_id}: SKIPPED (already processed)")
                    continue

                query         = row.get('query', '').strip()
                document_text = row.get('input', '').strip()
                row_route_intended = (row.get('route_intended') or '').strip().upper()
                row_metadata       = row.get('metadata') or ''

                if not query or not document_text:
                    print(f"Row {row_id}: SKIPPED (missing query or input)")
                    continue

                print(f"\nRow {row_id}/{len(rows)} (#{idx + 1} of {total} to run): {query[:60]}...")

                try:
                    result = coach.run_query(row_id, query, document_text)
                except Exception as e:
                    if row_id == 1 and "orchestrator failed" in str(e):
                        # Retry first row once (cold start recovery)
                        time.sleep(2)
                        result = coach.run_query(row_id, query, document_text)
                    else:
                        print(f"  ERROR (attempt 1): {e} — retrying in 10s...")
                        time.sleep(10)
                        try:
                            result = coach.run_query(row_id, query, document_text)
                            print("  Retry succeeded.")
                        except Exception as e2:
                            print(f"  ERROR (attempt 2): {e2} — marking as ERROR.")
                            result = {
                                'row_id': row_id,
                                'query': query,
                                'input_preview': document_text[:200],
                                'input': document_text,
                                'route': 'ERROR',
                                'intent': 'error',
                                'reasoning': str(e2),
                                'response': '',
                                'suggestions': [],
                                'references': [],
                                'research_papers': [],
                                'segments_count': 0,
                                'tools_used': [],
                            }

                # Annotate with input-dataset provenance fields
                result['folder_source']   = folder_source
                result['dataset_source']  = dataset_source
                result['route_intended']  = row_route_intended
                result['metadata']        = row_metadata

                # Write immediately — flush to disk so no progress is lost on crash
                _write_csv_row(csv_writer, result)
                csv_file.flush()
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                jsonl_file.flush()

                completed += 1
                print(f"  Route: {result['route']} | Intent: {result['intent']} | "
                      f"Papers: {len(result['research_papers'])} | "
                      f"Response: {len(result['response'])} chars "
                      f"[{completed}/{total} done]")

                if idx == 0:  # First row just completed
                    print("  Warming up connections for 5s...")
                    time.sleep(5)

                # Delay between rows to avoid HTTP 429 rate limit errors
                if idx < total - 1:
                    print(f"  Waiting {ROW_DELAY_SECONDS}s before next row...")
                    time.sleep(ROW_DELAY_SECONDS)

        finally:
            csv_file.close()
            jsonl_file.close()

    # Build summary from the full JSONL file (includes all previous runs)
    all_results = []
    with open(details_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print("\n" + "=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(all_results)}")

    print("\nRouting breakdown:")
    for route, count in Counter(r['route'] for r in all_results).most_common():
        print(f"  {route:20} {count:3} queries")

    print("\nIntent breakdown:")
    for intent, count in Counter(r['intent'] for r in all_results).most_common():
        print(f"  {intent:20} {count:3} queries")

    print(f"\nOutputs saved:")
    print(f"  Summary : {results_csv}")
    print(f"  Details : {details_jsonl}")

    for _log_file in ["llm_outputs.txt", "llm_prompts.txt"]:
        _log_path = Path(_log_file)
        if _log_path.exists():
            _log_path.unlink()
            print(f"  Deleted : {_log_file}")


if __name__ == "__main__":
    import argparse

    _load_environment()

    parser = argparse.ArgumentParser(
        description="Batch Writing Coach Query Processor (universal)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/scripts/store_output.py queries.csv --version v3\n"
            "  python src/scripts/store_output.py queries.csv --version v2 --output my_output\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --route RESEARCH\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --limit 5\n"
            "  python src/scripts/store_output.py queries.csv --version v3 --route RESPOND --output respond_only\n"
            "\nValid route values (must match 'route' column in CSV):\n"
            "  RESEARCH, REVISE_RESEARCH, REVISE_SIMPLE, RESPOND"
        ),
    )
    parser.add_argument('input_csv', help='Path to input CSV with query and input columns')
    parser.add_argument('--version', choices=['v2', 'v3'], default='v3',
                        help='Writing Coach version to run (default: v3)')
    parser.add_argument('--output', default='batch_outputs', help='Output directory (default: batch_outputs)')
    parser.add_argument('--route', default=None, help='Only process rows matching this route value (e.g. RESEARCH, RESPOND)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process the first N selected rows after route filtering (useful for debugging)')
    parser.add_argument('--results-csv', default=None, dest='results_csv',
                        help='Override output CSV path (useful for appending into an existing file)')
    parser.add_argument('--details-jsonl', default=None, dest='details_jsonl',
                        help='Override output JSONL path (useful for appending into an existing file)')
    parser.add_argument('--wc-app-src', default=(os.getenv('WC_APP_SRC') or '').strip('"').strip("'") or None,
                        dest='wc_app_src', help='Path to Writing Coach app root (optional if WC_APP_SRC is set)')

    args = parser.parse_args()

    process_csv(
        input_csv=args.input_csv,
        version=args.version,
        output_dir=args.output,
        filter_route=args.route,
        limit=args.limit,
        results_csv_override=args.results_csv,
        details_jsonl_override=args.details_jsonl,
        wc_app_src=args.wc_app_src,
    )