"""
Batch Writing Coach V2 Query Processor

Reads queries from CSV, executes Writing Coach V2 graph for each row,
and stores outputs including routing decisions.

CSV format:
  query,input
  "Find papers about X","Document text here..."
  "Strengthen this claim","More text..."

Output:
  batch_outputs/results_TIMESTAMP.csv
  batch_outputs/details_TIMESTAMP.json
"""

import csv
import json
import sys
import time
import types
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from collections import Counter

from src.graph_presets import get_preset, register_preset
from src.presets_config import PRESET_CONFIGS, get_preset_config
from src.state_definitions import WritingCoachV2State


def _install_learning_formatter_shim() -> None:
    """Provide the legacy learning_formatters module expected by WC V2 nodes."""
    module_name = 'src.tools.search.learning_formatters'
    if module_name in sys.modules:
        return

    from src.tools.search.llm_formatters import format_llm_results

    shim = types.ModuleType(module_name)

    def format_documents_for_learnings(results, include_reference_prefix=True):
        return format_llm_results(
            results,
            include_reference_prefix=include_reference_prefix,
        )

    shim.format_documents_for_learnings = format_documents_for_learnings
    shim.__all__ = ['format_documents_for_learnings']
    sys.modules[module_name] = shim


def _coerce_llm_text_result(result: Any) -> Any:
    """Convert LangChain message objects to plain text for legacy WC V2 consumers."""
    if isinstance(result, (dict, list, str)) or result is None:
        return result

    content = getattr(result, 'content', None)
    if content is None:
        return result

    if isinstance(content, list):
        return ' '.join(
            part.get('text', '') if isinstance(part, dict) else str(part)
            for part in content
        ).strip()

    if isinstance(content, str):
        return content

    return str(content)


def _install_invoke_llm_shim() -> None:
    """Normalize invoke_llm outputs for legacy WC V2 code used by this batch script."""
    import src.utils.llm_utils as llm_utils

    if getattr(llm_utils.invoke_llm, '_store_output_shimmed', False):
        return

    original_invoke_llm = llm_utils.invoke_llm

    def invoke_llm_compat(*args, **kwargs):
        result = original_invoke_llm(*args, **kwargs)
        response_format = kwargs.get('response_format', 'text')
        if response_format == 'json_object' or (
            isinstance(response_format, dict) and response_format.get('type') == 'json_object'
        ):
            return result
        if kwargs.get('stream') and not kwargs.get('writer'):
            return result
        return _coerce_llm_text_result(result)

    invoke_llm_compat._store_output_shimmed = True
    llm_utils.invoke_llm = invoke_llm_compat


def _build_writing_coach_v2_config() -> Dict[str, Any]:
    """Build a standalone WC V2 config without requiring a repo-wide preset entry."""
    copilot_config = get_preset_config('copilot_2_v4')
    copilot_models = dict(copilot_config.get('models', {}))
    copilot_prompts = dict(copilot_config.get('prompts', {}))
    copilot_parameters = dict(copilot_config.get('parameters', {}))

    return {
        'display_name': 'Writing Coach V2',
        'description': 'Standalone batch config for Writing Coach V2',
        'mode_indicator': 'Writing Coach V2',
        'type': 'writing_coach',
        'expose_in_ui': False,
        'tools': copilot_config.get('tools', {}),
        'parameters': {
            **copilot_parameters,
            'preset': 'writing_coach_v2',
            'copilot_preset': 'copilot_2_v4',
            'document_search_limit': copilot_parameters.get('document_search_limit', 20),
            'supports_conversation': True,
            'conversation_turn_limit': 10,
        },
        'prompts': {
            **copilot_prompts,
            'wc_orchestrator': 'v2',
            'wc_respond': 'v2',
            'wc_segment_analysis': 'v1',
            'research_response': 'v1',
            'parallel_research_transform': 'v1',
            'parallel_simple_transform': 'v1',
            'revision_explanation': 'v1',
        },
        'models': {
            **copilot_models,
            'orchestrator': copilot_models.get('orchestrator', 'gpt-5.1-chat'),
            'research_response': copilot_models.get('copilot_summary_standard', 'gpt-5.1-chat'),
            'structured_transform': 'gpt-5.1-chat',
            'text_transformation': 'gpt-5.1-chat',
            'respond': copilot_models.get('copilot_reinterpret', 'gpt-5.1-chat'),
            'segment_analysis': 'gpt-5-mini',
            'search_parameter_extraction': copilot_models.get('search_parameter_extraction', 'gpt-5.1-chat'),
            'learnings_extraction': copilot_models.get('learnings_extraction', 'gpt-5-mini'),
        },
    }


def _install_writing_coach_v2_config() -> Dict[str, Any]:
    """Register a runtime-only WC V2 preset config for the batch script."""
    config = PRESET_CONFIGS.get('writing_coach_v2')
    if config is None:
        config = _build_writing_coach_v2_config()
        PRESET_CONFIGS['writing_coach_v2'] = config
    return config


def _initialize_writing_coach_v2_only():
    """Initialize ONLY the writing_coach_v2 preset, skipping all others.

    Node names and edges mirror initialize_writing_coach_v2() in graph_presets.py.
    """
    from src.graph_builder import GraphBuilder
    from src.graph_nodes.writing_coach_v2_nodes import (
        wc_orchestrator_node, wc_orchestrator_router,
        segment_analysis_node, segment_analysis_router,
        search_node, search_router,
        research_response_node, research_transform_node,
        simple_transform_node, revision_explanation_node,
        wc_respond_node, output_node,
    )
    from langgraph.graph import START, END

    builder = GraphBuilder(WritingCoachV2State)
    builder.set_display_name("Writing Coach V2")
    builder.set_description("Conversational writing coach with hybrid graph architecture")

    # Register node functions (names match graph_presets.py exactly)
    for name, fn in [
        ("wc2/orchestrator", wc_orchestrator_node),
        ("wc2/segment_analysis", segment_analysis_node),
        ("wc2/copilot_search", search_node),
        ("wc2/research_response", research_response_node),
        ("wc2/research_transform", research_transform_node),
        ("wc2/simple_transform", simple_transform_node),
        ("wc2/revision_explanation", revision_explanation_node),
        ("wc2/respond", wc_respond_node),
        ("wc2/output", output_node),
    ]:
        builder.register_node_function(name, fn)
        builder.add_node(name)

    # Edges — orchestrator routes RESEARCH, REVISE_RESEARCH, REVISE_SIMPLE through segment_analysis
    builder.add_edge(START, "wc2/orchestrator")
    builder.add_conditional_edge("wc2/orchestrator", wc_orchestrator_router, {
        "wc2/segment_analysis": "wc2/segment_analysis",
        "wc2/respond": "wc2/respond",
    })

    # Segment analysis → route based on segments and action
    builder.add_conditional_edge("wc2/segment_analysis", segment_analysis_router, {
        "wc2/copilot_search": "wc2/copilot_search",
        "wc2/simple_transform": "wc2/simple_transform",
        "wc2/revision_explanation": "wc2/revision_explanation",
        "wc2/respond": "wc2/respond",
    })
    builder.add_conditional_edge("wc2/copilot_search", search_router, {
        "wc2/research_response": "wc2/research_response",
        "wc2/research_transform": "wc2/research_transform",
    })

    builder.add_edge("wc2/research_response", "wc2/output")

    builder.add_edge("wc2/research_transform", "wc2/revision_explanation")
    builder.add_edge("wc2/simple_transform", "wc2/revision_explanation")
    builder.add_edge("wc2/revision_explanation", "wc2/output")

    builder.add_edge("wc2/respond", "wc2/output")
    builder.add_edge("wc2/output", END)

    register_preset("writing_coach_v2", builder)
    print("Writing Coach V2 preset initialized (standalone)")


# Initialize ONLY writing_coach_v2
print("Initializing Writing Coach V2...")
_install_learning_formatter_shim()
_install_invoke_llm_shim()
preset_config = _install_writing_coach_v2_config()
_initialize_writing_coach_v2_only()
builder = get_preset("writing_coach_v2")
graph = builder.build_without_checkpointing()


def run_query(row_id: int, query: str, document_text: str) -> Dict[str, Any]:
    """Execute Writing Coach V2 for a single query."""
    initial_state: WritingCoachV2State = {
        'message': query,
        'document_text': document_text,
        'selected_text': None,
        'conversation_history': [],
        'prior_references': [],
        'conversation_id': f'batch_{row_id}',
        'conversation_turn': 1,
        'streaming': False,
        'writer': None,
        'preset': 'writing_coach_v2',
        'model_config': preset_config.get('models', {}),
        'prompt_versions': preset_config.get('prompts', {}),
        'parameters': preset_config.get('parameters', {}),
        'suggestions': [],
        'references': [],
        'research_papers': [],
    }

    final_state = graph.invoke(initial_state)

    orch_output = final_state.get('orchestrator_output', {})
    route = orch_output.get('next_action', 'UNKNOWN')
    intent = final_state.get('intent', 'unknown')
    reasoning = orch_output.get('reasoning', '')

    return {
        'row_id': row_id,
        'query': query,
        'input_preview': document_text[:200] + '...' if len(document_text) > 200 else document_text,
        'input': document_text,
        'route': route,
        'intent': intent,
        'reasoning': reasoning,
        'response': final_state.get('response', ''),
        'suggestions': final_state.get('suggestions', []),
        'references': final_state.get('references', []),
        'research_papers': final_state.get('research_papers', []),
        'segments_count': len(final_state.get('segments', [])),
        'tools_used': final_state.get('tools_used', []),
    }


CSV_FIELDNAMES = [
    'row_id', 'query', 'input_preview', 'input', 'route', 'intent', 'reasoning',
    'response_length', 'suggestions_count', 'references_count',
    'papers_count', 'segments_count', 'tools_used'
]

ROW_DELAY_SECONDS = 5  # Delay between rows to avoid HTTP 429 rate limit errors


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


def _write_csv_row(csv_writer, result: Dict[str, Any]) -> None:
    """Write a single result row to the CSV."""
    csv_writer.writerow({
        'row_id': result['row_id'],
        'query': result['query'],
        'input_preview': result['input_preview'],
        'input': result.get('input', ''),
        'route': result['route'],
        'intent': result['intent'],
        'reasoning': result['reasoning'],
        'response_length': len(result['response']),
        'suggestions_count': len(result['suggestions']),
        'references_count': len(result['references']),
        'papers_count': len(result['research_papers']),
        'segments_count': result['segments_count'],
        'tools_used': ','.join(result.get('tools_used') or []),
    })


def process_csv(
    input_csv: str,
    output_dir: str = 'batch_outputs',
    filter_route: Optional[str] = None,
    results_csv_override: Optional[str] = None,
    details_jsonl_override: Optional[str] = None,
) -> None:
    """Process rows in CSV, writing results incrementally after each row.

    Supports resume: rows already present in the JSONL output are skipped.

    Args:
        input_csv: Path to input CSV with 'query', 'input', and optionally 'route' columns.
        output_dir: Directory for output files.
        filter_route: If set, only process rows whose 'route' column matches this value
                      (case-insensitive). Pass None to process all rows.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Fixed filenames based on input stem so resume works across restarts
    # Allow caller to override output paths (e.g. to append into an existing file)
    stem = Path(input_csv).stem
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
        print(f"\nRoute filter: '{filter_route.upper()}' — {len(rows_to_run)} matching rows out of {len(rows)} total")
    else:
        rows_to_run = [(i + 1, row) for i, row in enumerate(rows)]
        print(f"\nProcessing all {len(rows)} rows from {input_csv}")

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

    # Warm up with dummy query to consume cold start (first row CONVERSATION fallback)
    print("\n" + "=" * 80)
    print("WARMUP: Running dummy query to initialize graph and consume cold start...")
    try:
        dummy_result = run_query(
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

            if not query or not document_text:
                print(f"Row {row_id}: SKIPPED (missing query or input)")
                continue

            print(f"\nRow {row_id}/{len(rows)} (#{idx + 1} of {total} to run): {query[:60]}...")

            try:
                result = run_query(row_id, query, document_text)
            except Exception as e:
                if row_id == 1 and "orchestrator failed" in str(e):
                    # Retry first row once (cold start recovery)
                    time.sleep(2)
                    result = run_query(row_id, query, document_text)
                else:
                    print(f"  ERROR: {e}")
                    result = {
                        'row_id': row_id,
                        'query': query,
                        'input_preview': document_text[:200],
                        'input': document_text,
                        'route': 'ERROR',
                        'intent': 'error',
                        'reasoning': str(e),
                        'response': '',
                        'suggestions': [],
                        'references': [],
                        'research_papers': [],
                        'segments_count': 0,
                        'tools_used': [],
                    }

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch Writing Coach V2 Query Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.store_output queries.csv\n"
            "  python -m src.store_output queries.csv --output my_output\n"
            "  python -m src.store_output queries.csv --route RESEARCH\n"
            "  python -m src.store_output queries.csv --route RESPOND --output respond_only\n"
            "\nValid route values (must match 'route' column in CSV):\n"
            "  RESEARCH, REVISE_RESEARCH, REVISE_SIMPLE, RESPOND"
        ),
    )
    parser.add_argument('input_csv', help='Path to input CSV with query and input columns')
    parser.add_argument('--output', default='batch_outputs', help='Output directory (default: batch_outputs)')
    parser.add_argument('--route', default=None, help='Only process rows matching this route value (e.g. RESEARCH, RESPOND)')
    parser.add_argument('--results-csv', default=None, dest='results_csv',
                        help='Override output CSV path (useful for appending into an existing file)')
    parser.add_argument('--details-jsonl', default=None, dest='details_jsonl',
                        help='Override output JSONL path (useful for appending into an existing file)')

    args = parser.parse_args()

    process_csv(
        args.input_csv,
        args.output,
        filter_route=args.route,
        results_csv_override=args.results_csv,
        details_jsonl_override=args.details_jsonl,
    )