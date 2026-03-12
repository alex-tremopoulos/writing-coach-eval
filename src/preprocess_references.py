"""
CLI entry point for preprocessing references.

Reads a .jsonl or .csv input file, filters RESEARCH / REVISE_RESEARCH rows,
preprocesses their references, and writes the results to a .json output file
as a list of dictionaries.

Usage:
    python preprocess_references.py <input_file> <output_file>
    python preprocess_references.py input.jsonl output.json
    python preprocess_references.py input.csv results.json
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent))
from metrics.accuracy import load_and_preprocess_revise_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess references in RESEARCH/REVISE_RESEARCH rows from a .jsonl or .csv file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the input file (.jsonl or .csv).",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output .json file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        print(f"Error: input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.output_file.suffix.lower() != ".json":
        print(
            f"Warning: output file '{args.output_file}' does not have a .json extension.",
            file=sys.stderr,
        )

    results = load_and_preprocess_revise_rows(args.input_file)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"Written {len(results)} record(s) to '{args.output_file}'.")


if __name__ == "__main__":
    main()
