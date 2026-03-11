"""
Pre-process an input JSON dict by remapping complex reference IDs (e.g. C11-1-2-1)
to simple sequential integers, updating both the `response` text and the
`references` list accordingly.

Reference syntax in the response text:
    [C<numbers separated by ->]                        single ref
    [C11-1-2-1, C11-1-3-6, C11-1-2-5]               multiple refs in one bracket
    [C11-1-2-1], [C11-1-3-6]                          consecutive single-ref brackets

All of the above are collapsed into a single bracketed list:  [1, 2, 3]

Usage:
    python preprocess_references.py <input_json> [output_json]

If no output path is given the result is printed to stdout.
"""

import json
import re
import warnings
import jsonlines
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# Matches a single reference ID such as C11-1-2-1 (C followed by digits/hyphens)
_SINGLE_ID = r"C\d[\d\-]*"

# Matches a bracket group: one or more comma-separated IDs, e.g.
#   [C11-1-2-1]  or  [C11-1-2-1, C11-1-3-6, C11-1-2-5]
_BRACKET_GROUP = re.compile(
    r"\[("
    + _SINGLE_ID
    + r"(?:\s*,\s*"
    + _SINGLE_ID
    + r")*)\]"
)

# Two consecutive bracket groups are "adjacent" when they are separated only by
# optional whitespace, commas and/or semicolons.
_SEPARATOR = re.compile(r"^[\s,;]*$")


def _ids_in_bracket(bracket_match: re.Match) -> list[str]:
    """Return the list of raw IDs inside a bracket match."""
    return [s.strip() for s in bracket_match.group(1).split(",")]


def _find_consecutive_groups(matches: list[re.Match]) -> list[list[re.Match]]:
    """
    Group consecutive bracket matches that appear next to each other in the
    text (separated only by optional whitespace, commas and/or semicolons).
    """
    if not matches:
        return []

    groups: list[list[re.Match]] = []
    current_group = [matches[0]]

    for prev, curr in zip(matches, matches[1:]):
        gap = prev.string[prev.end(): curr.start()]
        if _SEPARATOR.match(gap):
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]

    groups.append(current_group)
    return groups


def preprocess_references(data: dict) -> dict:
    """
    Given a parsed JSON dict with a 'response' string and a 'references' list,
    return a new dict where:
      - Every occurrence of a reference ID (e.g. 'C11-1-2-1') is assigned a
        unique sequential integer index starting at 1, even if the same ID
        appears multiple times.
      - All IDs within a bracket group are replaced with their per-occurrence
        integer indexes.
      - Consecutive bracket groups are merged into a single '[1, 2, 3]' tag.
      - Each occurrence of a reference ID produces a separate entry in
        'references', with 'index' as a single integer for that occurrence.
        The same reference can therefore appear multiple times in the list.
      - 'reference_mapping' maps each old ID to the list of all its indexes.
    """
    response: str = data["response"]
    references: list[dict] = data.get("references", [])

    # ------------------------------------------------------------------
    # 1. Walk every bracket group in order and assign a fresh index to
    #    each ID occurrence.  Build:
    #      occurrence_map  : list of (match, {raw_id: index}) per bracket group
    #      id_to_indexes   : {raw_id: [index, ...]}  (all indexes for an ID)
    # ------------------------------------------------------------------
    counter = 1
    id_to_indexes: dict[str, list[int]] = {}
    occurrence_map: list[tuple[re.Match, dict[str, int]]] = []

    for match in _BRACKET_GROUP.finditer(response):
        group_assignment: dict[str, int] = {}
        for raw_id in _ids_in_bracket(match):
            idx = counter
            counter += 1
            group_assignment[raw_id] = idx
            id_to_indexes.setdefault(raw_id, []).append(idx)
        occurrence_map.append((match, group_assignment))

    # ------------------------------------------------------------------
    # 2. Rewrite the response: group consecutive bracket matches, then
    #    replace from right to left to keep offsets valid.
    # ------------------------------------------------------------------
    all_matches = [m for m, _ in occurrence_map]
    # Pair each match with its assignment dict for quick lookup
    assignment_by_match: dict[int, dict[str, int]] = {
        id(m): a for m, a in occurrence_map
    }

    groups = _find_consecutive_groups(all_matches)

    replacements: list[tuple[int, int, str]] = []
    for group in groups:
        start = group[0].start()
        end = group[-1].end()
        int_ids: list[str] = []
        seen: set[int] = set()
        for m in group:
            for raw_id in _ids_in_bracket(m):
                idx = assignment_by_match[id(m)][raw_id]
                if idx not in seen:
                    int_ids.append(str(idx))
                    seen.add(idx)
        replacements.append((start, end, f"[{', '.join(int_ids)}]"))

    new_response = response
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        new_response = new_response[:start] + replacement + new_response[end:]

    # ------------------------------------------------------------------
    # 3. Build the references list: one entry per index occurrence,
    #    so the same reference may appear multiple times — each with a
    #    single integer 'index' reflecting that specific occurrence.
    # ------------------------------------------------------------------
    # Map referenceId -> original ref dict for quick lookup
    ref_by_id: dict[str, dict] = {
        ref.get("referenceId", ""): ref for ref in references
    }
    new_references = []
    for raw_id, indexes in id_to_indexes.items():
        original = ref_by_id.get(raw_id)
        for idx in indexes:
            if original is not None:
                new_ref = dict(original)
                new_ref["index"] = idx
            else:
                new_ref = {"index": idx}
            new_references.append(new_ref)

    # Restore original order: sort by index
    new_references.sort(key=lambda r: r["index"])

    # ------------------------------------------------------------------
    # 4. Return the updated dict.
    # ------------------------------------------------------------------
    result = dict(data)
    result["old_response"] = response
    result["response"] = new_response
    result["references"] = new_references
    result["reference_mapping"] = {old: idxs for old, idxs in id_to_indexes.items()}
    return result


def load_and_preprocess_revise_rows(file_path: str | Path) -> list[dict]:
    """
    Read a ``.jsonl`` or ``.csv`` file, filter rows by ``route_orch``, and
    return a list of pre-processed dicts.

    Routing logic:

    * ``RESEARCH`` — behaves as before: preprocesses ``output["response"]``
      together with ``output["references"]``.
    * ``REVISE_RESEARCH`` — iterates over ``output["suggestions"]``, and for
      each suggestion preprocesses ``suggestion["transformed_text"]`` with the
      subset of ``output["references"]`` whose ``referenceId`` is cited in that
      each suggestion's text.  Returns one entry per suggestion in the format
      ``{"query": ..., "input_line": ..., "summary": ..., "reference": [...]}``.

    Rows where ``output["references"]`` is empty are skipped for ``RESEARCH``.
    Suggestions without any cited references are skipped for
    ``REVISE_RESEARCH``.

    Every output entry contains:
      - ``query``      : the original query string from the input row.
      - ``input_line`` : 1-based line number of the source row in the input file.
      - ``summary``    : the preprocessed response text with integer-only citation tags.
      - ``reference``  : list of reference dicts, one per citation occurrence.

    Parameters
    ----------
    file_path:
        Path to a ``.jsonl`` or ``.csv`` file.

    Returns
    -------
    list[dict]
        One pre-processed dict per qualifying row / suggestion.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    results: list[dict] = []

    def _process_row(row: dict, input_line: int) -> None:
        route = str(row.get("route_orch", ""))
        if route not in {"RESEARCH", "REVISE_RESEARCH"}:
            if route:
                warnings.warn(
                    f"Skipping row with unrecognised route_orch value: {route!r}. "
                    "Expected 'RESEARCH' or 'REVISE_RESEARCH'.",
                    UserWarning,
                    stacklevel=2,
                )
            return

        output = row["output"]
        if isinstance(output, str):
            output = json.loads(output)

        query: str = row.get("query", "")
        all_references: list[dict] = output.get("references") or []

        if route == "RESEARCH":
            if not all_references:
                return
            data = {
                "response": output["response"],
                "references": all_references,
            }
            processed = preprocess_references(data)
            results.append({
                "query": query,
                "input_line": input_line,
                "summary": processed["response"],
                "reference": processed["references"],
                "old_response": processed["old_response"],
                "reference_mapping": processed["reference_mapping"],
            })

        else:  # REVISE_RESEARCH
            # Build a lookup from referenceId -> reference dict for fast filtering
            ref_lookup: dict[str, dict] = {
                r["referenceId"]: r for r in all_references if r.get("referenceId")
            }
            for suggestion in output.get("suggestions") or []:
                text: str = suggestion.get("transformed_text", "")
                # Collect only the references cited in this suggestion's text,
                # preserving order of first appearance
                cited_ids: list[str] = []
                seen_ids: set[str] = set()
                for match in _BRACKET_GROUP.finditer(text):
                    for raw_id in _ids_in_bracket(match):
                        if raw_id not in seen_ids:
                            cited_ids.append(raw_id)
                            seen_ids.add(raw_id)
                suggestion_refs = [
                    ref_lookup[rid] for rid in cited_ids if rid in ref_lookup
                ]
                if not suggestion_refs:
                    continue
                data = {
                    "response": text,
                    "references": suggestion_refs,
                }
                processed = preprocess_references(data)
                results.append({
                    "query": query,
                    "input_line": input_line,
                    "summary": processed["response"],
                    "reference": processed["references"],
                    "old_response": processed["old_response"],
                    "reference_mapping": processed["reference_mapping"],
                })

    match suffix:
        case ".jsonl":
            with jsonlines.open(file_path) as reader:
                for line_no, row in tqdm(enumerate(reader, start=1), desc="Processing rows", unit="row"):
                    _process_row(row, input_line=line_no)

        case ".csv":
            df = pd.read_csv(file_path)
            for df_idx, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df), unit="row"):
                # df_idx is 0-based; +2 accounts for 0-based index and the header row
                _process_row(row.to_dict(), input_line=df_idx + 2)

        case _:
            raise ValueError(f"Unsupported file format '{suffix}'. Expected .jsonl or .csv.")

    return results


