"""Extract all Kiwi triplets whose queries are not already selected in merge_public_datasets.py.

Builds the same (input, query, target) triplets from the Kiwi interaction data,
then removes any row whose query matches one of the already-assigned queries
(using the same exact + fuzzy matching logic as merge_public_datasets.py).

Output: data/data_routes_kiwi_remaining.csv
Columns: query, input, target, dataset, route
  - dataset is always 'Kiwi'
  - route is left empty (to be assigned later)
"""
import os
import ssl
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# ---------------------------------------------------------------------------
# SSL / HuggingFace Hub setup (same as merge_public_datasets.py)
# ---------------------------------------------------------------------------
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

ssl._create_default_https_context = ssl._create_unverified_context

import httpx
_original_client_init = httpx.Client.__init__


def _patched_client_init(self, *args, **kwargs):
    kwargs.setdefault('verify', False)
    _original_client_init(self, *args, **kwargs)


httpx.Client.__init__ = _patched_client_init

# ---------------------------------------------------------------------------
# Project root discovery
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for _parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (_parent / '.git').exists() or (_parent / 'pyproject.toml').exists():
        PROJECT_ROOT = _parent
        break

DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Queries already assigned a route in merge_public_datasets.py
# ---------------------------------------------------------------------------
ALREADY_SELECTED_QUERIES = [
    # REVISE_SIMPLE
    "Rewrite the answer to be concise and directly answer the question. Try not to delete any of the content, just make it a bit more concise.",
    # REVISE_RESEARCH
    "Can you provide more evidence for why in-context learning works spanning multiple works and what they've shown?",
    "Include more methods that have been tried, maybe starting with simpler methods that lead to more complex ones to help orient the reader and introduce more complex concepts iteratively.",
    "Add a few sentences at the beginning to briefly explain the area and motivate why this problem is important.",
    "In the first couple of sentences, provide a general definition of contrastive learning and explain why it is useful in building foundation models.",
    'As first sentence of the text, add a short definition of "table question answering"',
    # RESPOND
    "Summarize and answer the question directly. When summarizing, try to put methods together into larger categories for easier reading.",
    "What are the strong results that these models produce for mathematical reasoning and programmatic execution tasks?",
    "Which specific tasks have these models been applied to?",
    "How much better are these models than the baselines?",
    "What is the performance of the models using contrastive learning?",
    "How well do state-of-the-art models perform on these datasets?",
    "What metric was used to measure performance on these datasets?",
    "What are advantages and disadvantages for each method listed?",
    "How does the selection of the pretraining corpus affect the performance and output of language models?",
    # RESEARCH
    "Find every paper related to task-specific pre-training adaptation and include the methods mentioned there in your answer. Be exhaustive in your list.",
    "Find a bunch of papers that describe different methods. Take them and group them based on similarity. For each group, give it a name, high level description, then paper specific details.",
    "Find many more papers and use them to form groups that describe the different factors. For each group, give it a name and a description of the group. At the end of each group should be the list of paper references.",
    "Find more parameter efficient tuning methods and group them by their similarities rather than enumerating them one by one. Try to find their big similarities and differences to give the reader a comprehensive overview of all the types of all parameter efficient tuning methods.",
    "Find more methods from different papers. Look at reasoning skills. Or how textual + tabular data pretraining can help. Weak supervision may be a good source to include as well. Find more papers to add to the groups and add new groups as needed.",
    "Find more ways that human feedback was used, then group them into categories with a title and high-level description of what it is.",
    "Can you retrieve information about additional input perturbation-based methods from supporting papers or passages?",
    "Are there other ways of doing task-specific pretraining mentioned in the supporting papers or passages?",
    "Cool! Are there any other existing methods for leveraging information in knowledge bases for task-specific language model fine-tuning?",
    "Can you list additional approaches proposed for query expansion and reformulation?",
    "Can you add more information discussing empirical observations about in-context learning behavior to this answer?",
    'Retrieve "Compressing deep neural networks by matrix product operators" and summarize how it answers the question.',
    'Retrieve "Tensorizing Neural Networks" and summarize how just that paper answers the question',
]

# ---------------------------------------------------------------------------
# Load Kiwi and build ALL triplets (no deduplication by query)
# ---------------------------------------------------------------------------
print("Loading Kiwi dataset...")
df_kiwi = pd.read_json(
    "hf://datasets/fangyuan/kiwi/interaction_data_with_annotator_ids.jsonl",
    lines=True,
)
print(f"Loaded {len(df_kiwi)} Kiwi rows")

triplets = []
for _, row in df_kiwi.iterrows():
    interaction_list = row['interaction']
    if not isinstance(interaction_list, list) or len(interaction_list) == 0:
        continue

    for turn_idx, turn in enumerate(interaction_list):
        if not isinstance(turn, dict):
            continue

        instruction = turn.get('instruction')
        answer_1 = turn.get('answer_1')
        answer_2 = turn.get('answer_2')

        if not instruction:
            continue

        if turn_idx == 0:
            current_input = row['initial_answer']
        else:
            prev_turn = interaction_list[turn_idx - 1]
            prev_answer_2 = prev_turn.get('answer_2')
            prev_answer_1 = prev_turn.get('answer_1')
            current_input = prev_answer_2 if prev_answer_2 and prev_answer_2 != '' else prev_answer_1

        target = answer_2 if answer_2 and answer_2 != '' else answer_1

        if not current_input or not target:
            continue

        triplets.append({
            'query': instruction,
            'input': current_input,
            'target': target,
        })

df_triplets = pd.DataFrame(triplets)
print(f"Built {len(df_triplets)} total triplets")

# ---------------------------------------------------------------------------
# Identify already-selected queries via exact + fuzzy match
# ---------------------------------------------------------------------------
all_kiwi_queries = df_triplets['query'].unique().tolist()


def find_matching_queries(target_queries, df_queries, threshold=90):
    """Return dataset queries that match any of target_queries (exact or fuzzy)."""
    matched = []
    for target in target_queries:
        if target in df_queries:
            matched.append(target)
        else:
            match = process.extractOne(target, df_queries, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                matched.append(match[0])
    return matched


already_matched = find_matching_queries(ALREADY_SELECTED_QUERIES, all_kiwi_queries)
print(f"Identified {len(already_matched)} already-selected queries (out of {len(ALREADY_SELECTED_QUERIES)} target queries)")

# ---------------------------------------------------------------------------
# Keep only triplets whose query was NOT already selected
# ---------------------------------------------------------------------------
df_remaining = df_triplets[~df_triplets['query'].isin(already_matched)].copy()
df_remaining['dataset'] = 'Kiwi'
df_remaining['route'] = ''
df_remaining.drop_duplicates(subset=['query'], inplace=True)
df_remaining = df_remaining[['query', 'input', 'target', 'dataset', 'route']]
df_remaining.reset_index(drop=True, inplace=True)

print(f"Remaining triplets: {len(df_remaining)}")

# ---------------------------------------------------------------------------
# Save full remaining set
# ---------------------------------------------------------------------------
output_path = DATA_DIR / 'data_routes_kiwi_remaining.csv'
df_remaining.to_csv(output_path, index=False)
print(f"Saved to {output_path}")

# ---------------------------------------------------------------------------
# Select 200 representative queries with route assignments
#
# Routes are based on the Writing Coach orchestrator routing definitions:
#
# RESPOND:         Follow-up questions, asking for explanations of existing content,
#                  comparisons or analyses of already-provided information, and
#                  questions about writing process that don't require a document edit.
#
# REVISE_SIMPLE:   Clear mechanical edits that need no external evidence:
#                  removing/deleting specific sentences/paragraphs, reformatting
#                  into one paragraph, combining paragraphs, shortening, truncating,
#                  replacing explicit text with given replacement text.
#
# REVISE_RESEARCH: Revisions that require adding new content grounded in papers:
#                  adding definitions/explanations/analyses, elaborating on specific
#                  referenced papers, vague improvement commands, adding new
#                  introductory or concluding content without explicit text provided.
#
# RESEARCH:        Explicitly finding papers, gathering methods from literature,
#                  exploring what exists on a topic, retrieving/explaining specific
#                  named papers, or asking "what other/more approaches exist?".
# ---------------------------------------------------------------------------

# Row indices (0-based) in data_routes_kiwi_remaining.csv mapped to routes.
# Each index set was verified to be non-overlapping across routes.
_RESPOND_INDICES = [
    0, 1, 2, 3, 4, 8, 10, 11, 21, 22, 26, 29, 32, 47, 59, 60, 61, 62,
    67, 69, 70, 73, 89, 102, 104, 105, 106, 108, 110, 112, 124, 126,
    127, 128, 129, 132, 133, 135, 141, 143, 147, 172, 174, 182, 183,
    184, 187, 199, 200, 202,
]
_REVISE_SIMPLE_INDICES = [
    5, 12, 13, 20, 24, 28, 30, 33, 36, 37, 41, 42, 43, 50, 57, 63, 64,
    65, 66, 71, 72, 74, 82, 87, 92, 101, 107, 109, 115, 116, 122, 125,
    134, 136, 142, 148, 157, 158, 160, 165, 171, 177, 181, 185, 186,
    191, 198, 201, 203, 240,
]
_REVISE_RESEARCH_INDICES = [
    14, 15, 17, 18, 27, 38, 39, 40, 44, 46, 51, 54, 55, 75, 76, 77, 78,
    79, 83, 85, 90, 93, 94, 95, 96, 98, 103, 117, 118, 121, 130, 137,
    138, 144, 149, 150, 151, 154, 155, 166, 167, 168, 169, 179, 192,
    194, 204, 205, 213, 217,
]
_RESEARCH_INDICES = [
    16, 52, 88, 227, 229, 235, 236, 237, 238, 239, 302, 317, 318, 323,
    335, 336, 337, 390, 391, 403, 404, 412, 431, 443, 466, 484, 487,
    513, 524, 525, 526, 540, 541, 547, 628, 630, 646, 647, 648, 866,
    867, 902, 903, 904, 948, 963, 967, 970, 971, 980,
]

route_assignments = (
    {i: 'RESPOND' for i in _RESPOND_INDICES}
    | {i: 'REVISE_SIMPLE' for i in _REVISE_SIMPLE_INDICES}
    | {i: 'REVISE_RESEARCH' for i in _REVISE_RESEARCH_INDICES}
    | {i: 'RESEARCH' for i in _RESEARCH_INDICES}
)

selected_indices = sorted(route_assignments.keys())
df_selected = df_remaining.iloc[selected_indices].copy()
df_selected['route'] = [route_assignments[i] for i in selected_indices]
df_selected = df_selected[['query', 'input', 'target', 'dataset', 'route']]
df_selected.reset_index(drop=True, inplace=True)

print(f"\nSelected {len(df_selected)} rows")
print("Route distribution:")
print(df_selected['route'].value_counts())

selected_output_path = DATA_DIR / 'data_routes_kiwi_selected200.csv'
df_selected.to_csv(selected_output_path, index=False)
print(f"Saved to {selected_output_path}")
