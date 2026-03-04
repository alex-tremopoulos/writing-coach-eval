"""Merge multiple public writing/editing datasets and consolidate by route type."""
import os
import ssl
from pathlib import Path

import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# Configure environment for Hugging Face Hub access
# Disable SSL verification at the environment level
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

# Set up project paths
# Project root discovery
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for parent in [SCRIPT_DIR] + list(SCRIPT_DIR.parents):
    if (parent / '.git').exists() or (parent / 'pyproject.toml').exists():
        PROJECT_ROOT = parent
        break

DATA_DIR = PROJECT_ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

os.chdir(PROJECT_ROOT)

# ============================================================================
# LOAD AND PROCESS DATASETS
# ============================================================================

# Load OpenRewriteEval dataset and filter by query type
df_ore = pd.read_parquet("hf://datasets/gabrielmbmb/OpenRewriteEval/data/train-00000-of-00001.parquet")
df_ore.drop_duplicates(subset=['comment'], inplace=True)
df_ore.rename(columns={'source': 'input', 'comment': 'query'}, inplace=True)
df_ore.reset_index(drop=True, inplace=True)

queries_revise_simple = [
    "Fix grammatical errors and sentence structures.",
    "Correct grammar and spelling, improve style, cohesion, and tone.",
    "Make sure the text is clear and easy to understand",
    "Make the text more readable and comprehensive. Remove jargon.",
    "write in a more formal and professional style",
    "Use a more conversational tone.",
    "Use bullet points to make this more readable",
    "Reorganize the information.",
    "Rewrite the text to be neutral.",
    "make it more objective and less biased",
    "Paraphrase this text",
]

queries_revise_research = [
    "Copyedit to improve the text.",
    "add more supporting evidence",
    "add more examples to support your argument",
    "add references",
    "add more facts and statistics",
    "make it more persuasive with stronger arguments",
    "add a counter-argument",
    "add more details",
]

queries_respond = [
    "Elaborate and write your opinion on the topic",
    "write a set of guidelines for new copyeditors",
    "create a list of questions",
    "write a persuasive argument for or against the text",
]

df_ore_revise_simple = df_ore[df_ore['query'].isin(queries_revise_simple)]
df_ore_revise_research = df_ore[df_ore['query'].isin(queries_revise_research)]
df_ore_respond = df_ore[df_ore['query'].isin(queries_respond)]

df_ore_sample = pd.concat([
    df_ore_revise_simple.assign(dataset='OpenRewriteEval', route='REVISE_SIMPLE'),
    df_ore_revise_research.assign(dataset='OpenRewriteEval', route='REVISE_RESEARCH'),
    df_ore_respond.assign(dataset='OpenRewriteEval', route='RESPOND'),
]).reset_index(drop=True)

# Load CoEdiT dataset and filter by query type
splits = {'train': 'train.jsonl', 'validation': 'validation.jsonl'}
df_coedit = pd.read_json("hf://datasets/grammarly/coedit/" + splits["train"], lines=True)

df_coedit['query'] = df_coedit['src'].apply(lambda x: x.split(':')[0])
df_coedit['input'] = df_coedit['src'].apply(lambda x: x.split(':')[1])
df_coedit.drop_duplicates(subset=['query'], inplace=True)
df_coedit.rename(columns={'tgt': 'target'}, inplace=True)
df_coedit.drop(columns=['src', '_id'], inplace=True)
df_coedit.reset_index(drop=True, inplace=True)

queries_to_keep = ['Fix disfluencies in the sentence']

df_coedit_sample = df_coedit[df_coedit['query'].isin(queries_to_keep)]
df_coedit_sample = pd.concat([
    df_coedit_sample.assign(dataset='CoEdiT', route='REVISE_SIMPLE'),
]).reset_index(drop=True)

# Load Kiwi dataset (interaction data) and construct triplets
# Each row contains a sequence of edits (turns) where users iteratively refine outputs
df_kiwi = pd.read_json("hf://datasets/fangyuan/kiwi/interaction_data_with_annotator_ids.jsonl", lines=True)

triplets = []
for _, row in df_kiwi.iterrows():
    # Skip rows with no interaction sequence
    interaction_list = row['interaction']
    if not isinstance(interaction_list, list) or len(interaction_list) == 0:
        continue

    # Process each turn in the interaction sequence
    for turn_idx, turn in enumerate(interaction_list):
        if not isinstance(turn, dict):
            continue

        # Extract instruction and candidate answers from this turn
        instruction = turn.get('instruction')
        answer_1 = turn.get('answer_1')
        answer_2 = turn.get('answer_2')
        rating = turn.get('rating')

        if not instruction:
            continue

        # Input is the initial text for first turn, or previous turn's output for subsequent turns
        if turn_idx == 0:
            current_input = row['initial_answer']
        else:
            prev_turn = interaction_list[turn_idx - 1]
            prev_answer_2 = prev_turn.get('answer_2')
            prev_answer_1 = prev_turn.get('answer_1')
            current_input = prev_answer_2 if prev_answer_2 and prev_answer_2 != '' else prev_answer_1

        # Target is user-edited answer if available, otherwise first model answer
        target = answer_2 if answer_2 and answer_2 != '' else answer_1

        # Skip incomplete triplets
        if not current_input or not target:
            continue

        # Store this (input, instruction, output) triplet
        triplets.append({
            'input': current_input,
            'query': instruction,
            'target': target,
            'rating': rating,
            'turn_number': turn_idx,
            'model_name': row.get('model_name'),
            'has_answer_2': bool(answer_2 and answer_2 != ''),
        })

df_kiwi_triplets = pd.DataFrame(triplets)
df_kiwi_triplets.drop_duplicates(subset=['query'], inplace=True)
df_kiwi_triplets.drop(columns=['rating', 'turn_number', 'model_name', 'has_answer_2'], inplace=True)

queries_revise_simple_kiwi = [
    "Rewrite the answer to be concise and directly answer the question. Try not to delete any of the content, just make it a bit more concise."
]

queries_revise_research_kiwi = [
    "Can you provide more evidence for why in-context learning works spanning multiple works and what they've shown?",
    "Include more methods that have been tried, maybe starting with simpler methods that lead to more complex ones to help orient the reader and introduce more complex concepts iteratively.",
    "Add a few sentences at the beginning to briefly explain the area and motivate why this problem is important.",
    "In the first couple of sentences, provide a general definition of contrastive learning and explain why it is useful in building foundation models.",
    "As first sentence of the text, add a short definition of \"table question answering\"",
]

queries_respond_kiwi = [
    "Summarize and answer the question directly. When summarizing, try to put methods together into larger categories for easier reading.",
    "What are the strong results that these models produce for mathematical reasoning and programmatic execution tasks?",
    "Which specific tasks have these models been applied to?",
    "How much better are these models than the baselines?",
    "What is the performance of the models using contrastive learning?",
    "How well do state-of-the-art models perform on these datasets?",
    "What metric was used to measure performance on these datasets?",
    "What are advantages and disadvantages for each method listed?",
    "How does the selection of the pretraining corpus affect the performance and output of language models?",
]

queries_research_kiwi = [
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
    "Retrieve \"Compressing deep neural networks by matrix product operators\" and summarize how it answers the question.",
    "Retrieve \"Tensorizing Neural Networks\" and summarize how just that paper answers the question",
]


# Helper function for query matching
def find_matching_queries(target_queries, df_queries, threshold=90):
    """Find queries in dataframe using exact match first, then fuzzy matching."""
    matched_queries = []
    for target in target_queries:
        if target in df_queries:
            matched_queries.append(target)
        else:
            match = process.extractOne(target, df_queries, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                matched_queries.append(match[0])
    return matched_queries


all_df_queries = df_kiwi_triplets['query'].unique().tolist()

matched_revise_simple = find_matching_queries(queries_revise_simple_kiwi, all_df_queries)
matched_revise_research = find_matching_queries(queries_revise_research_kiwi, all_df_queries)
matched_respond = find_matching_queries(queries_respond_kiwi, all_df_queries)
matched_research = find_matching_queries(queries_research_kiwi, all_df_queries)

df_kiwi_revise_simple = df_kiwi_triplets[df_kiwi_triplets['query'].isin(matched_revise_simple)]
df_kiwi_revise_research = df_kiwi_triplets[df_kiwi_triplets['query'].isin(matched_revise_research)]
df_kiwi_respond = df_kiwi_triplets[df_kiwi_triplets['query'].isin(matched_respond)]
df_kiwi_research = df_kiwi_triplets[df_kiwi_triplets['query'].isin(matched_research)]

df_kiwi_sample = pd.concat([
    df_kiwi_revise_simple.assign(dataset='Kiwi', route='REVISE_SIMPLE'),
    df_kiwi_revise_research.assign(dataset='Kiwi', route='REVISE_RESEARCH'),
    df_kiwi_respond.assign(dataset='Kiwi', route='RESPOND'),
    df_kiwi_research.assign(dataset='Kiwi', route='RESEARCH'),
]).reset_index(drop=True)

# ============================================================================
# MERGE AND EXPORT
# ============================================================================

# Concatenate all datasets, reorder columns, and save
common_cols = list(set(df_ore_sample.columns) & set(df_coedit_sample.columns) & set(df_kiwi_sample.columns))

df_combined = pd.concat([
    df_ore_sample[common_cols],
    df_coedit_sample[common_cols],
    df_kiwi_sample[common_cols],
], ignore_index=True)

df_reordered = df_combined.iloc[:, [1, 3, 0, 2, 4]]
df_reordered.sort_values(by=['route', 'dataset'], inplace=True)
df_reordered.reset_index(drop=True, inplace=True)

df_reordered.to_csv("data/data_routes_original_test.csv", index=False)
