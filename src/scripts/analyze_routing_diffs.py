"""
Compares a run-specific all_results CSV against the gold-standard all_results.csv.

Edit NEW_FILE below to point at the file you want to analyse.

Rows are matched by (row_id_previous_folder, folder_source) — the stable
logical key — NOT by global row_id, which differs between files because the
two scripts process folders in different orders.

Report:
  * How many rows in the new file have route_orch != route_intended
  * Of those rows, which have the SAME route values in all_results
  * Of those rows, which have DIFFERENT route values in all_results
"""
import pandas as pd
from pathlib import Path

# ── Configure here ─────────────────────────────────────────────────────────
NEW_FILE  = 'final_data/all_results_0414.csv'
GOLD_FILE = 'final_data/all_results.csv'
# ───────────────────────────────────────────────────────────────────────────

pd.set_option('display.max_colwidth', 150)
pd.set_option('display.width', 300)

KEY = ['row_id_previous_folder', 'folder_source']

new_label = Path(NEW_FILE).stem   # e.g. "all_results_0409"

df_new = pd.read_csv(NEW_FILE)
df_all = pd.read_csv(GOLD_FILE)

# ── Find rows in new file where route_orch != route_intended ───────────────
diff_new = df_new[df_new['route_orch'] != df_new['route_intended']].copy()
print(f'{new_label}: {len(diff_new)} / {len(df_new)} rows where route_orch != route_intended')

# ── Merge on logical key ───────────────────────────────────────────────────
gold_cols = KEY + ['route_orch', 'route_intended']
merged = diff_new.merge(
    df_all[gold_cols].rename(columns={
        'route_orch':     'orch_all',
        'route_intended': 'intended_all',
    }),
    on=KEY,
    how='left',
)
merged.rename(columns={'route_orch': 'orch_new', 'route_intended': 'intended_new'}, inplace=True)

# ── Classify ───────────────────────────────────────────────────────────────
same_mask = (
    (merged['orch_new'] == merged['orch_all']) &
    (merged['intended_new'] == merged['intended_all'])
)
same_rows = merged[same_mask]
diff_rows = merged[~same_mask]

print(f'\nOf those {len(diff_new)} rows (matched by prev_id + folder):')
print(f'  Same in all_results     : {len(same_rows)}')
print(f'  Different in all_results: {len(diff_rows)}')

def print_row(row):
    print(f'\n  (prev_id={row["row_id_previous_folder"]}, folder={row["folder_source"]})')
    print(f'    query    : {row["query"]}')
    print(f'    {new_label:<30}: orch={row["orch_new"]}  |  intended={row["intended_new"]}')
    print(f'    all_results                   : orch={row["orch_all"]}  |  intended={row["intended_all"]}')

# ── SAME ──────────────────────────────────────────────────────────────────
print('\n' + '='*120)
print(f'ROWS SAME BETWEEN {new_label} AND all_results  ({len(same_rows)} rows)')
print('route_orch and route_intended match exactly — both files agree this is a mismatch')
print('='*120)
for _, row in same_rows.iterrows():
    print_row(row)

# ── DIFFERENT ─────────────────────────────────────────────────────────────
print('\n' + '='*120)
print(f'ROWS DIFFERENT BETWEEN {new_label} AND all_results  ({len(diff_rows)} rows)')
print('At least one of route_orch or route_intended differs across files')
print('='*120)
for _, row in diff_rows.iterrows():
    changed = []
    if row['orch_new']     != row['orch_all']:     changed.append('route_orch')
    if row['intended_new'] != row['intended_all']: changed.append('route_intended')
    print(f'\n  (prev_id={row["row_id_previous_folder"]}, folder={row["folder_source"]})  [changed: {", ".join(changed)}]')
    print(f'    query    : {row["query"]}')
    print(f'    {new_label:<30}: orch={row["orch_new"]}  |  intended={row["intended_new"]}')
    print(f'    all_results                   : orch={row["orch_all"]}  |  intended={row["intended_all"]}')
