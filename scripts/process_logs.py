from zipfile import ZipFile, Path
from baysic.utils import json_to_df, df_to_json, to_sorted_pretty_string, to_pretty_name
from pymatgen.core import Composition
import json
from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
import pathlib

def process(num, redo=False):
    path = (pathlib.Path('logs') / f'{num}.feather')
    if path.exists() and not redo:
        print(f'Already did {num}, skipping')
        return pd.read_feather(path)
    dfs = []
    with ZipFile(f'logs/{num}.zip', 'r') as logs:
        base_path = Path(logs, f'work/miklaucn/logs/{num}/')
        print(len(logs.filelist))
        for comp_path in base_path.iterdir():
            rows = []
            for group_path in (comp_path / '1').iterdir():
                with group_path.open('r') as data:
                    try:
                        run_df = json_to_df(data)
                        run_df = run_df[['e_form',  'group_num', 'volume_ratio',
                        'lattice_type', 'wyckoff_letters', 'total_dof']]
                        run_df['i'] = num
                        rows.append(run_df)
                    except Exception as e:
                        print(group_path.at)
                        print(e)
                        continue

            try:
                run_df = pd.concat(rows)
                run_df['comp'] = to_sorted_pretty_string(Composition(comp_path.name))
                dfs.append(run_df)
            except Exception as e:
                print(comp_path.at)
                print(e)
                continue

    df = pd.concat(dfs)
    df = df.reset_index(names='comp_ind')
    df['group_num'] = pd.Categorical(df['group_num'], ordered=True)
    df['group_symbol'] = to_pretty_name(df['group_num'])
    df['group_symbol'] = pd.Categorical(df['group_symbol'], df.groupby('group_num').first()['group_symbol'].values, ordered=True)
    df['group_symbol'] = pd.Categorical(df['group_symbol'], to_pretty_name(pd.Series(list(range(1, 231)))), ordered=True)
    df['lattice_type'] = pd.Categorical(df['lattice_type'], ['triclinic', 'monoclinic', 'orthorhombic', 'tetragonal', 'hexagonal', 'cubic'], ordered=True)

    for cat_col in ['comp', 'wyckoff_letters']:
        df[cat_col] = pd.Categorical(df[cat_col])

    df.to_feather(path)
    return df

def main(nums):
    return process_map(process, nums, max_workers=8)

if __name__ == '__main__':
    nums = list(range(81, 101))
    dfs = main(nums)
    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df.to_feather(f'logs/81-100.feather')
