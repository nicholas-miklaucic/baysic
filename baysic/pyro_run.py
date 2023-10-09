"""Generating lots of crystals using Pyro."""

from copy import deepcopy
from networkx import chain_decomposition
import numpy as np

import torch
from baysic.lattice import CubicLattice
from baysic.pyro_generator import SystemStructureModel
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from baysic.structure_evaluation import e_form, point_energy, relaxed_energy
from pymatgen.core import Composition
import pandas as pd

from baysic.utils import df_to_json

torch.manual_seed(29433)

smoke_test = False

log_dir = Path('logs/')
overwrite = False

if smoke_test:
    num_generations = 1
else:
    num_generations = 200
    
failure_factor = 100

date = datetime.now().strftime('%m-%d')
date_dir = log_dir / date
if not date_dir.exists():
    date_dir.mkdir()

run_num = 1
while (date_dir / str(run_num)).exists() and not overwrite:
    run_num += 1
    if run_num >= 100:
        raise ValueError('You sure you want to make 100 folders for a day?')
    
run_dir = date_dir / str(run_num)
run_dir.mkdir(exist_ok=True)

df = pd.read_pickle('merged_test_data3.pkl')

test_df = df.query('CrystalSystem == "Cubic"').sort_values('nsites')[::2]

if smoke_test:
    test_df = test_df.iloc[:2]

task_rows = []
for i, row in tqdm(test_df.iterrows(), total=test_df.shape[0], colour='#1d71df', desc='Molecules'):
    struct = row['struct']    

    sga = SpacegroupAnalyzer(struct)
    symm = sga.get_symmetrized_structure()
    conv = sga.get_conventional_standard_structure()
    model = SystemStructureModel(conv.composition, CubicLattice)
    structs = []
    wsyms = []
    lat_a = []
    lat_vol = []
    e_forms = []
    groups = []
    pre_generated = 0
    success_tries = 0
    fail_2 = 0
    tries = 0
    with tqdm(total=num_generations, colour='#df1d71') as bar:
        while len(structs) < num_generations and tries < failure_factor * num_generations:
            tries += 1
            try:
                coords, lattice, elems, wsets, sg = model()
            except ValueError:
                continue
            
            new_structs = model.to_structures()[:10]
            e_form_vals = []
            good_structs = []            
            for struct in new_structs:
                e_form_val = point_energy(deepcopy(struct))
                if e_form_val < 80:
                    bar.update()
                    e_form_vals.append(e_form_val)
                    good_structs.append(struct)                 
                else:
                    fail_2 += 1
                        
            if not good_structs:
                continue
            
            success_tries += 1                               
            structs.extend(good_structs)
            e_forms.extend(e_form_vals)
            bar.set_description(f'{tries + 1}')
            for _ in range(len(good_structs)):
                wsyms.append('_'.join([wset.wp.letter for wset in wsets]))
                lat_a.append(lattice[0, 0].item())
                lat_vol.append(lattice[0, 0].item() ** 3)
                groups.append(sg)         

    if not structs:
        print(f'Generation of {row["formula_pretty"]} failed')
        continue

    if tries >= failure_factor * num_generations:
        print(f'Only {success_tries} successes, not {num_generations}')
    
    run_df = pd.DataFrame({
        'gen': structs,
        'e_form': e_forms,
        'wsyms': wsyms,
        'lat_a': lat_a,
        'lat_vol': lat_vol,
        'group': groups       
    })

    df_to_json(run_df, run_dir / Path(symm.formula.replace(' ', '') + '.json'))

    symm_data = sga.get_symmetry_dataset()
    scale = conv.num_sites / symm.num_sites
    best_struct = structs[np.argmin(e_forms)]    
    best_relaxed, best_gen_e_form = relaxed_energy(deepcopy(best_struct), long=True)
    task_rows.append({
        'sg_symbol': symm_data['international'],
        'symm': symm,
        'conv': conv,
        'scale': scale,
        'wyckoffs': '_'.join(symm.wyckoff_symbols),
        'prop_generated': success_tries / tries,
        'avg_num_successful': len(structs) / tries,
        'prop_actual_success': 1 - (fail_2 / len(structs)),
        'best_struct': best_struct,
        'best_relaxed': best_relaxed,
        'best_gen_e_form': best_gen_e_form,
        'best_group': groups[np.argmin(e_forms)],
        'true_e_form': e_form(deepcopy(row['struct']))
    })

task_df = pd.concat([pd.DataFrame(task_rows).reset_index(drop=True), test_df.reset_index(drop=True)], axis=1)

try:
    df_to_json(task_df, run_dir / 'total.json')
except Exception as e:
    raise e
    print(task_df.to_string())
    
print('Complete!')