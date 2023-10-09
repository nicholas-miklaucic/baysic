"""Methodically search for candidate structures for a single composition."""

from copy import deepcopy
import logging
import numpy as np

import torch
from baysic.lattice import LATTICES, CubicLattice
from baysic.pyro_generator import SystemStructureModel
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm, trange
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from baysic.structure_evaluation import e_form, point_energy, relaxed_energy
from pymatgen.core import Composition
import pandas as pd

from baysic.utils import df_to_json

torch.manual_seed(29437)

comp = Composition("K4C2N4")
# https://next-gen.materialsproject.org/materials/mp-510376
# https://next-gen.materialsproject.org/materials/mp-11251
# https://next-gen.materialsproject.org/materials/mp-2554
# https://next-gen.materialsproject.org/materials/mp-10408


smoke_test = False
mode = 'append'


log_dir = Path('logs/')

if smoke_test:
    num_generations = 1
    max_gens_at_once = 1
else:
    num_generations = 50
    max_gens_at_once = 10
    
failure_factor = 5

date = datetime.now().strftime('%m-%d')
date_dir = log_dir / date
date_dir = date_dir / Path(comp.formula.replace(' ', ''))
if not date_dir.exists():
    date_dir.mkdir(parents=True)

run_num = 1
while (date_dir / str(run_num)).exists() and mode == 'new':
    run_num += 1
    if run_num >= 100:
        raise ValueError('You sure you want to make 100 folders for a day?')
    
run_dir = date_dir / str(run_num)
run_dir.mkdir(exist_ok=True)

group_rows = []

for lattice_type in LATTICES:
    groups = lattice_type.get_groups()
    if smoke_test:
        groups = [groups[0]]

    for i in trange(0, len(groups), colour='#1d71df', desc=lattice_type.lattice_type):
        if mode == 'append' and (run_dir / Path(f'{groups[i].number}.json')).exists():
            print(f'{groups[i].number} already done, continue')
            continue
        try:
            model = SystemStructureModel(comp, lattice_type, i)
        except ValueError as e:
            print(f'No valid Wyckoff assignments for {groups[i].number} ({groups[i].symbol})')
            continue

        structs = []
        wsyms = []
        lat_matrix = []
        lat_vol = []
        e_forms = []    
        pre_generated = 0
        success_tries = 0
        fail_2 = 0
        tries = 0
        with tqdm(total=num_generations, colour='#df1d71', desc=f'{groups[i].symbol} ({groups[i].number})') as bar:
            while len(structs) < num_generations and tries < failure_factor * num_generations:
                tries += 1
                try:
                    coords, lattice, elems, wsets, sg = model()
                except ValueError:
                    continue
                except AttributeError as e:
                    logging.error(f'AttributeError for {groups[i].number} ({groups[i].symbol})')
                    logging.error(e)
                    continue

                
                new_structs = model.to_structures()[:max_gens_at_once]
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
                    lat_matrix.append(torch.flatten(lattice).detach().cpu().numpy())
                    lat_vol.append(torch.det(lattice).item())                

        if not structs:
            print(f'Generation for group {groups[i].number} ({groups[i].symbol}) failed')
            continue

        if tries >= failure_factor * num_generations:
            print(f'Only {success_tries} successes, not {num_generations}')
        
        run_df = pd.DataFrame({
            'gen': structs,
            'e_form': e_forms,
            'wsyms': wsyms,
            'lat_matrix': lat_matrix,
            'lat_vol': lat_vol,
        })

        df_to_json(run_df, run_dir / Path(f'{groups[i].number}.json'))
        
        best_struct = structs[np.argmin(e_forms)]    
        # best_relaxed, best_gen_e_form = relaxed_energy(deepcopy(best_struct), long=True)
        group_rows.append({
            'lattice_type': lattice_type.lattice_type,
            'sg_symbol': groups[i].symbol,
            'sg_number': groups[i].number,
            'prop_generated': success_tries / tries,
            'avg_num_successful': len(structs) / tries,
            'prop_actual_success': len(structs) / (fail_2 + len(structs)),
            'best_struct': best_struct,
            # 'best_relaxed': best_relaxed,
            # 'best_gen_e_form': best_gen_e_form,
        })
    
group_df = pd.DataFrame(group_rows)

try:
    df_to_json(group_df, run_dir / 'total.json')
except Exception as e:
    raise e    
    
print('Complete!')