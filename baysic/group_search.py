"""Methodically search for candidate structures for a single composition."""

import warnings

from baysic.errors import BaysicError, CoordinateGenerationFailed, StructureGenerationError, WyckoffAssignmentImpossible

warnings.filterwarnings('ignore', module='.*mprester.*')

import gc
from copy import deepcopy
from dataclasses import dataclass
from turtle import update
from flask import cli
from pyrallis import field
import logging
import numpy as np

import torch
from baysic.config import FileLoggingMode, MainConfig
from baysic.lattice import LATTICES, CubicLattice
from baysic.pyro_generator import SystemStructureModel
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm, trange
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from baysic.structure_evaluation import e_form, point_energy, relaxed_energy
from pymatgen.core import Composition, Structure, Lattice
import pandas as pd
import pyrallis

from baysic.utils import df_to_json
from rich.logging import RichHandler
from rich.progress import Progress

# comp = Composition("K4C2N4")
# https://next-gen.materialsproject.org/materials/mp-510376
# https://next-gen.materialsproject.org/materials/mp-11251
# https://next-gen.materialsproject.org/materials/mp-2554
# https://next-gen.materialsproject.org/materials/mp-10408

@pyrallis.wrap(Path('configs') / 'config.toml')
def main(conf: MainConfig):
    """Runs a search to generate structures for a specific composition."""

    if conf.search.rng_seed is not None:
        torch.manual_seed(conf.search.rng_seed)

    FORMAT = "%(message)s"
    logging.basicConfig(
        level=conf.cli.verbosity.value, format=FORMAT, datefmt="[%X]", 
        handlers=[RichHandler(
            rich_tracebacks=True,
            show_time = False,
            show_level = False,
            show_path = False,            
        )]
    )

    if conf.log.use_directory:
        date = datetime.now().strftime('%m-%d')
        date_dir = conf.log.log_directory / date
        date_dir = date_dir / Path(conf.target.formula)
        if not date_dir.exists():
            date_dir.mkdir(parents=True)

        run_num = 1
        while (date_dir / str(run_num)).exists() and conf.log.log_dir_mode == FileLoggingMode.new:
            run_num += 1
            if run_num >= 100:
                raise ValueError('You sure you want to make 100 folders for a day?')
            
        run_dir = date_dir / str(run_num)
        run_dir.mkdir(exist_ok=True)
    
    big_df = []
    with Progress(disable=not conf.cli.show_progress) as progress:        
        for lattice_type in LATTICES:
            groups = lattice_type.get_groups()
            # manually specified groups override everything
            # smoke testing only requires testing two groups per system
            if conf.search.groups_to_search is not None:
                groups = [g for g in groups if g.number in conf.search.groups_to_search]
            elif conf.search.smoke_test:
                groups = groups[:2]
            

            lattice_task = progress.add_task(lattice_type.lattice_type.title(), total=len(groups))
            for group in groups:
                str_group = f'[sky_blue3] [bold] {group.number} [/bold] [italic] ({group.symbol}) [/italic] [/sky_blue3]'
                extra = {'markup': True}
                log_dir = run_dir / Path(f'{group.number}.json')
                if conf.log.log_dir_mode == FileLoggingMode.append and log_dir.exists():
                    logging.info(f'{str_group} already done, continue', extra=extra)
                    progress.update(lattice_task, advance=1)
                    continue
                try:
                    model = SystemStructureModel(conf.log, conf.search, conf.target.composition, lattice_type, group)
                except WyckoffAssignmentImpossible as e:
                    logging.info(f'No valid Wyckoff assignments for {str_group}', extra=extra)
                    progress.update(lattice_task, advance=1)
                    continue
                
                rows = []
                total_allowed = round(conf.search.allowed_attempts_per_gen * conf.search.num_generations)
                group_task = progress.add_task(str_group, total=conf.search.num_generations)
                for gen_attempt in range(total_allowed):
                    if len(rows) >= conf.search.num_generations:
                        break

                    try:
                        coords, lattice, elems, wsets, sg = model()
                        log_info = deepcopy(model.log_info)                        
                    except CoordinateGenerationFailed:
                        continue
                    except AttributeError as e:
                        logging.error(f'AttributeError for {str_group}', extra=extra)
                        logging.error(e)
                        continue

                    
                    new_structs = model.to_structures()[:conf.search.max_gens_at_once]                    
                    for struct in new_structs:
                        e_form_val = point_energy(deepcopy(struct))
                        if e_form_val > 80:
                            continue

                        progress.update(group_task, advance=1)
                        row = {
                            'struct': struct,
                            'e_form': e_form_val,
                            'lat_matrix': lattice.detach().cpu().numpy(),
                            'gen_attempt': gen_attempt,                            
                        }
                        row.update(log_info)
                        rows.append(row)                    

                progress.update(lattice_task, advance=1)                    
                if gen_attempt == total_allowed:
                    # ran out of attempts
                    progress.stop_task(group_task)
                    if rows:
                        logging.warning(f'{str_group}: Only {len(rows)} successes, not {conf.search.num_generations}', extra=extra)
                    else:
                        logging.info(f'{str_group}: Generation failed', extra=extra)                    
                    continue

                group_df = pd.DataFrame(rows)
                group_df['group_number'] = group.number
                group_df['group_symbol'] = group.symbol
                group_df['lattice_type'] = lattice_type.lattice_type
                group_df['num_attempts'] = gen_attempt
                big_df.append(group_df)

                if conf.log.use_directory:
                    df_to_json(group_df, run_dir / Path(f'{group.number}.json'))
        
    big_df = pd.concat(big_df).reset_index(drop=True)
    if conf.log.use_directory:
        df_to_json(big_df, run_dir / Path(f'total.json')) 
        
    print('Complete!')


if __name__ == '__main__':
    main()