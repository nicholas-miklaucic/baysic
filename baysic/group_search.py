"""Methodically search for candidate structures for a single composition."""

import warnings

from baysic.errors import BaysicError, CoordinateGenerationFailed, StructureGenerationError, WyckoffAssignmentFailed, WyckoffAssignmentImpossible

warnings.filterwarnings('ignore', module='.*mprester.*')

import gc
import os
from copy import deepcopy
from dataclasses import dataclass
from pyrallis import field
import logging
import numpy as np

import torch
from baysic.config import FileLoggingMode, MainConfig, TargetStructureConfig
from baysic.lattice import LATTICES, CubicLattice, LatticeModel
from pyxtal import Group
from baysic.pyro_generator import SystemStructureModel
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm, trange
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from baysic.structure_evaluation import e_forms, point_energies, point_energy, relaxed_energy
from pymatgen.core import Composition, Structure, Lattice
import pandas as pd
import pyrallis

from baysic.utils import df_to_json
from rich.logging import RichHandler
from rich.progress import Progress
import rich.progress as prog

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from queue import Empty, Queue

# comp = Composition("K4C2N4")
# https://next-gen.materialsproject.org/materials/mp-510376
# https://next-gen.materialsproject.org/materials/mp-11251
# https://next-gen.materialsproject.org/materials/mp-2554
# https://next-gen.materialsproject.org/materials/mp-10408

def search_group(
    conf: MainConfig,
    lattice_type: LatticeModel,
    group: Group,
    str_group: str,
    run_dir: Path | None,
    progress_queue: Queue,
    task_id: int):
    extra = {'markup': True}
    if run_dir is not None:
        log_dir = run_dir / Path(f'{group.number}.json')
        if conf.log.log_dir_mode == FileLoggingMode.append and log_dir.exists():
            logging.info(f'{str_group} already done, continue', extra=extra)
            return conf.search.num_generations
    try:
        model = SystemStructureModel(conf.log, conf.search, conf.target.composition, lattice_type, group)
    except WyckoffAssignmentImpossible as e:
        # logging.info(f'No valid Wyckoff assignments for {str_group}', extra=extra)
        return conf.search.num_generations

    rows = []
    total_allowed = round(conf.search.allowed_attempts_per_gen * conf.search.num_generations)

    # this is a special message, because start_task is different from update_task
    progress_queue.put({'task_id': task_id, 'start': True})
    progress_queue.put({'task_id': task_id, 'visible': True})
    for gen_attempt in range(total_allowed):
        if len(rows) >= conf.search.num_generations:
            break

        try:
            # gc.collect()
            coords, lattice, elems, wsets, sg = model()
            log_info = deepcopy(model.log_info)
        except WyckoffAssignmentFailed as e:
            # this ideally shouldn't be happening
            # track these failures more closely
            logging.exception(e)
            continue
        except CoordinateGenerationFailed:
            continue
        except AttributeError as e:
            logging.error(f'AttributeError for {str_group}', extra=extra)
            logging.error(e)
            continue


        new_structs = model.to_structures()[:conf.search.max_gens_at_once]
        e_form_vals = e_forms(new_structs, conf.device.device)
        for struct, e_form_val in zip(new_structs, e_form_vals):
            if e_form_val > 80:
                continue

            progress_queue.put({'task_id': task_id, 'advance': 1})

            row = {
                'struct': struct,
                'e_form': e_form_val,
                'lat_matrix': lattice.detach().cpu().numpy(),
                'gen_attempt': gen_attempt,
            }
            row.update(log_info)
            rows.append(row)


    if gen_attempt == total_allowed:
        # ran out of attempts
        if rows:
            logging.warning(f'{str_group}: Only {len(rows)} successes, not {conf.search.num_generations}', extra=extra)
        else:
            logging.info(f'{str_group}: Generation failed', extra=extra)
    else:
        if conf.log.use_directory:
            group_df = pd.DataFrame(rows)
            group_df['group_number'] = group.number
            group_df['group_symbol'] = group.symbol
            group_df['lattice_type'] = lattice_type.lattice_type
            group_df['num_attempts'] = gen_attempt
            df_to_json(group_df, run_dir / Path(f'{group.number}.json'))

    progress_queue.put({'task_id': task_id, 'visible': False})
    return conf.search.num_generations - len(rows)


def main_(conf: MainConfig):
    """Runs a search to generate structures for a specific composition."""
    torch.set_default_device(conf.device.device)
    torch.set_num_threads(conf.device.torch_threads)
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
        date_dir = conf.log.log_directory / Path(conf.target.formula)
        if not date_dir.exists():
            date_dir.mkdir(parents=True)

        run_num = 1
        while (date_dir / str(run_num)).exists() and conf.log.log_dir_mode == FileLoggingMode.new:
            run_num += 1
            if run_num >= 100:
                raise ValueError('You sure you want to make 100 folders for a day?')

        run_dir = date_dir / str(run_num)
        run_dir.mkdir(exist_ok=True)

    with Progress(
        prog.TextColumn('[progress.description]{task.description}'),
        prog.BarColumn(80,
                       'light_pink3',
                       'deep_sky_blue4',
                       'green'),
        prog.MofNCompleteColumn(),
        prog.TimeElapsedColumn(),
        prog.TimeRemainingColumn(),
        prog.SpinnerColumn(),
        refresh_per_second=3,
        disable=not conf.cli.show_progress) as progress:
        total = 0
        lat_groups = {}
        for lattice_type in LATTICES:
            groups = lattice_type.get_groups()
            # manually specified groups override everything
            # smoke testing only requires testing two groups per system
            if conf.search.groups_to_search is not None:
                groups = [g for g in groups if g.number in conf.search.groups_to_search]
            elif conf.search.smoke_test:
                groups = groups[:2]

            lat_groups[lattice_type.lattice_type] = groups
            total += len(groups) * conf.search.num_generations


        total_task = progress.add_task(
            f'[bold] [deep_pink3] {conf.target.formula} [/deep_pink3] Generation [/bold]',
            total=total)


        num_avail_cpus = len(os.sched_getaffinity(0))
        if conf.device.max_workers <= 0:
            # represents number of threads to *not* use
            # anything above 30 should be manually specified
            max_workers = min(num_avail_cpus - conf.device.max_workers, 30)
        else:
            max_workers = min(num_avail_cpus, conf.device.max_workers)

        futures = []
        if conf.cli.show_progress:
            manager = mp.Manager()
            progress_queue = manager.Queue()
        else:
            progress_queue = None
        with ProcessPoolExecutor(max_workers) as executor:
            for lattice_type in LATTICES:
                groups = lat_groups[lattice_type.lattice_type]

                for group in groups:
                    str_group = f'[sky_blue3][bold]{group.number}[/bold][italic] ({group.symbol})[/italic][/sky_blue3]'
                    group_task = progress.add_task(str_group, total=conf.search.num_generations, visible=False, start=False)
                    future = executor.submit(search_group, conf, lattice_type, group, str_group, run_dir, progress_queue if progress_queue is not None else Queue(), group_task)
                    def process_result(f):
                        try:
                            total_change = f.result()
                            progress.update(total_task, advance=total_change)
                        except Exception as e:
                            logging.exception(e)

                    if conf.cli.show_progress:
                        future.add_done_callback(process_result)
                    futures.append(future)

            while any(not future.done() for future in futures):
                if conf.cli.show_progress:
                    while not progress_queue.empty():
                        update = progress_queue.get(timeout=10)
                        if 'start' in update:
                            progress.start_task(update['task_id'])
                        else:
                            progress.update(**update)
                            if 'advance' in update:
                                progress.update(total_task, advance=update['advance'])

    print('Complete!')


@pyrallis.wrap()
def main(main: MainConfig):
    """Searches for structures."""
    main_(main)

if __name__ == '__main__':
    mp.set_start_method('fork', force=True)
    main()
    # from baysic.config import TargetStructureConfig, MainConfig
    # main_(MainConfig(target=TargetStructureConfig('mp-20674')))
