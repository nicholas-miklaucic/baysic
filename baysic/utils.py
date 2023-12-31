"""General utilities."""

import json
from os import PathLike
import typing
import numpy as np
import pandas as pd
from pyxtal import Group, Wyckoff_position
from pyxtal.symmetry import symbols as group_symbols

from pymatgen.core import Structure
import torch
import logging
import inspect
from functools import lru_cache
from pathlib import Path
from spglib import get_spacegroup_type

full_symbols = pd.Series([get_spacegroup_type(Group(g).hall_number)['international_short'] for g in range(1, 231)], name='group_symbol', index=range(1, 231))
full_symbols = full_symbols.str.replace(r'_(\d)', lambda m: chr(int(f'208{m.groups()[0]}', base=16)), regex=True)
full_symbols = full_symbols.str.replace(r'-(\d)', '\\1\u0305', regex=True)

def to_pretty_name(g_nums):
    g = full_symbols.loc[g_nums]
    return g.values

def to_sorted_pretty_string(comp):
    els = []
    for k in sorted(comp.keys(), key=lambda e: e.symbol):
        els.append(f'{k.symbol}{int(comp[k])}')
    return ''.join(els)


@lru_cache(maxsize=1024)
def get_wp(sg: Group, wp: int | str | Wyckoff_position) -> Wyckoff_position:
    if isinstance(wp, Wyckoff_position):
        return wp
    elif isinstance(wp, str):
        return Wyckoff_position.from_group_and_letter(sg.number, wp)
    elif isinstance(wp, int):
        return Wyckoff_position.from_group_and_index(sg.number, wp)


@lru_cache(maxsize=256)
def get_group(group: int | str | Group) -> Group:
    if isinstance(group, Group):
        return group
    elif isinstance(group, str):
        group_sym = group.strip().replace(' ', '')
        if group_sym not in group_symbols['space_group']:
            raise ValueError(f'Group {group} cannot be identified')
        else:
            return Group(group_symbols['space_group'].index(group_sym) + 1)
    else:
        return Group(group)


def quick_view(struct: Structure, port: int = 8051, **kwargs):
    import crystal_toolkit.components as ctc
    import dash
    from dash import html
    app = dash.Dash()

    component = ctc.StructureMoleculeComponent(struct, **kwargs)
    ctc.register_crystal_toolkit(app, layout=html.Div([html.H2(struct.composition.to_unicode_string()), component.layout()]))

    return app.run(port=port)


M = typing.TypeVar('M', np.ndarray, torch.Tensor)
def upper_tri(mat: M) -> M:
    """Get the upper triangle of the matrix."""
    inds0, inds1 = np.triu_indices(mat.shape[-1], 1)
    return mat[..., inds0, inds1]


def debug_shapes(*names):
    """Shows the shapes of PyTorch tensor inputs."""
    frame = inspect.currentframe().f_back.f_locals
    try:
        shapes = [frame[name].shape for name in names]
        max_len = int(max(map(len, shapes)))
        max_digits = len(str(max(map(max, shapes))))
        max_name_len = max(len(name) for name in names)
        for name, shape in zip(names, shapes):
            logging.debug(f'{name:>{max_name_len}} = ' + ' '.join([' ' * max_digits] * (max_len - len(shape)) + [f'{dim:>{max_digits}}' for dim in shape]))
    finally:
        del frame

def min_mod_1_(x: torch.Tensor):
    """Transforms x -> min(x % 1, -x % 1), without making copies."""
    # equal to 0.5 - |0.5 - (x % 1)|
    x.frac_().abs_().neg_().add_(0.5).abs_().neg_().add_(0.5).abs_()

def _pairwise_dist_ratio(c1: torch.Tensor, c2: torch.Tensor, rads1: torch.Tensor, rads2: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Gets pairwise distances (as a ratio of the radii sum) using the given lattice.

    c1: [B, 3]
    c2: [C, D, 3]
    rads1: broadcastable to [B]
    rads2: broadcastable to [D]
    lattice: [3, 3]

    returns: [C]
    """
    set_diffs = c1.unsqueeze(-2).unsqueeze(-2) - c2.unsqueeze(0)
    # set_diffs = torch.minimum(set_diffs % 1, -set_diffs % 1)
    min_mod_1_(set_diffs)
    set_diffs = torch.matmul(set_diffs, lattice.T)
    set_diffs **= 2
    # [B, C, D, 3]
    # sum over last axis
    # then take min over B and D
    diffs = torch.sum(set_diffs, dim=-1)
    diffs.sqrt_()
    rads = rads1.reshape(-1, 1, 1) + rads2.reshape(1, 1, -1)
    return (diffs / rads).min(dim=-1).values.min(dim=0).values

pairwise_dist_ratio = torch.vmap(_pairwise_dist_ratio, (0, None, None, None, None), chunk_size=128)

def _pairwise_diag_dist_ratio(c1: torch.Tensor, radius: torch.Tensor, lattice: torch.Tensor) -> torch.Tensor:
    """Gets pairwise one-vs-rest (as a ratio of the radii sum) using the given lattice.

    c1: [B, 3]
    radius: [1] or [B]
    lattice: [3, 3]

    Computes inner distance matrix [A, B - 1, 3]

    returns: [A]
    """
    set_diffs = c1[1:, :] - c1[[0], :]
    set_diffs = torch.minimum(set_diffs % 1, -set_diffs % 1)
    set_diffs = torch.matmul(set_diffs, lattice.T)
    set_diffs **= 2
    # [B - 1, 3]
    # sum over last axis
    # then take min over B and D
    diffs = torch.sum(set_diffs, dim=-1)
    diffs.sqrt_()
    rads = torch.broadcast_to(radius, c1[:, 0].shape)
    rads = rads[1:] + rads[0]
    return (diffs / rads).min(dim=-1)[0]

pairwise_diag_dist_ratio = torch.vmap(_pairwise_diag_dist_ratio, (0, None, None), chunk_size=128)

def struct_dist_ratio(struct: Structure) -> torch.Tensor:
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    coords = torch.tensor(struct.frac_coords).float()
    radii = torch.tensor([CovalentRadius.radius[site.specie.symbol] for site in struct.sites]).float()
    lat = torch.tensor(struct.lattice.matrix).float()
    return pairwise_diag_dist_ratio(coords, radii, lat)


from monty.json import MontyDecoder, MontyEncoder
def df_to_json(df: pd.DataFrame, file: PathLike):
    '''Saves the DataFrame, serializing Structure objects properly.'''
    df.to_json(file, orient='records', default_handler=MontyEncoder().default)


def json_to_df(fn: typing.Any) -> pd.DataFrame:
    '''Converts a JSON file to DataFrame, deserializing pymatgen objects as appropriate.'''
    if hasattr(fn, 'read'):
        return pd.json_normalize(json.load(fn, cls=MontyDecoder))
    with open(fn, 'r') as infile:
        return pd.json_normalize(json.load(infile, cls=MontyDecoder))


def load_mp20(split: typing.Literal['test', 'train', 'valid'],
              force_redownload: bool = False) -> pd.DataFrame:
    """Loads the MP20 dataset, downloading from the Internet if not already available or
    if force_redownload is True."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    remote_url = f'https://raw.githubusercontent.com/txie-93/cdvae/main/data/mp_20/{split}.csv'
    file_path = Path('data') / 'mp20'/ f'{split}.json'
    if not file_path.exists() or force_redownload:
        logging.info(f'Downloading MP20 and processing. This takes a couple minutes...')
        df = pd.read_csv(remote_url)
        df['struct'] = [
            Structure.from_str(cif, 'cif', primitive=True)
            for cif in df['cif']
        ]
        df['sga'] = [SpacegroupAnalyzer(s, symprec=0.01) for s in df['struct']]
        df['comp'] = [s.composition for s in df['struct']]
        df['sg_number'] = df['spacegroup.number']
        df['sg_symbol'] = df['sga'].apply(lambda sga: sga.get_space_group_symbol())
        df['conv'] = df['sga'].apply(lambda sga: sga.get_refined_structure())
        datasets = df['sga'].apply(lambda sga: sga.get_symmetry_dataset())
        for key in ['hall', 'wyckoffs', 'crystallographic_orbits', 'equivalent_atoms', 'std_mapping_to_primitive']:
            df[key] = [ds[key] for ds in datasets]
        df['lattice'] = [s.lattice for s in df['struct']]
        df['num_atoms'] = [int(comp.num_atoms) for comp in df['comp']]

        df.drop(columns=['elements', 'cif', 'spacegroup.number', 'sga'], inplace=True)
        df_to_json(df, file_path)
    else:
        df = json_to_df(file_path)
        df['sg'] = df['sg_number'].apply(get_group)

    return df

def to_np(x: torch.Tensor) -> np.array:
    """Converts to numpy."""
    return x.detach().cpu().numpy()


if __name__ == '__main__':
    test_pairwise_diag = True
    test_min_mod1 = True
    from copy import deepcopy

    if test_pairwise_diag:
        arr = torch.randn(10, 9, 3)
        old_arr = arr.clone()
        radii = torch.ones(arr.shape[1]) * 2.3
        lattice = torch.eye(3)

        dists = pairwise_diag_dist_ratio(arr, radii, lattice)
        real_dists = arr[:, [0], :] - arr[:, 1:, :]
        real_dists = torch.minimum(real_dists % 1, (-real_dists) % 1)
        real_dists = real_dists @ lattice.T
        real_dists = real_dists.square().sum(dim=-1).sqrt()
        real_dists /= (radii[[0]] + radii[1:]).unsqueeze(0)
        real_dists = real_dists.min(dim=1).values
        assert torch.allclose(arr, old_arr)
        assert torch.allclose(dists, real_dists)

    if test_min_mod1:
        arr = torch.randn(1000, 3)
        old_arr = arr.clone()
        min_mod_1_(arr)

        real_ans = torch.minimum(arr % 1, (-arr) % 1)
        assert torch.allclose(arr.abs(), real_ans.abs())