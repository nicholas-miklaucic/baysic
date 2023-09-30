"""General utilities."""

import json
from os import PathLike
import os
import typing
import numpy as np
import pandas as pd
from pyxtal import Group, Wyckoff_position
from pyxtal.symmetry import symbols as group_symbols

import crystal_toolkit.components as ctc
import dash
from dash import html
from pymatgen.core import Structure
import torch
import logging
import inspect

def get_wp(sg: Group, wp: int | str | Wyckoff_position) -> Wyckoff_position:
    if isinstance(wp, Wyckoff_position):
        return wp
    elif isinstance(wp, str):
        return Wyckoff_position.from_group_and_letter(sg.number, wp)
    elif isinstance(wp, int):
        return Wyckoff_position.from_group_and_index(sg.number, wp)
    

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

def pairwise_dist_ratio(c1, c2, rads1, rads2, lattice):
    """Gets pairwise distances (as a ratio of the radii sum) using the given lattice.

    c1: [A, B, 3]
    c2: [C, D, 3]
    rads1: broadcastable to [A, B]
    rads2: broadcastable to [C, D]
    lattice: [3, 3]

    returns: [C, D, A, B]
    """
    set_diffs = c1.unsqueeze(0).unsqueeze(0) - c2.unsqueeze(-2).unsqueeze(-2)
    set_diffs = set_diffs % 1
    set_diffs = torch.minimum(set_diffs, 1 - set_diffs)
    set_cart_diffs = torch.matmul(set_diffs, lattice.T)                
    diffs = torch.sqrt(torch.sum(torch.square(set_cart_diffs), axis=-1))
    rads = rads1.unsqueeze(0).unsqueeze(1) + rads2.unsqueeze(-1).unsqueeze(-1)
    return diffs / rads


from monty.json import MontyDecoder, MontyEncoder
def df_to_json(df: pd.DataFrame, file: PathLike):
    '''Saves the DataFrame, serializing Structure objects properly.'''
    df.to_json(file, orient='records', default_handler=MontyEncoder().default)


def json_to_df(fn: PathLike) -> pd.DataFrame:
    '''Converts a JSON file to DataFrame, deserializing pymatgen objects as appropriate.'''
    with open(fn, 'r') as infile:
        return pd.json_normalize(json.load(infile, cls=MontyDecoder))