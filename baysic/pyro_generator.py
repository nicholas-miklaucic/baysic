"""Full stochastic structure generator using Pyro."""

from copy import deepcopy
from ctypes.wintypes import WPARAM
from doctest import debug
import logging
from math import floor
from signal import struct_siginfo
import numpy as np
from sympy import N
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pymatgen.core import Composition, Lattice, Structure, Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from tqdm import trange
from baysic.structure_evaluation import MIN_DIST_RATIO, e_form, point_energy
from baysic.pyro_wp import WyckoffSet
from baysic.lattice import CubicLattice, atomic_volume, LatticeModel
from baysic.interpolator import LinearSpline
from baysic.feature_space import FeatureSpace
from pyxtal import Group, Wyckoff_position
from baysic.utils import get_group, get_wp, debug_shapes, pairwise_dist_ratio
from cctbx.sgtbx.direct_space_asu.reference_table import get_asu
from scipy.spatial import ConvexHull
import networkx as nx


ngrid = 12
nbeam = 1000

class SystemStructureModel(PyroModule):
    """A stochastic structure generator working within a particular lattice type."""    
    def __init__(self, comp: Composition, lattice: LatticeModel):
        super().__init__()
        self.comp = comp
        self.lattice_model = lattice
        
        # mode 4.5/5, mean 5.5/5
        # around 1, matches empirical distribution well
        self.volume_ratio = PyroSample(dist.Gamma(5.5, 5))            
        # self.volume_ratio = PyroSample(dist.Gamma(18, 15))            
        self.atom_volume = atomic_volume(comp)

        groups = self.lattice_model.get_groups()
        self.group_options = []
        self.wyckoff_options = []
        self.group_cards = []
        self.opt_cards = []
        self.count_cards = []
        self.inds = []

        n_els = np.array(list(comp.values()))        
        for sg in groups:
            combs, _has_freedom, _inds = sg.list_wyckoff_combinations(n_els)
            if combs:
                self.group_options.extend([sg.number] * len(combs))
                self.wyckoff_options.extend(combs)
                self.group_cards.extend([1 / len(combs)] * len(combs))
                self.opt_cards.extend([1] * len(combs)) 
                self.count_cards.extend([len(sum(comb, [])) + 1 for comb in combs])

        self.group_cards = torch.tensor(self.group_cards).float()
        self.group_cards /= self.group_cards.sum().float()
        self.opt_cards = torch.tensor(self.opt_cards).float()
        self.opt_cards /= self.opt_cards.sum().float()
        self.count_cards = torch.tensor(self.count_cards).float()
        self.count_cards = 0.2 ** (self.count_cards - min(self.count_cards))
        self.count_cards /= self.count_cards.sum().float()
        
        self.wyck_opt = PyroSample(dist.Categorical(probs=self.count_cards))      
                
        
    def forward(self):
        self.volume = self.volume_ratio * self.atom_volume
        self.lattice = self.lattice_model(self.volume)()
        
        opt = self.wyck_opt        
        self.sg = self.group_options[opt]
        comb = self.wyckoff_options[opt]
                
        self.coords = torch.tensor([])
        self.elems = []
        self.wsets = []        
        spots = sum(comb, [])
        elements = sum([[elem] * len(spots) for elem, spots in zip(self.comp.elements, comb)], [])
        wsets: list[WyckoffSet] = [WyckoffSet(self.sg, spot) for spot in spots]
        dofs: list[int] = np.array([wset.dof for wset in wsets])
        # WPs with 0 degrees of freedom should go first, because they're very cheap to expand out
        # then, letting the high-multiplicity elements go first is best
        # they're the toughest to place, and thus make the best use of parallelism
        mults = np.array([wset.multiplicity for wset in wsets])
        mult_order = np.argsort(-mults)
        no_dofs = mult_order[dofs[mult_order] == 0]
        some_dofs = mult_order[dofs[mult_order] != 0]
        best_order = np.concatenate([no_dofs, some_dofs])

        elements = np.array(elements)[best_order]
        wsets = np.array(wsets)[best_order]
        spots = np.array(spots)[best_order]

        for spot, elem, wset in zip(spots, elements, wsets):
            radius = torch.tensor([CovalentRadius.radius[elem.symbol]])

            wset = WyckoffSet(self.sg, spot)
            if wset.dof == 0:
                posns = torch.zeros(3)
                set_coords = wset.to_all_positions(posns)
            else:
                base = torch.cartesian_prod(*[torch.linspace(0, 1, ngrid + 2)[1:-1] for _ in range(wset.dof)])
                debug_shapes('base')
                base = base.reshape(ngrid ** wset.dof, wset.dof)
                max_move = 0.49 / (ngrid + 1)
                low = base - max_move
                high = base + max_move
                posns = pyro.sample(f'coords_{len(self.elems)}', dist.Uniform(low, high))
            
                set_coords = wset.to_all_positions(wset.to_asu(posns))
                
            debug_shapes('set_coords', 'posns')
            if set_coords.shape[-2] > 1:
                # check pairwise distances
                set_diffs = pairwise_dist_ratio(set_coords[..., 1:, :], set_coords[..., [0], :], radius, radius, self.lattice)
                debug_shapes('set_diffs')
                # [ngrid, 1, ngrid, dof - 1] if used a grid search
                # [1, 1, 1, dof - 1] if no degrees of freedom
                # here, we only care about comparing a single WP to its own copies, not the full pairwise
                n_new_coords = set_diffs.shape[0]
                set_diffs = set_diffs[torch.arange(n_new_coords), 0, torch.arange(n_new_coords), :].reshape(-1, set_diffs.shape[-1])
                # [ngrid, dof - 1]
                debug_shapes('set_diffs')
                set_valid = (set_diffs >= MIN_DIST_RATIO).all(dim=-1)
                debug_shapes('set_valid')
            else:
                # 1 coordinate is always valid
                set_valid = torch.Tensor([1])
        
            if not set_valid.any():
                raise ValueError('Could not find assignment')
            
            debug_shapes('set_coords', 'set_valid')
            good_all_coords = set_coords[torch.where(set_valid)[0], :, :]
            # only need to check base coord
            good_coords = good_all_coords[:, :1, :]
            
            
            if self.coords.numel():                    
                radii = torch.tensor([CovalentRadius.radius[el.symbol] for el in self.elems])
                coords = self.coords                          
                debug_shapes('good_coords', 'coords', 'radius', 'radii') 
                # print(self.elems, self.wsets, wset.multiplicity)
                cdists = pairwise_dist_ratio(good_coords, coords, radius, radii, self.lattice)
                # shape [coords_batch, coords_num, good_batch, good_num]
                
                min_cdists = cdists.permute((0, 2, 1, 3)).min(dim=-1)[0].min(dim=-1)[0]
                # shape [coords_batch, good_batch]                    
            
                if not (min_cdists >= MIN_DIST_RATIO).any():
                    raise ValueError('Could not find assignment')
                
                # take the best nbeam pairs of (old_coords, new_coords) that work
                all_old, all_new = torch.where(min_cdists >= MIN_DIST_RATIO)
                adds = torch.argsort(min_cdists[all_old, all_new], descending=True)[:nbeam]

                old = self.coords[all_old[adds]]
                new = good_all_coords[all_new[adds]]
                debug_shapes('old', 'new')
                self.coords = torch.cat([old, new], dim=1)                
                # self.coords.append(set_coords[torch.where(set_valid)[0][0]].unsqueeze(0))

            else:
                # no other coordinates to worry about, just add all found coordinates
                self.coords = good_all_coords
                
            self.elems.extend([elem] * wset.multiplicity)
            self.wsets.append(wset)
                        
        return (self.coords, self.lattice, self.elems, self.wsets, self.sg)
    
    def to_structures(self) -> list[Structure]:
        np_coords = self.coords.detach().cpu().numpy()
        return [Structure(self.lattice, self.elems, coords) for coords in np_coords]
    
    def to_gen_coords(self) -> torch.Tensor:
        """Gets just the free coordinates."""
        curr_i = 0
        coords = []
        for wset in self.wsets:
            if wset.dof == 0:
                # no general coordinates to add
                curr_i += wset.multiplicity
                continue
            else:
                coords.append(wset.inverse(self.coords[..., curr_i, :]))
                curr_i += wset.multiplicity
        return torch.cat(coords, dim=-1)


    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    torch.manual_seed(34761)
    mod = SystemStructureModel(
        Composition({'Mg': 8, 'Al': 16, 'O': 32}),
        # Composition.from_dict({'K': 8, 'Li': 4, 'Cr': 4, 'F': 24}),        
        # Composition.from_dict({'Sr': 3, 'Ti': 1, 'O': 1}),
        CubicLattice
    )

    structs = []
    success = []
    actual_success = []
    for _ in trange(10):
        try:
            coords, lat, elems, wsets, sg = mod.forward()
            new_structs = mod.to_structures()
            print(len(new_structs))
            structs.extend(new_structs)
            actual_success.extend([point_energy(deepcopy(struct)) < 80 for struct in new_structs])
            success.append(len(new_structs))
        except ValueError:
            success.append(0)

    print(np.mean(np.array(success) > 0), np.mean(success), np.mean(actual_success))