"""Code to perform Wyckoff assignments."""

import os

from pathlib import Path
from baysic.errors import WyckoffAssignmentCacheError
from baysic.utils import get_group
from pyxtal import Group
from collections import defaultdict
import numpy as np
import itertools
from pymatgen.core import Composition
import pickle
from rich.progress import track
from rich.prompt import Confirm

class Wyckoffs:
    """A representation of a set of Wyckoff positions suitable for assignment."""
    def __init__(self, group: int | str | Group, n_max: int = 256) -> None:
        """
        n_max is the maximum number of atoms of a single type the object
        will support.
        """
        self.group = get_group(group)        
        self.general = set()
        special_counts = defaultdict(int)
        for wp in self.group.Wyckoff_positions:
            if wp.get_dof() == 0:
                special_counts[wp.multiplicity] += 1
            else:
                self.general.add(wp.multiplicity)

        self.special_counts = []
        self.special_mults = []
        for k in sorted(special_counts.keys()):
            self.special_mults.append(k)
            self.special_counts.append(special_counts[k])

        self.n_max = n_max
        self._build(self.n_max)

    def __repr__(self):
        return f'''    
Group {self.group.number}
General: {" ".join(map(str, self.general))}
Special:
Mult. | {" | ".join(map(str, self.special_mults))}
# WPs | {" | ".join(map(str, self.special_counts))}
'''    

    @staticmethod
    def pareto_front(pts: np.array) -> np.array:
        front = []
        for pt in pts:
            if not any(all(pt >= front_pt) for front_pt in front):
                front.append(pt)

        return np.array(front, dtype=int)
    
    def _build(self, n_max: int):   
        ndim = len(self.special_mults)        
        def zero() -> np.array:
            return np.zeros((1, ndim), dtype=int)

        all_counts = [zero()]

        self.limit = np.array(self.special_counts)

        costs = []
        mults = []
        for j_vec, j_mult in zip(np.eye(ndim), self.special_mults):
            if j_mult not in self.general:
                costs.append(j_vec)
                mults.append(j_mult)

        for mult in self.general:
            costs.append(zero())
            mults.append(mult)

        O = tuple(np.zeros(ndim))
        for k in range(1, n_max + 1):
            all_vecs = set()
            for cost, mult in zip(costs, mults):        
                if mult <= k and all_counts[k - mult] is not None:        
                    vecs = all_counts[k - mult] + cost
                    all_vecs.update(tuple(vec) for vec in vecs[np.all(vecs <= self.limit, axis=1)])
            
            if O in all_vecs:
                all_counts.append(zero())
            elif all_vecs:
                all_counts.append(self.pareto_front(np.array(list(all_vecs))))
            else:
                all_counts.append(None)

        
        self.frontiers = {}        

        for i, counts in enumerate(all_counts):
            if i == 0 or counts is None:
                continue
            else:
                self.frontiers[i] = counts                
        
        self.n_dim = ndim        

    def _can_make_single(self, card: int) -> bool:
        if card > self.n_max:
            raise WyckoffAssignmentCacheError(f'Only computed values up to {self.n_max}, not {card}')
        
        return card in self.frontiers
        
    def can_make(self, num_atoms: list[int] | Composition) -> bool:
        if isinstance(num_atoms, Composition):
            num_atoms = list(num_atoms.values())

        int_num_atoms = np.array(num_atoms).astype(int)
        if not np.allclose(int_num_atoms, num_atoms):
            raise WyckoffAssignmentCacheError(f'Does not support fractional composition: {num_atoms}')

        # also checks for too many atoms
        if not all(self._can_make_single(card) for card in int_num_atoms):
            return False
                
        # can we choose vectors v from self.frontiers[c], for each
        # c in num_atoms, such that the sum of all v <= self.limit?
        choices = [range(len(self.frontiers[c])) for c in int_num_atoms]
        frontiers = [self.frontiers[c] for c in num_atoms]
        for choice in itertools.product(*choices):
            if np.all(sum(vec[i] for vec, i in zip(frontiers, choice)) <= self.limit):
                return True
            
        return False

cache_path = Path('precomputed') / 'wyckoffs.pkl'
if cache_path.exists():
    with open(cache_path, 'rb') as infile:
        WYCKOFFS = pickle.load(infile)        

if not cache_path.exists() and __name__ != '__main__':
    # don't allow name to be __main__, which breaks pickling
    # we want the namespace of the above load() to be the same
    if Confirm.ask('Generate Wyckoff assignment cache?'):
        WYCKOFFS = [None]    
        for group in track(range(1, 231), description='Generating WP assignments...'):
            WYCKOFFS.append(Wyckoffs(group, n_max=256))
        
        with open(cache_path, 'wb') as outfile:
            pickle.dump(WYCKOFFS, outfile)    
    else:
        print('Exiting: things will probably break from here!') 