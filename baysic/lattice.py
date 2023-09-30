import functools
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pymatgen.core import Composition, Lattice
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pyxtal import Group

system_groups = {k: [] for k in ['monoclinic', 'tetragonal', 'cubic', 'hexagonal', 'triclinic', 'orthorhombic']}
for g_num in range(2, 231):
    g = Group(g_num)
    system_groups[g.lattice_type].append(g)


def atomic_volume(comp: Composition) -> float:
    """Atomic volume using covalent radii."""
    radii = torch.tensor([CovalentRadius.radius[el.symbol] for el in comp.elements])
    count = torch.tensor(list(comp.values())).float()
    return torch.dot(count, 4 * torch.pi / 3 * radii ** 3).item()


class LatticeModel(PyroModule):
    lattice_type = 'base'
    """Stochastic representation of a lattice."""
    def __init__(self, volume: float):
        """
        volume (float): Lattice volume in angstroms cubed.
        """
        super().__init__()
        self.volume = volume       

    def forward(self) -> torch.Tensor:
        """Returns the matrix for the lattice."""
        raise NotImplementedError()

    def to_lattice(self, params: torch.Tensor) -> Lattice:
        """Makes lattice with the given parameters."""        
        return Lattice(self().numpy())
        
    @classmethod
    def get_groups(cls) -> list[Group]:
        """Get groups matching a given lattice type."""
        return system_groups[cls.lattice_type]


class CubicLattice(LatticeModel):
    lattice_type = 'cubic'
    """A cubic lattice."""    
    def forward(self) -> torch.Tensor:
        '''Returns the lattice matrix.'''
        # nothing to sample
        return torch.diag(torch.ones(3) * self.volume ** (1/3))

if __name__ == '__main__':
    mod = CubicLattice(atomic_volume(Composition({'O': 3, 'Sr': 1, 'Ti': 1})) * 1.3)
    samps = torch.tensor([mod.forward()[0] for _ in range(100)])
    # actual value: 3.91, and pretty tightly packed at that
    print(f'{samps.mean():.3f} ± {samps.std():.3f}')
    print(mod.to_lattice(mod.forward()))
