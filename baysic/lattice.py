import functools
from numpy import stack
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

    def to_lattice(self) -> Lattice:
        """Makes lattice with the given parameters."""
        return Lattice(self().numpy())

    @classmethod
    def get_groups(cls) -> list[Group]:
        """Get groups matching a given lattice type."""
        return system_groups[cls.lattice_type]


class CubicLattice(LatticeModel):
    """A cubic lattice."""
    lattice_type = 'cubic'
    def forward(self) -> torch.Tensor:
        '''Returns the lattice matrix.'''
        # nothing to sample
        return torch.diag(torch.ones(3) * self.volume ** (1/3))

class TetragonalLattice(LatticeModel):
    """A tetragonal lattice. Here, always set up so a == b."""

    lattice_type = 'tetragonal'

    def __init__(self, volume: float):
        super().__init__(volume)

        # lattice volume is ca^2
        # sample c / a, very rough match
        # in the future, this could be more specific: c / a is often roughly an integer or the reciprocal of an integer
        # definitely room for improvement
        self.c_over_a = PyroSample(dist.Gamma(2.75, 1.29))

    def forward(self) -> torch.Tensor:

        # a x a x c = volume
        # a^3 = volume / (c / a)
        self.a = (self.volume / self.c_over_a) ** (1/3)
        self.c = self.volume / self.a ** 2
        return torch.diag(torch.tensor([self.a, self.a, self.c]))


class TriclinicLattice(LatticeModel):
    """A triclinic lattice."""
    lattice_type = 'triclinic'

    def __init__(self, volume: float):
        super().__init__(volume)

        # pyxtal uses Gaussians and generates a shear matrix at random.
        # Here we just generate angles, get the scaling factor, and then
        # partition it into a, b, c
        self.alpha = PyroSample(dist.Gamma(63, 0.7))
        self.beta = PyroSample(dist.Gamma(63, 0.7))
        self.gamma = PyroSample(dist.Gamma(63, 0.7))
        # see orthorhombic lattice for strategy
        self.abc_lat = OrthorhombicLattice(volume)

    def forward(self) -> torch.Tensor:
        aby = torch.tensor([self.alpha, self.beta, self.gamma]).deg2rad()
        c_a, c_b, c_y = aby.cos()
        s_a, s_b, s_y = aby.sin()

        scale_factor = torch.sqrt(1 - c_a ** 2 - c_b ** 2 - c_y ** 2 + 2 * c_a * c_b * c_y)


        self.abc = torch.diag(self.abc_lat.forward())
        self.abc /= scale_factor ** (1/3)
        a, b, c = self.abc
        # https://www.ucl.ac.uk/~rmhajc0/frorth.pdf

        c2 = (c_b * c_y - c_a) / (s_b * s_y)
        # clip to avoid NaNs
        a_star = torch.arccos(torch.clip(c2, -1, 1))
        c_a_star = torch.cos(a_star)
        s_a_star = torch.sin(a_star)
        return torch.tensor([
            [a,  b * c_y,  c * c_b],
            [0,  b * s_y, -c * s_b * c_a_star],
            [0., 0.,       c * s_b * s_a_star]])


class OrthorhombicLattice(LatticeModel):
    """An orthogonal lattice."""
    lattice_type = 'orthorhombic'

    def __init__(self, volume: float):
        super().__init__(volume)

        # There is some stuff going on here that's beyond my ken.
        # There are huge spikes in the distribution of a, b, and c,
        # which I think is due to certain geometrical motifs appearing:
        # preferred bond angles, etc.
        # I'm not sure this matters for simple generation, but it's
        # challenging to find the patterns. The real solution is probably
        # dependent on the space group and should be tackled with something
        # like neural flows.

        # we want to generate a, b, c such that abc = volume
        # generate a, b
        # c = 1 / ab
        # then scale by volume ^ (1/3)
        self.a = PyroSample(dist.InverseGamma(9.783, 7.286))
        self.b = PyroSample(dist.Gamma(6.402, 6.358))

    def forward(self) -> torch.Tensor:
        a, b = self.a, self.b
        c = 1 / (a * b)
        abc = torch.tensor([a, b, c])
        abc *= self.volume ** (1/3)
        return torch.diag(abc)

class MonoclinicLattice(LatticeModel):
    """Monoclinic lattice of dimensions a x b x c with non right-angle
    beta between lattice vectors a and c."""
    lattice_type = 'monoclinic'
    def __init__(self, volume: float):
        super().__init__(volume)

        self.beta = PyroSample(dist.Gamma(63, 0.7))
        # see orthorhombic lattice for strategy
        self.abc_lat = OrthorhombicLattice(volume)

    def forward(self) -> torch.Tensor:
        beta = torch.deg2rad(self.beta)
        c_b = torch.cos(beta)
        s_b = torch.sin(beta)
        scale_factor = s_b

        self.abc = torch.diag(self.abc_lat.forward())
        self.abc /= scale_factor ** (1/3)
        a, b, c = self.abc

        return torch.tensor([
            [a,       0., 0.],
            [0.,      b,  0.],
            [c * c_b, 0., c * s_b]])


class HexagonalLattice(LatticeModel):
    """Hexagonal lattice."""
    lattice_type = 'hexagonal'
    def __init__(self, volume: float):
        super().__init__(volume)

        # most hexagonal lattices aren't super weirdly shaped
        # we want to generate a, b, c such that abc = volume
        # generate c tightly around 1
        # a = sqrt(1 / c)
        # then scale by volume ^ (1/3) and sqrt(3)/2 factor
        self.a_over_c = PyroSample(dist.FoldedDistribution(
            dist.SoftAsymmetricLaplace(
                loc=0.078, scale=.081, asymmetry=0.116, softness=0.122
            )))

    def forward(self) -> torch.Tensor:
        # ca^2 = 1
        # a^3 = a/c
        a = self.a_over_c ** (1/3)
        c = 1 / (a ** 2)
        ac = torch.tensor([a, c])
        scale_factor = torch.sqrt(torch.tensor([3.])) / 2.
        ac *= (self.volume / scale_factor) ** (1/3)
        a_scaled, c_scaled = ac
        return torch.tensor([
            [a_scaled / 2, -scale_factor * a_scaled, 0.],
            [a_scaled / 2, +scale_factor * a_scaled, 0.],
            [0.,           0.,                       c_scaled]
        ])

# for progress bars, it's nice if this is roughly balanced in terms of difficulty
LATTICES: list[LatticeModel] = [
    HexagonalLattice, TetragonalLattice, OrthorhombicLattice,
    TriclinicLattice, MonoclinicLattice, CubicLattice]

if __name__ == '__main__':
    for lat in LATTICES:
        print(lat.lattice_type)
        vol = atomic_volume(Composition({'O': 3, 'Sr': 1, 'Ti': 1})) * 1.3
        mod = lat(vol)
        mats = []
        lats = []
        for _ in range(100):
            lats.append(mod.to_lattice())
            mats.append(torch.tensor(lats[-1].matrix))
            assert abs(lats[-1].volume - vol) <= 1e-3

        mats = torch.stack(mats, dim=0)
        # actual value: 3.91, and pretty tightly packed at that
        print(f'Mean:\n{mats.mean(dim=0).round(decimals=2)}\nSD:\n{mats.std(dim=0).round(decimals=2)}')
