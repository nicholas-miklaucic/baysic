"""Pyro model for a specific Wyckoff position."""

from math import floor
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pymatgen.core import Composition, Lattice
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from baysic.interpolator import LinearSpline
from baysic.feature_space import FeatureSpace
from pyxtal import Group, Wyckoff_position
from baysic.utils import get_group, get_wp
from cctbx.sgtbx.direct_space_asu.reference_table import get_asu
from scipy.spatial import ConvexHull
import networkx as nx
from torch.distributions import AffineTransform

class WyckoffSet(torch.nn.Module):
    """A stochastic representation of a Wyckoff set."""
    def __init__(self, group: int | str, wp: int | str):
        """Initializes representation of a specific Wyckoff position."""
        super().__init__()
        self.group = get_group(group)
        self.asu = get_asu(self.group.number)
        self.wp = get_wp(self.group, wp)
        self.ops = torch.stack([
            torch.tensor(op.affine_matrix)
            for op in self.wp.ops
        ])

        self._initialize_space()

    @property
    def dof(self) -> int:
        return int(self.wp.get_dof())

    @property
    def multiplicity(self) -> int:
        return len(self.wp.ops)

    def forward(self) -> torch.Tensor:
        """Returns a new Wyckoff general position."""
        self.xyz = PyroSample(dist.Uniform(torch.zeros(3), torch.ones(3)))

    def _initialize_space(self):
        verts = np.array(self.asu.shape_vertices()).astype(float)

        xyverts = verts[:, [0, 1]]
        xyhull = ConvexHull(xyverts)

        self.x_trans = AffineTransform(
            loc=torch.tensor(xyverts[:, 0].min()),
            scale=torch.tensor(xyverts[:, 0].ptp())
        )


        edges = []
        for i, j in xyhull.simplices:
            xi, xj = xyverts[:, 0][[i, j]]
            if xi == xj:
                # can only happen at end or start, ignore
                pass
            else:
                if xi > xj:
                    i, j = j, i
                edges.append((i, j))

        xygraph = nx.DiGraph(edges)

        def find_path(op):
            vertx = xyhull.points[:, 0]
            verty = xyhull.points[:, 1]

            inits = [v for v in xyhull.vertices if vertx[v] == min(vertx)]

            start = op(inits, key=lambda v: verty[v])
            path = [start]
            neighbors = list(xygraph.neighbors(start))
            while neighbors:
                next_pt = op(neighbors, key=lambda v: verty[v])
                path.append(next_pt)
                neighbors = list(xygraph.neighbors(next_pt))

            return path

        min_path = find_path(min)
        max_path = find_path(max)


        self.ymin = LinearSpline(*torch.tensor(xyverts[min_path].T))
        self.ymax = LinearSpline(*torch.tensor(xyverts[max_path].T))

        self.N = torch.tensor([cut.as_float_cut_plane().n for cut in self.asu.cuts]).float()
        self.c = torch.tensor([cut.as_float_cut_plane().c for cut in self.asu.cuts]).reshape(-1, 1)

        self.ops = torch.stack([torch.tensor(op.affine_matrix) for op in self.wp.ops], dim=0).float()

    def _get_z_bounds(self, x, y):
        if x.shape != y.shape:
            raise ValueError()
        # Nr + c >= 0

        # ((N @ np.array([[0, 0, 0.1]]).T).flatten() + c) >= 0
        c_xy = torch.stack([x, y], dim=-1) @ self.N[:, :2].T + self.c.T
        N_xy = self.N[:, 2]

        N_xy, c_xy = N_xy[..., abs(N_xy.reshape(-1)) > 1e-6], c_xy[..., abs(N_xy.reshape(-1)) > 1e-6]
        # N_xy * z + c_xy >= 0
        # if N_xy > 0, z >= -c_xy / N_xy
        # else, z <= -c_xy / N_xy

        z_vals = -c_xy / N_xy
        zmin = z_vals[..., N_xy.reshape(-1) > 0].max(dim=-1)[0]
        zmax = z_vals[..., N_xy.reshape(-1) < 0].min(dim=-1)[0]
        return (zmin, zmax)

    def to_all_positions(self, asu_xyz: torch.Tensor) -> torch.Tensor:
        """Expands to all of the positions.

        asu_xyz: [..., 3]

        Output: [..., multiplicity, 3]
        """
        if asu_xyz.ndim < 2:
            asu_xyz = asu_xyz.unsqueeze(0)
        out = torch.matmul(
            # shape ..., 1, 1, 4
            torch.cat([asu_xyz, torch.ones_like(asu_xyz[..., [0]])], dim=-1).unsqueeze(-2).unsqueeze(-2),
            # shape mult, 4, 4, but transposed
            self.ops.swapaxes(-1, -2)
        )  # to ..., mult, 1, 4
        return out[..., 0, :3]

    @staticmethod
    def _inv_transform(transform: AffineTransform) -> AffineTransform:
        """Inverts the transform, handling the case of near-zero scale by returning a scale of one."""
        inv_scale = torch.where(torch.abs(transform.scale) <= 1e-6, 1, 1 / transform.scale)
        return AffineTransform(-transform.loc * inv_scale, inv_scale)


    def inverse(self, xyz: torch.Tensor) -> torch.Tensor:
        """Inverts the transformation.
        xyz: [..., 3]

        Output: [..., dof]
        """
        x = xyz[..., [0]]
        x_u = self._inv_transform(self.x_trans)(x)
        ylo, yhi = self.ymin(x).float(), self.ymax(x).float()
        y_trans = AffineTransform(ylo, yhi - ylo)
        y = xyz[..., [1]]
        y_u = self._inv_transform(y_trans)(y)

        zlo, zhi = self._get_z_bounds(x, y)
        z_trans = AffineTransform(zlo, zhi - zlo)
        z = xyz[..., [2]]
        z_u = self._inv_transform(z_trans)(z)

        free_axes = [i for i in range(3) if i not in self.wp.get_frozen_axis()]
        return torch.cat([x_u, y_u, z_u], dim=-1)[..., free_axes]


    def to_asu(self, xyz: torch.Tensor) -> torch.Tensor:
        """Converts to the transformed space.

        xyz: [..., num_dof]

        Output: [..., 3]
        """
        if len(xyz.shape) == 0 or len(xyz.shape) == 1 and xyz.shape[-1] >= 3:
            xyz = xyz.reshape(1, -1)

        for op in self.wp.ops:
            gen_rot = torch.tensor(op.rotation_matrix).float()
            gen_tau = torch.tensor(op.translation_vector).float()
            pos_i = 0
            if 0 not in self.wp.get_frozen_axis():
                # x is free parameter, we know it
                x = self.x_trans(xyz[..., [pos_i]])
                pos_i += 1
            else:
                # x is constant
                # note that WPs never have, e.g., -y, y, z
                # it would be x, -x, z instead

                # sometimes the first position isn't the one in the ASU
                # so search for one of those
                x = gen_tau[0] % 1
                if (0 <= torch.round(self.x_trans.inv(x), decimals=2) <= 1):
                    x = torch.ones_like(xyz[..., :1]) * x
                    if xyz.numel() == 0:
                        x = torch.tensor([[gen_tau[0]]])
                else:
                    continue


            # x_u = (x - self.xmin) / (self.xmax - self.xmin)
            ylo, yhi = self.ymin(x).float(), self.ymax(x).float()
            y_trans = AffineTransform(ylo, yhi - ylo)

            if 1 not in self.wp.get_frozen_axis():
                # y is free parameter, we know it
                y = y_trans(xyz[..., [pos_i]])
                pos_i += 1
            else:
                # y is function of x
                # note that WPs never have, e.g., x, x-z, z
                # it would be x, y, x-y instead
                y = (gen_rot[1, 0].item() * x + gen_tau[1].item()) % 1


            zlo, zhi = self._get_z_bounds(x, y)
            z_trans = AffineTransform(zlo, zhi - zlo)

            if 2 not in self.wp.get_frozen_axis():
                # z is free parameter
                z = z_trans(xyz[..., [pos_i]])
                pos_i += 1
            else:
                # z is function of x and/or y
                z = (torch.cat([x, y], dim=-1) @ gen_rot[2, :2] + gen_tau[2]) % 1
                z = z.unsqueeze(-1)

            if z.shape != x.shape:
                raise ValueError()
            return torch.cat([x, y, z], dim=-1)


if __name__ == '__main__':
    test_asu = True
    test_all_pos = False
    test_inv = True
    test_screw = True
    test_Immm = True

    from rich.progress import track

    if test_Immm:
        wps = Group('Immm').Wyckoff_positions
        for wp in wps:
            if wp.get_dof() == 0:
                continue
            ws = WyckoffSet('Immm', wp.letter)
            zz = torch.cartesian_prod(*[
                torch.linspace(0.01, 0.99, 10) for _ in range(wp.get_dof())
            ]).reshape(10, -1)

            posns = ws.to_asu(zz)
            for i in range(len(posns)):
                if not torch.allclose(ws.to_asu(ws.inverse(posns[[i]])), posns[i]):
                    raise ValueError('Whoops')
            all_posns = ws.to_all_positions(posns)
        print('Immm success!')

    if test_screw:
        ws = WyckoffSet(143, 'c')
        zz = torch.cartesian_prod(
            torch.linspace(0.01, 0.99, 10),
        ).reshape(10, 1)

        posns = ws.to_asu(zz)
        for i in range(len(posns)):
            if not torch.allclose(ws.to_asu(ws.inverse(posns[[i]])), posns[i]):
                raise ValueError('Whoops')
        all_posns = ws.to_all_positions(posns)
        print('Screw success!')

    if test_inv:
        for group in track(range(1, 231), 'Testing inverses...'):
            wps = Group(group).Wyckoff_positions
            for wp in wps[:2] + wps[-2:]:
                ws = WyckoffSet(group, wp.index)
                if wp.get_dof() == 0:
                    continue

                xyz = torch.cartesian_prod(*[
                    torch.linspace(0.01, 0.99, 5)
                    for _ in range(wp.get_dof())
                ]).reshape(5 ** wp.get_dof(), wp.get_dof())

                posns = ws.to_asu(xyz)
                for i in range(len(posns)):
                    if not torch.max(torch.abs(ws.to_asu(ws.inverse(posns[[i]])) - posns[i])) < 1e-3:
                        print(wp)
                        print(group)
                        raise ValueError('Whoops')
                all_posns = ws.to_all_positions(posns)
        print('Inverses successful!')

    if test_all_pos:
        for group in track(range(1, 231), 'Testing all positions...'):
            wps = Group(group).Wyckoff_positions
            for wp in wps[:2] + wps[-2:]:
                ws = WyckoffSet(group, wp.index)
                xyz = torch.cartesian_prod(
                    torch.linspace(0.01, 0.99, 5),
                    torch.linspace(0.01, 0.99, 5),
                    torch.linspace(0.01, 0.99, 5)
                ).reshape(125, 3)[:, :wp.get_dof()]

                if xyz.numel() == 0:
                    continue
                posns = ws.to_asu(xyz)
                all_posns = ws.to_all_positions(posns)
                for all_posn, posn in zip(all_posns, posns):
                    assert np.allclose(all_posn.numpy(), wp.apply_ops(posn.numpy()), rtol=1e-4, atol=1e-4)

        print('Generating all positions successful!')

    if test_asu:
        for group in track(range(1, 231), 'Testing ASU...'):
            wp_index = 0
            if len(Group(group).Wyckoff_positions) > wp_index:
                wp = WyckoffSet(group, wp_index)
            else:
                continue

            xyz = torch.cartesian_prod(
                torch.linspace(0.01, 0.99, 5),
                torch.linspace(0.01, 0.99, 5),
                torch.linspace(0.01, 0.99, 5)
            ).reshape(125, 3)[:, :wp.wp.get_dof()]

            posns = wp.to_asu(xyz)
            for pos in posns:
                for cut in wp.asu.cuts:
                    if not cut.is_inside(pos.numpy()):
                        print(group, wp.wp)
                        print(wp.asu.shape_vertices())
                        print(pos)
                        raise ValueError('Whoops!')

        print('Group general position test successful!')