"""
Given a specific symmetrized problem, defines the minimal covering feature space.
"""

from scipy.spatial import ConvexHull
import networkx as nx
import numpy as np
from cctbx.sgtbx.direct_space_asu.reference_table import get_asu
import scipy.interpolate as interp
from pyxtal import Group, Wyckoff_position
from pyxtal.symmetry import symbols as group_symbols
from numpy.typing import ArrayLike


class FeatureSpace:
    """The feature space for a group."""

    def __init__(self, group: int | str):
        """Initializes feature space for a specific space group number or symbol."""
        if isinstance(group, str):
            group_sym = group.strip().replace(' ', '')
            if group_sym not in group_symbols['space_group']:
                raise ValueError(f'Group {group} cannot be identified')
            else:
                self.sg = Group(group_symbols['space_group'].index(group_sym) + 1)
        else:
            self.sg = Group(group)

        self.asu = get_asu(self.sg.number)

        self._initialize_space()

    def get_wp(self, wp: int | str | Wyckoff_position) -> Wyckoff_position:
        if isinstance(wp, Wyckoff_position):
            return wp
        elif isinstance(wp, str):
            return Wyckoff_position.from_group_and_letter(self.sg.number, wp)
        elif isinstance(wp, int):
            return Wyckoff_position.from_group_and_index(self.sg.number, wp)

    def to_general_positions(self, xyz: ArrayLike) -> np.ndarray:
        """Converts points in [0, 1]^3 to Wyckoff coordinates."""
        return np.vstack(self.xyz_to_wp(*np.asanyarray(xyz).T)).T        
    
    def to_all_positions(self, xyz: ArrayLike, wp: int | str | None = None):
        """Gets all positions, not just the general one.
        
        If wp is an index or letter, uses that WP. If None, searches
        to find which symmetry fits the given position best."""
        gen_pos = self.to_general_positions(xyz)
        print(gen_pos)
        
        if wp is None:
            wyckoffs = self.sg.Wyckoff_positions[::-1]
        else:
            wyckoffs = [self.get_wp(wp)]

        all_pos = []
        for gp in gen_pos:
            # iterate in order of most to least symmetric            
            # last GP is (x, y, z), which should always work
            # we can't just compare the general positions, because
            # we don't know which WP is inside the ASU for any particular WP            
            for wp in wyckoffs:                
                if any(np.allclose(gp, other_p) for other_p in wp.apply_ops(gp)):                    
                    all_pos.append(wp.apply_ops(gp))
                    break

        return all_pos

    def _initialize_space(self):
        """Identifies the feature space."""

        verts = np.array(self.asu.shape_vertices()).astype(float)

        xyverts = verts[:, [0, 1]]
        xyhull = ConvexHull(xyverts)

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

        ymin = interp.interp1d(*xyverts[min_path].T)
        ymax = interp.interp1d(*xyverts[max_path].T)

        N = np.array([cut.as_float_cut_plane().n for cut in self.asu.cuts])
        c = np.array([cut.as_float_cut_plane().c for cut in self.asu.cuts])
        def get_z_bounds(x, y):
            # Nr + c >= 0

            # ((N @ np.array([[0, 0, 0.1]]).T).flatten() + c) >= 0    
            c_xy = N[:, :2] @ np.vstack([x, y]) + c.reshape(-1, 1)
            N_xy = N[:, [2]]

            N_xy, c_xy = N_xy[abs(N_xy.reshape(-1)) > 1e-6], c_xy[abs(N_xy.reshape(-1)) > 1e-6]
            # N_xy * z + c_xy >= 0
            # if N_xy > 0, z >= -c_xy / N_xy
            # else, z <= -c_xy / N_xy

            z_vals = -c_xy / N_xy
            zmin = z_vals[N_xy.reshape(-1) > 0].max(axis=0)
            zmax = z_vals[N_xy.reshape(-1) < 0].min(axis=0)
            return (zmin, zmax)

        zmins, zmaxs = get_z_bounds(*xyverts.T)
        zmin = interp.LinearNDInterpolator(xyverts, zmins)
        zmax = interp.LinearNDInterpolator(xyverts, zmaxs)

        def get_unbounded_xyz(x_u, y_u, z_u):
            xmin, xmax = xyverts[:, 0].min(), xyverts[:, 0].max()
            x = xmin + x_u * (xmax - xmin)
            ylo = ymin(x)
            yhi = ymax(x)
            y = ylo + y_u * (yhi - ylo)
            zlo = zmin(np.vstack([x, y]).T)
            zhi = zmax(np.vstack([x, y]).T)
            z = zlo + z_u * (zhi - zlo)
            return np.array([np.asanyarray(x), np.asanyarray(y), np.asanyarray(z)])

        self.xmin, self.xmax = xyverts[:, 0].min(), xyverts[:, 0].max()
        self.xyz_to_wp = get_unbounded_xyz
        self.ymin, self.ymax = ymin, ymax        
        self.zmin, self.zmax = zmin, zmax  

    def from_free_transformed_xyz(self, pos: ArrayLike, wp: int | str):
        """Given just the free xyzs in [0, 1], returns the equivalent position
        in the full [x, y, z] space."""    
        wp = self.get_wp(wp)

        pos = np.asanyarray(pos)
        if len(pos) == 0:
            return np.array(wp.gen_pos().operate([0, 0, 0]))
        if len(pos.shape) < 2:
            pos = pos.reshape(1, -1)
        

        for op in wp.ops:
            gen_rot = op.rotation_matrix
            gen_tau = op.translation_vector
            pos_i = 0
            if 0 not in wp.get_frozen_axis():
                # x is free parameter, we know it
                x = self.xmin + pos[:, pos_i] * (self.xmax - self.xmin)            
                pos_i += 1
            else:
                # x is constant
                # note that WPs never have, e.g., -y, y, z
                # it would be x, -x, z instead
                
                # sometimes the first position isn't the one in the ASU
                # so search for one of those
                x = gen_tau[0]
                if x < self.xmin or x > self.xmax:
                    continue

            # x_u = (x - self.xmin) / (self.xmax - self.xmin)
            ylo, yhi = self.ymin(x), self.ymax(x)

            if 1 not in wp.get_frozen_axis():
                # y is free parameter, we know it
                y = ylo + pos[:, pos_i] * (yhi - ylo)            
                pos_i += 1
            else:
                # y is function of x
                # note that WPs never have, e.g., x, x-z, z
                # it would be x, y, x-y instead
                y = (gen_rot[1, 0] * x + gen_tau[1]) % 1

            
            xy = np.vstack([x, y]).T

            try:
                zlo, zhi = self.zmin(xy), self.zmax(xy)
            except ValueError:
                # out of bounds, try a different coordinate
                continue

            if 2 not in wp.get_frozen_axis():
                # z is free parameter
                z = zlo + pos[:, pos_i] * (zhi - zlo)
                pos_i += 1
            else:
                # z is function of x and/or y
                z = (xy @ gen_rot[[2], :2].T + gen_tau[2]) % 1
            
            return np.vstack([x, y, z]).T    
            

if __name__ == '__main__':
    fs = FeatureSpace(71)
    print(fs.from_free_transformed_xyz(np.array([]), 'a'))