"""Finds the span of a Wyckoff position."""

import itertools
from math import e
from turtle import circle
import numpy as np
from pymatgen.core import SymmOp, Lattice
from pyxtal import Wyckoff_position
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from baysic.utils import json_to_df

def is_nonzero(arr, **kwargs):
    return np.linalg.norm(arr, **kwargs) >= 1e-6


class WyckoffSpan:
    """ABC representing a Wyckoff span in 0, 1, or 2 dimensions."""
    dim: int = -1
    def __init__(self, op: SymmOp):
        self.free_vars = np.where(is_nonzero(op.rotation_matrix, axis=0))[0]
        if len(self.free_vars) != self.dim:
            raise ValueError(f'{op} is not a line, len({self.free_vars}) != 1')

    def dist_ratio(self, lattice: Lattice, radii: np.array) -> float:
        """Computes the maximum dist_ratio needed to make the atoms fit. Returns
        np.inf if all assignments work, and 0 if no assignments work."""
        raise NotImplementedError()

    def __le__(self, other):
        if self.dim > other.dim:
            return False
        elif self.dim == other.dim:
            return self == other
        else:
            test_params = np.random.randn(10, self.dim)
            test_pts = np.array([
                self(*params) for params in test_params
            ])

            return all([other.contains(test_pt) for test_pt in test_pts])

class Point(WyckoffSpan):
    """A 0D Wyckoff span."""
    dim = 0
    def __init__(self, op: SymmOp):
        super().__init__(op)
        self.point = op.translation_vector % 1

    def dist_ratio(self, lattice: Lattice, radii: np.array) -> float:
        return np.inf if len(radii) <= 1 else 0

    def __eq__(self, other):
        return np.allclose(self.point, other.point)

    def __repr__(self):
        return f'{self.point}'

    def contains(self, pt):
        return np.allclose(self.point, pt)

    def __call__(self):
        return self.point

class Line(WyckoffSpan):
    """A 1D Wyckoff span."""
    dim = 1
    def __init__(self, op: SymmOp):
        super().__init__(op)
        rot = op.rotation_matrix
        tau = op.translation_vector

        p = rot[:, self.free_vars].flatten()

        p /= np.linalg.norm(p)
        # p * t + τ is line
        # want to find intersection with unit cube

        # we're working mod 1: a WP like x, -x, 0.25
        # doesn't ever intersect the unit cube except at 0
        # instead, we need to "wiggle" τ to get out of edges
        # and then find how to move it into the unit cube
        # first, wiggle the point to get out of corners
        # this is equivalent to x+ε, -(x+ε), 0.25

        self.tau = (p * 0.01 + tau) % 1
        self.t_min = -np.inf
        self.t_max = np.inf
        for ax in range(3):
            if abs(p[ax]) <= 1e-6:
                # no movement on this axis
                # can safely ignore: doesn't help us bound
                continue
            else:
                # solve for t such that p[ax] * t + τ[ax] = 0 and 1
                bound = [(b - self.tau[ax]) / p[ax] for b in (0, 1)]
                min_b, max_b = sorted(bound)
                # print(ax, min_b, max_b)
                self.t_min = max(min_b, self.t_min)
                self.t_max = min(max_b, self.t_max)

        # the center is independent of which intercept we started
        # with and antiparallel lines: two lines are equivalent
        # iff centers match and angle between dirs is 0 or 180
        self.bottom = p * self.t_min + self.tau
        self.top = p * self.t_max + self.tau
        self.middle = 0.5 * (self.bottom + self.top)
        self.p = p

    def __eq__(self, other):
        # this will break with __hash__: we're going to ignore that
        return np.allclose(self.middle, other.middle) and np.isclose(abs(np.dot(self.p, other.p)), 1)

    def __repr__(self):
        px, py, pz = self.p.round(3).flatten()
        tx, ty, tz = self.middle.round(3).flatten()
        return f'{px}x + {tx}, {py}y + {ty}, {pz}z + {tz}'

    def __call__(self, t):
        return self.p * t + self.tau

    @property
    def vec(self) -> np.array:
        """A vector representing the segment in fractional space."""
        return self.p * (self.t_max - self.t_min)

    def dist_ratio(self, lattice: Lattice, radii: np.array) -> float:
        line_len = np.linalg.norm(self.vec @ lattice.matrix)
        # diameters have to be less than the total length
        return line_len / np.sum(2 * radii)

    def contains(self, pt):
        if np.allclose(pt, self.tau):
            return True
        return abs(np.dot((pt - self.tau), self.p)) / np.linalg.norm(pt - self.tau) == 1


class Plane(WyckoffSpan):
    dim = 2
    def __init__(self, op: SymmOp):
        super().__init__(op)

        M = op.rotation_matrix[:, self.free_vars]
        n = np.cross(M.T[0], M.T[1])
        tau = op.translation_vector

        # why we need this: e.g., x, -x, z has τ = (0, 0, 0),
        # which intersects eight cubes, and only some work
        # we can move τ to ensure that there are four intersection
        # points in the cube between 0, 0, 0 and 1, 1, 1
        # 0.01 ensures that we're not moving anything onto an edge:
        # this could be done so that τ is easier to interpret
        self.tau = ((M @ np.ones(2) * 0.01) + tau) % 1

        self.corners = []
        for free_ax in (0, 1, 2):
            # three sets of four parallel lines
            l = np.zeros(3)
            l[free_ax] = 1
            nonfree = [0, 1, 2]
            nonfree.remove(free_ax)
            for other_axs in itertools.product(range(2), repeat=2):
                l0 = np.zeros(3)
                l0[free_ax] = 0.5
                l0[nonfree] = other_axs
                # plane equation is (t - τ) . n = 0
                # line equation is l * d + l0
                # intersection is d = ((τ - l0) . n) / (l . n)
                # if l . n is 0, then plane is parallel and no intersection
                l_dot_n = np.dot(l, n)
                if is_nonzero(l_dot_n):
                    d = np.dot(self.tau - l0, n) / l_dot_n
                    self.corners.append(np.dot(l, d) + l0)

        # either 4, 8, or 12 intersections depending on how many of the sides the plane
        # is parallel with: take the four inside the cube, which should always be exactly
        # four
        self.corners = np.array(self.corners)
        inside_cube = np.all((self.corners >= 0) & (self.corners <= 1), axis=1)
        self.corners = self.corners[inside_cube]
        # remove duplicates: intersections at corners appear twice
        unique = list(range(len(self.corners)))
        for i, corner in enumerate(self.corners):
            if any(np.allclose(corner, prev) for prev in self.corners[:i]):
                unique.remove(i)

        assert len(unique) == 4
        self.corners = self.corners[unique]

        # centroid can be used to easily compare planes
        self.centroid = self.corners.mean(axis=0)
        self.normal = n
        self.M = M

    def __eq__(self, other):
        # this will break with __hash__: we're going to ignore that
        return np.allclose(self.centroid, other.centroid) and np.isclose(abs(np.dot(self.normal, other.normal)), 1)

    def __call__(self, u, v):
        return self.M @ np.array([u, v]) + self.centroid

    def contains(self, pt):
        return np.allclose(np.dot(self.normal, (pt - self.centroid)), 0)

    def __repr__(self):
        return "(\u27e8x, y, z\u27e9 - \u27e8{}, {}, {}\u27e9) \u22c5 \u27e8{}, {}, {}\u27e9 = 0".format(
            *self.centroid.round(3).flatten(),
            *self.normal.round(3).flatten())

    def capacity(self, lattice: Lattice) -> float:

        verts = self.corners @ lattice.matrix
        centroid = verts.mean(axis=0)

        # we don't know what order we're given the coordinates in, unfortunately
        #     a ------ b
        #   /   .    /
        # c ------ d
        # we can take the distance from the centroid to each point:
        # the min and max of that should give us two opposite points
        # with a fair bit of error tolerance, and then we can compute
        # the area of that triangle x4
        half_diags = np.linalg.norm(verts - centroid, axis=1)
        a_i, b_i = np.argsort(half_diags)[[0, -1]]
        a, b = verts[[a_i, b_i]]
        a_o, b_o = half_diags[[a_i, b_i]]
        a_b = np.linalg.norm(a - b)
        # never thought I'd use Heron's formula again...
        s = a_o + b_o + a_b
        return 4 * np.sqrt(np.prod(s - np.array([a_o, b_o, a_b])) * s)


    def dist_ratio(self, lattice: Lattice, radii: np.array) -> float:
        # the atom centers can be anywhere on the span, but the radii have to
        # fit within it: otherwise there would be an overlap between unit cells

        # unequal circle packing in parallelograms is...not even close to a well-understood
        # problem. Intuitively, if we're packing roughly equal circle areas, we're not
        # going to be able to do much better than the maximum density for equal circles
        # which is π/(2 root 3) ≅ 0.9069. However, if you're allowed to fit arbitrarily
        # small circles into gaps, you can get asymptotically perfect packing, and estimating
        # where between 0.9069 and 1 a given set of radii falls is very hard.
        #
        # Instead of the perfect result, we try to find a workable lower bound. It's OK
        # if a few examples get through that aren't technically doable, but we can't mistakenly
        # rule out valid setups.
        #
        # A combination of two bounds will suffice. First, the packing density is at most 1. Second,
        # if replacing every radius with the smallest one is unpackable, than making spheres bigger
        # won't help. We can use the exact bound here, which means we'll give a very good bound
        # in the common case that all radii are equal.
        #
        # http://www.packomania.com/
        # 0.9069 is quite generous: with only a few circles, and with radii that don't
        # fit perfectly into the parallelogram, you get smaller packings. There's no
        # easy bound for this I can find, unfortunately.
        plane_area = self.capacity(lattice)

        MAX_DENSITY_PACKING = np.pi / (2 * np.sqrt(3))
        circle_areas = np.pi * np.square(radii)
        min_needed_area = max(
            np.sum(np.min(circle_areas) * len(circle_areas)) / MAX_DENSITY_PACKING,
            np.sum(circle_areas)
        )

        return np.sqrt(plane_area / min_needed_area)


def min_dist_ratio(sg_num, wps, species, lattice: Lattice) -> float:
    spans = []
    span_elements = []

    for specie, wp in zip(species, wps):
        if wp.get_dof() != 3:
            radius = CovalentRadius.radius[specie.symbol]
            for op in wp.ops:
                n_dof = sum(is_nonzero(op.rotation_matrix, axis=0))
                span = (Point, Line, Plane)[n_dof](op)
                did_add = False
                for span_i, prev_span in enumerate(spans):
                    if span <= prev_span:
                        if span.dim == prev_span.dim:
                            # already tracking this span
                            did_add = True
                        span_elements[span_i].append(specie)

                if not did_add:
                    spans.append(span)
                    span_elements.append([specie])

    dist_ratios = []
    for span, species in zip(spans, span_elements):
        radii = [CovalentRadius.radius[specie.symbol] for specie in species]
        dist_ratios.append(span.dist_ratio(lattice, radii))

    return np.min(dist_ratios) if dist_ratios else np.inf


if __name__ == '__main__':
    test_lines = True
    test_planes = True
    test_materials = True

    import pandas as pd
    from rich.progress import track
    from baysic.utils import load_mp20
    from pyxtal import Wyckoff_position
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

    import pickle
    with open('all_wps.pkl', 'rb') as infile:
        all_wps = pickle.load(infile)

    for wp in track(all_wps, 'Testing...'):
        for op in wp.ops:
            n_dof = len(np.where(is_nonzero(op.rotation_matrix, axis=0))[0])
            if n_dof == 1 and test_lines:
                l = Line(op)
            elif n_dof == 2 and test_planes:
                pl = Plane(op)

    if test_materials:
        materials = load_mp20('train')

        dist_ratios = []
        elements = []
        radiis = []
        span_dims = []
        for row_i in track(range(0, len(materials.index), 113)):
            row = materials.iloc[row_i]
            if row['sg_symbol'] != row['sg'].symbol:
                # tolerance mismatch, continue
                continue

            spans = []
            span_elements = []

            symm = row['conv']
            sg_num = row['sg_number']
            species = symm.species
            i = 0
            processed = set()
            for specie, symbol, equiv in zip(species, row['wyckoffs'], row['equivalent_atoms']):
                if equiv in processed:
                    continue
                else:
                    processed.add(equiv)
                wp = Wyckoff_position.from_group_and_letter(sg_num, symbol, style='spglib')
                if wp.get_dof() != 3:
                    radius = CovalentRadius.radius[specie.symbol]
                    for op in wp.ops:
                        n_dof = sum(is_nonzero(op.rotation_matrix, axis=0))
                        span = (Point, Line, Plane)[n_dof](op)
                        did_add = False
                        for span_i, prev_span in enumerate(spans):
                            if span <= prev_span:
                                if span.dim == prev_span.dim:
                                    did_add = True
                                span_elements[span_i].append(specie)

                        if not did_add:
                            spans.append(span)
                            span_elements.append([specie])

            for span, species in zip(spans, span_elements):
                radii = [CovalentRadius.radius[specie.symbol] for specie in species]
                radiis.append(radii)
                elements.append(species)
                span_dims.append(span.dim)
                dist_ratios.append(span.dist_ratio(symm.lattice, radii))


        dist_ratios = np.array(dist_ratios)
        span_dims = np.array(span_dims)
        has_special_elements = np.array([
            any([e.symbol in ['Pu', 'U', 'Ce', 'Np', 'Pa'] for e in comp])
        for comp in elements])

        for dim in (0, 1, 2):
            print(dim)
            for arr in (
                dist_ratios[(~has_special_elements) & (span_dims == dim)],
                dist_ratios[(has_special_elements) & (span_dims == dim)]):
                if len(arr):
                    print(np.quantile(arr, np.linspace(0, 1, 11)).round(4))

    print('Success!')