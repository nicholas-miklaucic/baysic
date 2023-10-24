"""Code to generate relaxed structures from values."""
import functools
from math import e
from pymatgen.core import Structure
from scipy.spatial.distance import pdist, squareform
import itertools
import numpy as np

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

# from matgl import matgl
# from matgl.matgl.ext.ase import M3GNetCalculator, MolecularDynamics, Relaxer
from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet
import torch

from baysic.utils import upper_tri


MIN_DIST_RATIO = 0.6
VACUUM_SIZE = 7

def is_structure_valid(struct: Structure) -> bool:
    """Tests structure validity."""
    s = struct.copy()
    struct.make_supercell([1, 1, 1])
    radii = np.array([CovalentRadius.radius[site.specie.symbol] for site in struct.sites])
    # distance threshold
    dists = upper_tri(squareform(pdist(struct.cart_coords)))
    rads = upper_tri(np.add.outer(radii, radii))
    if np.any(dists / rads <= MIN_DIST_RATIO):
        return False


    def get_foot(p, a, b):
        p = np.array(p)
        a = np.array(a)
        b = np.array(b)
        ap = p - a
        ab = b - a
        result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    def get_distance(a, b):
        return np.sqrt(np.sum(np.square(b - a)))


    line_a_points = [[0, 0, 0], ]
    line_b_points = [[0, 0, 1], [0, 1, 0], [1, 0, 0],
                        [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, -1], [1, 0, -1], [1, -1, 0],
                        [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1]]
    for a in line_a_points:
        for b in line_b_points:
            foot_points = []
            for p in struct.frac_coords:
                f_p = get_foot(p, a, b)
                foot_points.append(f_p)
            foot_points = sorted(foot_points, key=lambda x: [x[0], x[1], x[2]])

            # 转为笛卡尔坐标
            foot_points = np.asarray(np.mat(foot_points) * np.mat(struct.lattice.matrix))
            for fp_i in range(0, len(foot_points) - 1):
                fp_distance = get_distance(foot_points[fp_i + 1], foot_points[fp_i])
                if fp_distance > VACUUM_SIZE:
                    return False

    return True

# eform = matgl.load_model('matgl/pretrained_models/M3GNet-MP-2018.6.1-Eform/model.json')
# pot = matgl.load_model("matgl/pretrained_models/M3GNet-MP-2021.2.8-PES/model.json")
# relaxer = Relaxer(potential=pot)

chgnet = None
relaxer = None

def point_energy(struct: Structure, device: str = "cpu") -> float:
    global chgnet
    if chgnet is None:
        chgnet = CHGNet.load().to(device)

    prediction = chgnet.predict_structure(struct, task='e')
    if not is_structure_valid(struct):
        return 100 + prediction['e'].item()
    else:
        return prediction['e'].item()

def point_energies(structs: list[Structure], device: str = "cpu") -> list[float]:
    if len(structs) == 1:
        return [point_energy(structs[0], device=device)]

    global chgnet
    if chgnet is None:
        chgnet = CHGNet.load().to(device)

    predictions = chgnet.predict_structure(structs, task='e')
    preds = []
    for struct, pred in zip(structs, predictions):
        if not is_structure_valid(struct):
            preds.append(100 + pred['e'].item())
        else:
            preds.append(pred['e'].item())

    return preds

def relaxed_energy(struct: Structure, long: bool = False) -> (Structure, float):
    global relaxer
    if relaxer is None:
        relaxer = StructOptimizer()
    if long:
        params = dict(fmax=0.01, steps=150)
    else:
        params = dict(fmax=0.02, steps=5)

    relax_results = relaxer.relax(struct, **params)
    # extract results
    final_structure = relax_results["final_structure"]
    final_energy = relax_results["trajectory"].energies[-1]
    # print out the final relaxed structure and energy

    return (final_structure, final_energy / final_structure.num_sites)

def e_forms(structs: Structure, *args, **kwargs) -> list[float]:
    if isinstance(structs, Structure):
        structs = [structs]

    invalid_structs = []
    valid_structs = []
    valid_is = []
    for i, struct in enumerate(structs):
        if is_structure_valid(struct):
            valid_structs.append(struct)
            valid_is.append(i)
        else:
            invalid_structs.append(struct)

    energies = np.ones(len(structs)) * 100
    energies[valid_is] = point_energies(valid_structs, *args, **kwargs)
    return energies


if __name__ == '__main__':
    import pandas as pd
    from rich.progress import track, Progress, SpinnerColumn
    import torch
    df = pd.read_pickle('merged_test_data3.pkl')
    energies = []

    with torch.device('cpu'):
        subset = df.iloc[:40]
        for i, row in track(subset.iterrows()):
            energies.append(point_energy(row['struct']))

        with Progress(SpinnerColumn(), speed_estimate_period=0) as progress:
            energies2 = point_energies(subset['struct'])

            assert np.allclose(energies, energies2)
