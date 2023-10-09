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


MIN_DIST_RATIO = 0.8
VACUUM_SIZE = 7

def is_structure_valid(struct: Structure) -> bool:
    """Tests structure validity."""    
    struct.make_supercell([2, 2, 2], to_unit_cell=False)
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

chgnet = CHGNet.load()
relaxer = StructOptimizer()

def point_energy(struct: Structure) -> float:
    prediction = chgnet.predict_structure(struct, task='e')
    if not is_structure_valid(struct):        
        return 100 + prediction['e'].item()
    else: 
        return prediction['e'].item()

def relaxed_energy(struct: Structure, long: bool = False) -> (Structure, float):
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

def e_form(struct: Structure) -> float:    
    if not is_structure_valid(struct):
        return 100
    else:
        return relaxed_energy(struct)[1]
    

if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    import torch
    df = pd.read_pickle('merged_test_data3.pkl')
    energies = []

    with torch.device('cpu'):
        for i in tqdm(df.index[:20]):
            energies.append(point_energy(df.loc[i, 'struct']))