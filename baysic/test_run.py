import logging
import ray
from raytune import RaytuneOptimizer
import pandas as pd
from pyxtal import Group
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from structure_evaluation import e_form, relaxed_energy
from tqdm import tqdm

ray.init(log_to_driver=False, logging_level=logging.ERROR)

df = pd.read_pickle('merged_test_data3.pkl')

subs = df.iloc[:2]

results = []

for i, row in tqdm(subs.iterrows(), total=subs.shape[0]):
    struct = row['struct']
    sg = Group(row['spacegroup'])        

    sga = SpacegroupAnalyzer(struct)
    symm = sga.get_conventional_standard_structure()
    scale = int(symm.num_sites / struct.num_sites)

    model = RaytuneOptimizer(sg, struct.composition, scale)
    out_structure, out_conf = model.fit(e_form)

    out_relaxed, out_e_form = relaxed_energy(out_structure, long=True)
    real_relaxed, real_e_form = relaxed_energy(symm.get_conventional_structure())
    results.append([out_e_form - real_e_form, out_e_form, real_e_form, out_relaxed.matches(symm)])

results = pd.DataFrame(results, columns=['delta_e_form', 'pred_e', 'real_e', 'matches'])
print(results['delta_e_form'].quantile([0.1, 0.5, 0.9]).round(3))
results.to_pickle('tests.pkl')