import pickle
import pandas as pd

from mp_api.client import MPRester

df = pd.read_csv('merged_test_data3.csv')

structs = []
with MPRester() as mpr:
    for mp_id in df['material_id']:
        structs.append(mpr.get_structure_by_material_id(mp_id))

df['struct'] = structs
df.to_pickle('merged_test_data3.pkl')