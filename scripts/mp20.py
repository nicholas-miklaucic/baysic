from baysic.utils import load_mp20
from baysic.group_search import main_
from baysic.config import MainConfig, TargetStructureConfig, SearchConfig
import gc
import logging

df = load_mp20('train')

for row in df.index[::100]:
    mat_id = df.loc[row, 'material_id']
    std_to_primitive = df.loc[row, 'std_mapping_to_primitive']
    if len(set(std_to_primitive)) != len(std_to_primitive):
        continue
    try:
        main_(MainConfig(target=TargetStructureConfig(mat_id), search=SearchConfig(
            allowed_attempts_per_gen = 5.0,
            max_gens_at_once = 15,
            num_generations = 150)))
    except Exception as e:
        logging.error(e)
        continue
    finally:
        gc.collect()