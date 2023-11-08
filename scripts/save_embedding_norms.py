"""Saves a mean and standard deviation for the compositional embeddings. This allows 
distributed data loading and processing while keeping standardized inputs."""


if __name__ == '__main__':
    import torch
    from torch import nn, optim, utils
    from pyro import distributions as dist
    from torch.nn import functional as F
    import lightning.pytorch as pl
    from xenonpy.datatools import preset
    from xenonpy.descriptor import Compositions
    import pandas as pd
    from pymatgen.core import Composition

    cal = Compositions()

    df = pd.read_feather('logs/12345.feather')

    comps = [Composition(x) for x in set(df['comp'])]

    vals = torch.tensor(cal.transform(comps).values)

    # inverse transform: 1/sigma * x - mu/sigma
    mu = vals.mean(dim=0)
    sigma = vals.std(dim=0)    

    torch.save({
        'loc': -mu/sigma,
        'scale': 1/sigma,
    }, 'precomputed/xenonpy_scale.py')

    print('Done!')