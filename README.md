# BAYesian Symmetry-Informed Crystal structure prediction

**This repository is still in a beta stage. Expect breaking changes and potential bugs.**

This repository allows you to generate candidate structures for a given chemical composition. It does this in several steps:

- Pick a space group.
- Generate an appropriate lattice for that space group.
- Generate a valid Wyckoff position assignment for the composition.
- Find specific coordinates for each free Wyckoff axis (every letter `x, y, z` in the Wyckoff operations) such that the generated structure has feasible inter-atomic distances.

Baysic tempers unconstrained random generation with reasonable prior distributions (for example, generating lattices with feasible volumes) and employs fairly sophisticated algorithms to generate structures efficiently. In many cases, one of the generated structures can be relaxed into the correct structure.

## Installation

In the same directory as this file:

```bash
conda create -n baysic python=3.10 pytorch --channel pytorch
conda activate baysic
conda install -c conda-forge cctbx-base
pip install --upgrade pandas seaborn chgnet mp_api rich pyrallis scipy pyxtal monty tqdm pyro-ppl toml rho-plus lightning umap-learn xenonpy numpy==1.25
pip install --editable .
```

The `--editable` in the last command means changes in the source code will work properly. Annoyingly, however, it makes imports incredibly slow, which can be aggravating if you're running small scripts and don't want several seconds of latency before the program is responsive. In these cases, consider doing `pip install .`, which freezes the package, running whatever short scripts you're interested in, and then later running `pip install --editable .` if you continue to make edits to the source code.

To test things are working, run `pyt`

## Execution
Run `python baysic/group_search.py` with suitable command-line options.

Try `python baysic/group_search.py --help` for hyperparameters, or check the files in `configs/`.

## Documentation
Good documentation will come later, when the program is not undergoing such frequent updates. The help descriptions in `config.py`, also reachable by `python baysic/group_search.py --help`, give some guidance for using the program. For now, while the program is in beta, the defaults represent an educated guess at good options for a specific application.

For anything else, you're best off contacting me on GitHub or at [nmiklaucic@sc.edu](mailto:nmiklaucic@sc.edu).

The many notebooks in this directory (VSCode makes it convenient to put them all in the outer directory, unfortunately) are mostly for development purposes and may be misleading. `priors.ipynb` is an exception: it shows the data exploration that has motivated my choices for several important hyperparameters, and is a worthwhile read if you're interested in why Baysic generates more realistic structures more quickly than prior random generation libraries.

It also gives a sense of promising future developments: replacing random samples from static distributions with samples from generated, dynamic distributions has some potential.
