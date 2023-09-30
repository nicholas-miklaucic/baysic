# BAYesian Symmetry-Informed Crystal structure prediction

**Warning: this repository is still in a *very* early development stage. Expect breaking changes and bugs!**

The Bayesian part is a bit of a misnomer right now. What *is* working is the ability to generate crystal structures matching a particular lattice system (cubic for the time being).

## Installation

In the same directory as this file:

```bash
conda create -n baysic python=3.10
conda activate baysic
pip install -r requirements.txt
pip install --editable . 
```

## Execution
Run `python baysic/pyro_run.py` and go meditate—it might take a while.

Check that file for hyperparameters.

## Results
When it's done, `logs/mm-dd/1/` will contain logs of each run.
Each separate molecule will have its own `json` with the data for each generated structure. Additionally, `total.json` will contain the molecule-level data, using only the best generated structure.
See `diagnostics` and `task_diagnostics` for some example plots using that data.