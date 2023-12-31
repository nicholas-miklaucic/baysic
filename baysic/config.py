"""Configuration definitions."""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import pyrallis
from pymatgen.core import Composition, Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from enum import Enum
import json
from torch import Value
from pyrallis import field
from typing import Optional

from baysic.errors import BaysicError

PLACEHOLDER_MP_ID = 'mp-XXXXX'

pyrallis.set_config_type('toml')

@dataclass
class TargetStructureConfig:
    """Configuration that defines a "target" structure: a composition used to search, and a known structure
    to compare the search results against."""
    # The Materials Project ID: used to get the composition and check results against the true structure
    mp_id: str
    # The API key used to grab the data. If None, use the environment variable MP_API_KEY.
    api_key: Optional[str] = None

    # Maximum number of atoms to use. Intended to prevent accidental freezes.
    max_num_atoms: int = 20


    def __post_init__(self):
        """Get the structure and spacegroup info from the ID."""
        if self.mp_id == PLACEHOLDER_MP_ID:
            if __name__ == '__main__':
                # allow placeholder name: all we want to do is write it to a file
                pass
            else:
                # invalid name, but we can be helpful
                raise BaysicError(f'Target MP ID {self.mp_id} is invalid. Did you forget to change the placeholder?')
        else:
            # for some reason, this is a huge import
            # so only load if we need it
            from mp_api.client import MPRester
            with MPRester(self.api_key, mute_progress_bars=True) as mpr:
                self.struct = mpr.get_structure_by_material_id(self.mp_id, conventional_unit_cell=True)
                self.sga = SpacegroupAnalyzer(self.struct)
                self.symm = self.sga.get_symmetrized_structure()

            if self.struct.num_sites > self.max_num_atoms:
                raise BaysicError(f'Too many atoms: {self.mp_id}, {self.struct.num_sites}, {self.max_num_atoms}')

    @property
    def sg_symbol(self) -> str:
        """The space group symbol, e.g., Pm-3m."""
        return self.sga.get_space_group_symbol()

    @property
    def sg_number(self) -> int:
        """The space group number, e.g., 225."""
        return self.sga.get_space_group_number()

    @property
    def composition(self) -> Composition:
        """The conventional cell Composition, e.g., Al6V2."""
        return self.struct.composition

    @property
    def pretty_formula(self) -> str:
        """The pretty Unicode formula, e.g., Mg₆Au₂."""
        return self.struct.composition.to_pretty_string()

    @property
    def formula(self) -> str:
        """The simple formula name, e.g., K2C1N2. Suitable for filenames and the like."""
        return self.struct.composition.to_pretty_string()

    @property
    def conv_struct(self) -> Structure:
        """The conventional structure."""
        return self.struct

    @property
    def conv_lattice(self) -> Lattice:
        """The conventional lattice."""
        return self.struct.lattice

    @property
    def crystal_system(self) -> str:
        """The crystal system, e.g., "monoclinic". """
        return self.sga.get_crystal_system()

    @property
    def wyckoff_letters(self) -> str:
        """A string identifying the Wyckoff letters."""
        return '_'.join(self.symm.wyckoff_letters)


class WyckoffSelectionStrategy(Enum):
    """How to weight which Wyckoff assignments are selected. This is basically only configurable for future ablation:
    the correct answer is sample_distinct."""
    # Assign an equal probability for a WP assignment from each space group to be chosen.
    uniform_sg = 'uniform_sg'
    # Assign an equal probability for each WP assignment to be chosen. Some groups have
    # far more potential options than others, so this is not recommended.
    uniform_wp = 'uniform_wp'
    # Weight assignments so the distribution of distinct Wyckoff position counts roughly follows
    # actual data.
    weighted_wp_count = 'weighted_wp_count'
    # Weight assignments with fewer distinct Wyckoff positions. This does not correct for the number
    # of assignments of different multiplicities, which is probably strictly worse than the above,
    # but it's included for backwards compatibility and testing.
    fewer_distinct = 'fewer_distinct'
    # Randomly generate assignments without first enumerating every possible assignment. *Much* faster, consumes
    # *much* less memory, and highly recommended.
    sample_distinct = 'sample_distinct'


@dataclass
class SearchConfig:
    """Configuration for the structure-based search."""
    # The seed used to initialze the RNG. If None, don't set a seed.
    rng_seed: Optional[int] = None

    # The number of potential structures being searched at once. Increasing this increases computational load
    # and RAM usage significantly, but makes finding difficult configurations easier. (Only max_gens_at_once
    # structures are actually returned, so for successful runs increasing this won't increase the total
    # number of generations.)
    n_parallel_structs: int = 500

    # The number of distinct coordinates generated for each free Wyckoff axis during search. Increasing this
    # increases computational load and RAM usage but makes finding difficult configurations more likely.
    n_grid: int = 12

    # How to randomly choose Wyckoff position assignments. The default is strongly recommended: it's
    # much, much faster, and it roughly matches the distribution of Wyckoff positions found in the real
    # world.
    wyckoff_strategy: WyckoffSelectionStrategy = WyckoffSelectionStrategy.sample_distinct

    # If True, overrides other parameters to create a barebones configuration that can be used to quickly iterate
    # on code and test that things work before trying a real run.
    smoke_test: bool = False

    # Controls how many candidate structures are generated for each space group.
    num_generations: int = 50

    # Controls how many candidate structures are taken from a single successful generation. Decreasing this increases
    # the runtime to make the output structures more diverse.
    max_gens_at_once: int = 10

    # This number times (num_generations / max_gens_at_once) is the number of attempted generations before generations
    # are stopped early. Keeping this low avoids wasting time on difficult generations, many of which are unlikely to
    # be useful, but for some structures you may need to increase this to better search for needles in a haystack.
    allowed_attempts_per_gen: float = 10.0

    # Whether to use covalent radii to determine the order to fill Wyckoff positions. This has no impact on successful runs,
    # but the order Wyckoff positions are filled has a big impact on how quickly failed iterations take. (Ideally, we want
    # to fail quickly, so the idea is that bigger atoms are more likely to be impossible to fit in a lattice.) Otherwise,
    # orders by degrees of freedom.
    order_positions_by_radius: bool = True

    # The space groups to search. This should mainly be used for testing purposes: invalid groups are already factored out.
    # Takes precedence over smoke_test, which by default only tries the first two groups of each lattice system.
    groups_to_search: Optional[list[int]] = None

    def __post_init__(self):
        if self.smoke_test:
            self.num_generations = 3
            self.max_gens_at_once = 2

            if self.rng_seed is None:
                self.rng_seed = 0

class FileLoggingMode(Enum):
    """How to log outputs to a file."""
    # Overwrite existing directories.
    overwrite = 'overwrite'
    # Use existing directory if available, but skip over groups already done. Use to restart after a failure.
    append = 'append'
    # Make an entirely new run.
    new = 'new'


@dataclass
class LogConfig:
    """Controls how outputs are logged."""
    # Whether to use W&B to log experimental data. Requires a W&B API key.
    use_wandb: bool = False

    # A W&B API key. Read from the environment if None.
    wandb_api_key: Optional[str] = None

    # The W&B project name to use.
    wandb_project: str = "baysic"

    # Whether to log outputs to a directory.
    use_directory: bool = True

    # The log directory to use.
    log_directory: Path = Path('logs/')

    # Whether to make a new directory for the run (new), completely overwrite an existing run (overwrite), or
    # use an existing directory but only add new files (append).
    log_dir_mode: FileLoggingMode = FileLoggingMode.append

    def __post_init__(self):
        if not (self.use_wandb or self.use_directory):
            raise ValueError("Not logging to either W&B or the filesystem will not save the results!")


class LoggingLevel(Enum):
    """The logging level."""
    debug = logging.DEBUG
    info = logging.INFO
    warning = logging.WARNING
    error = logging.ERROR
    critical = logging.CRITICAL


@dataclass
class CliConfig:
    """Configuration of the command-line output."""
    # The verbosity of the output, controlling how much gets logged.
    verbosity: LoggingLevel = LoggingLevel.info
    # Whether to show progress bars.
    show_progress: bool = True


@dataclass
class DeviceConfig:
    """Configuration of computing power."""
    # Torch device: probably should be either 'cpu' or 'cuda'.
    device: str = 'cpu'

    # Number of threads for Pytorch. I think PyTorch parallelism
    # is less efficient than parallelized space group enumeration,
    # but for single-group workloads perhaps consider increasing this.
    torch_threads: int = 1

    # Number of worker threads for processing groups. Nonpositive numbers
    # instead mean the number of available threads to leave open: 0 means use
    # all threads, and -2 means to leave two threads.
    max_workers: int = -2


@dataclass
class MainConfig:
    """Configuration of an entire run."""
    log: LogConfig = field(default_factory=LogConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    target: TargetStructureConfig = field(default_factory=TargetStructureConfig)
    cli: CliConfig = field(default_factory=CliConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)


def to_dict(config) -> dict:
    """Converts a config (dataclass) to a dictionary suitable for using with W&B."""
    with pyrallis.config_type('json'):
        print(json.loads(pyrallis.dump(config)))


if __name__ == '__main__':
    from rich.prompt import Confirm
    from pathlib import Path

    if Confirm.ask('Generate configs/defaults.toml and configs/minimal.toml?'):
        default_path = Path('configs') / 'defaults.toml'
        minimal_path = Path('configs') / 'minimal.toml'
        default = MainConfig(target=TargetStructureConfig(mp_id=PLACEHOLDER_MP_ID))
        with open(default_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile)

        with open(minimal_path, 'w') as outfile:
            pyrallis.cfgparsing.dump(default, outfile, omit_defaults=True)
