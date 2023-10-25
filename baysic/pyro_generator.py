"""Full stochastic structure generator using Pyro."""

from copy import deepcopy
import logging
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
from pymatgen.core import Composition, Lattice, Structure, Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from tqdm import trange
from baysic.errors import CoordinateGenerationFailed, StructureGenerationError, WyckoffAssignmentFailed, WyckoffAssignmentImpossible
from baysic.structure_evaluation import point_energy
from baysic.pyro_wp import WyckoffSet
from baysic.lattice import CubicLattice, OrthorhombicLattice, atomic_volume, LatticeModel
from baysic.utils import debug_shapes, get_group, pairwise_dist_ratio, pairwise_diag_dist_ratio
from baysic.config import SearchConfig, WyckoffSelectionStrategy, LogConfig
from pyxtal import Group
from baysic.wp_assignment import WYCKOFFS
from baysic.wyckoff_span import min_dist_ratio

class SystemStructureModel(PyroModule):
    """A stochastic structure generator working within a particular lattice type."""
    def __init__(
        self,
        log: LogConfig,
        config: SearchConfig,
        comp: Composition,
        lattice: LatticeModel,
        force_group: int | str | Group | None = None
    ):
        super().__init__()
        self.log = log
        self.config = config
        self.comp = comp
        self.elements = self.comp.elements
        self.lattice_model = lattice


        self.needs_tight_ratio = any([e.symbol in ['Pu', 'U', 'Ce', 'Np', 'Pa'] for e in comp.elements])
        # Pymatgen's CovalentRadius as a predictor of bond length becomes
        # looser and looser as the atomic number increases. The trans-uranium
        # elements, and to a lesser extent the lanthanides, can form bonds
        # at a much closer distance. Because this is so rare, we special-case it,
        # so the vast majority of crystals can use a tighter bound that will
        # eliminate more bad crystals faster. This also lets us generate better
        # lattices, because we can avoid generating lattices that are too small. Those
        # tend to waste a lot of computational power.
        if not self.needs_tight_ratio:
            # μ = 1.01, σ = 0.83
            self.MIN_DIST_RATIO = 0.7
            self.MIN_LATTICE_RATIO = 0.77
            self.volume_ratio = PyroSample(dist.LogNormal(loc=-0.252, scale=0.72))
        else:
            self.MIN_DIST_RATIO = 0.6
            self.MIN_LATTICE_RATIO = 0.6
            self.volume_ratio = PyroSample(dist.FoldedDistribution(
            dist.SoftAsymmetricLaplace(
                loc=0.548, scale=.211, asymmetry=0.672, softness=0.03
            )))

        self.atom_volume = atomic_volume(comp)

        groups = self.lattice_model.get_groups()
        if force_group is not None:
            force_group = get_group(force_group)
            if force_group.number not in [g.number for g in groups]:
                raise StructureGenerationError(
                    self.comp,
                    self.lattice_model,
                    [force_group],
                    f'{force_group.symbol} is not compatible with the crystal system: {force_group.lattice_type} ≠ {self.lattice_model.lattice_type}'
                )

            groups = [force_group]


        self.group_options = []
        self.wyckoff_options = []
        self.group_cards = []
        self.opt_cards = []
        self.count_cards = []
        self.inds = []

        n_els = np.array(list(comp.values()))
        strategy = self.config.wyckoff_strategy
        if strategy == WyckoffSelectionStrategy.sample_distinct:
            for sg in groups:
                if WYCKOFFS[sg.number].can_make(self.comp):
                    self.group_options.append(sg)

            if len(self.group_options) == 0:
                raise WyckoffAssignmentImpossible(
                    self.comp,
                    self.lattice_model,
                    groups,
                    'No possible Wyckoff assignments'
                )

            self.group_opt = PyroSample(dist.Categorical(logits=torch.zeros(len(self.group_options))))

            self.log_info = {
                'num_assignments': 0
            }
        else:
            for sg in groups:
                combs, _has_freedom, _inds = sg.list_wyckoff_combinations(n_els)
                if combs:
                    self.group_options.extend([sg.number] * len(combs))
                    self.wyckoff_options.extend(combs)
                    self.group_cards.extend([1 / len(combs)] * len(combs))
                    self.opt_cards.extend([1] * len(combs))
                    self.count_cards.extend([len(sum(comb, [])) + 1 for comb in combs])

            if len(self.wyckoff_options) == 0:
                raise WyckoffAssignmentImpossible(
                    self.comp,
                    self.lattice_model,
                    groups,
                    'No possible Wyckoff assignments')

            strategy = self.config.wyckoff_strategy
            if strategy == WyckoffSelectionStrategy.uniform_sg:
                probs = torch.tensor(self.group_cards).float()
            elif strategy == WyckoffSelectionStrategy.uniform_wp:
                probs = torch.tensor(self.opt_cards).float()
            elif strategy == WyckoffSelectionStrategy.fewer_distinct:
                probs = torch.tensor(self.count_cards).float()
                probs = 0.2 ** (probs - min(probs))
            elif strategy == WyckoffSelectionStrategy.weighted_wp_count:
                probs = torch.tensor(self.count_cards).float()
                unique, frequency = probs.unique(return_counts=True)
                weights = 0.2 ** (unique - min(unique))
                for value, freq, weight in zip(unique, frequency, weights):
                    probs[probs == value] = weight / freq
            else:
                raise ValueError(f'Could not interpret {strategy}')

            self.probs = probs / probs.sum().float()
            self.wyck_opt = PyroSample(dist.Categorical(probs=self.probs))
            self.log_info = {
                'num_assignments': len(combs)
            }


    def forward(self):
        self.volume = (self.volume_ratio + self.MIN_LATTICE_RATIO) * self.atom_volume
        self.lattice = self.lattice_model(self.volume)()
        self.lattice_obj = Lattice(self.lattice.detach().cpu().numpy())

        strategy = self.config.wyckoff_strategy
        if strategy == WyckoffSelectionStrategy.sample_distinct:
            group = self.group_options[self.group_opt]
            num_atoms = list(self.comp.values())
            all_wps = group.Wyckoff_positions
            symb_to_wp = {wp.letter: wp for wp in all_wps}
            mults = torch.tensor([wp.multiplicity for wp in all_wps]).float()
            has_freedom = torch.tensor([wp.get_dof() != 0 for wp in all_wps])
            def try_assignment():
                complete_assignment = []
                for count in num_atoms:
                    removed = torch.zeros_like(has_freedom)
                    assignment = []
                    curr_count = count
                    while curr_count != 0:
                        is_possible = torch.where(~removed & (mults <= curr_count))[0]
                        if len(is_possible) == 0:
                            return None
                        weights = mults[is_possible] * 3
                        _uniq, inv, counts = torch.unique(mults[is_possible], return_inverse=True, return_counts=True)
                        weights /= counts[inv]
                        weights /= weights.sum()
                        selection = is_possible[dist.Categorical(probs=weights).sample()].item()
                        assignment.append(all_wps[selection].letter)
                        curr_count -= mults[selection].item()
                        if not has_freedom[selection]:
                            removed[selection] = True

                    complete_assignment.append(assignment)
                return complete_assignment

            tries = 0
            failed_span = 0
            comb = None
            while tries < 100 and comb is None:
                tries += 1
                comb = try_assignment()
                if comb is not None:
                    dist_ratio = min_dist_ratio(
                        group.number,
                        [symb_to_wp[x] for x in sum(comb, [])],
                        sum([[el] * len(wp) for wp, el in zip(comb, self.comp.elements)], []),
                        self.lattice_obj)
                    if dist_ratio < self.MIN_DIST_RATIO:
                        failed_span += 1
                        comb = None

            self.log_info['num_assignments'] = -tries
            self.log_info['num_assignments_failed_span'] = failed_span
            if comb is None:
                # we currently are unable to explicitly rule out many WP assignments
                # that are impossible to do without breaking one of the span requirements
                # if that's why we're failing, it's not that big of an issue
                if tries - failed_span <= 0.5 * tries:
                    raise CoordinateGenerationFailed(self.log_info,
                    "Couldn't find an assignment with feasible subspaces")

                # note the difference from WyckoffAssignmentImpossible: this should
                # (hopefully) never happen and is a problem with the model
                raise WyckoffAssignmentFailed(
                    self.comp,
                    self.lattice_model,
                    [group],
                    'Could not find valid Wyckoff assignment')

            self.sg = group.number
        else:
            opt = self.wyck_opt
            self.sg = self.group_options[opt]
            comb = self.wyckoff_options[opt]

        self.coords = torch.tensor([])
        self.elems = []
        self.wsets = []
        spots = sum(comb, [])
        elements = sum([[elem] * len(spots) for elem, spots in zip(self.comp.elements, comb)], [])
        wsets: list[WyckoffSet] = [WyckoffSet(self.sg, spot) for spot in spots]
        dofs: list[int] = np.array([wset.dof for wset in wsets])
        # WPs with 0 degrees of freedom should go first, because they're very cheap to expand out
        # then, letting the high-multiplicity elements go first is best
        # they're the toughest to place, and thus make the best use of parallelism

        mults = np.array([wset.multiplicity for wset in wsets])
        mult_order = np.argsort(-mults)
        no_dofs = mult_order[dofs[mult_order] == 0]
        some_dofs = mult_order[dofs[mult_order] != 0]

        if self.config.order_positions_by_radius:
            radii = np.array([CovalentRadius.radius[elem.symbol] for elem in np.array(elements)[some_dofs]])
            some_dofs = some_dofs[np.argsort(-radii)]
        best_order = np.concatenate([no_dofs, some_dofs])

        elements = np.array(elements)[best_order]
        wsets = np.array(wsets)[best_order]
        spots = np.array(spots)[best_order]

        self.log_info['volume_ratio'] = (self.volume / self.atom_volume).item()
        self.log_info['volume'] = self.volume.item()
        self.log_info['wyckoff_letters'] = '_'.join([wset.wp.letter for wset in wsets[np.argsort(best_order)]])
        self.log_info['total_dof'] = sum(dofs)
        self.log_info['group_num'] = self.sg
        self.log_info['num_total_coords'] = []
        self.log_info['num_filtered_coords'] = []
        self.log_info['num_outputs'] = 0
        self.log_info['num_distance_checks'] = 0

        for spot, elem, wset in zip(spots, elements, wsets):
            radius = torch.tensor([CovalentRadius.radius[elem.symbol]])

            wset = WyckoffSet(self.sg, spot)
            if wset.dof == 0:
                posns = torch.zeros(3)
                set_coords = wset.to_all_positions(posns)
            else:
                base = torch.cartesian_prod(*[torch.linspace(0, 1, self.config.n_grid + 2)[1:-1] for _ in range(wset.dof)])
                debug_shapes('base')
                base = base.reshape(self.config.n_grid ** wset.dof, wset.dof)
                max_move = 0.49 / (self.config.n_grid + 1)
                low = base - max_move
                high = base + max_move
                posns = pyro.sample(f'coords_{len(self.elems)}', dist.Uniform(low, high))

                set_coords = wset.to_all_positions(wset.to_asu(posns))

            debug_shapes('set_coords', 'posns')
            if set_coords.shape[-2] > 1:
                # check pairwise distances
                self.log_info['num_distance_checks'] += set_coords.shape[0] * set_coords.shape[1]
                set_diffs = pairwise_diag_dist_ratio(set_coords, radius, self.lattice)
                debug_shapes('set_diffs')
                # [ngrid, 1] if used a grid search
                # [1, 1, 1, dof - 1] if no degrees of freedom
                # here, we only care about comparing a single WP to its own copies, not the full pairwise
                # n_new_coords = set_diffs.shape[0]
                # set_diffs = torch.diag(set_diffs)
                # set_diffs = set_diffs[torch.arange(n_new_coords), 0, torch.arange(n_new_coords), :].reshape(-1, set_diffs.shape[-1])
                # [ngrid, dof - 1]
                set_valid = set_diffs >= self.MIN_DIST_RATIO
                debug_shapes('set_valid')
                good_all_coords = set_coords[set_valid, :, :]
            else:
                # 1 coordinate is always valid
                good_all_coords = set_coords

            if not good_all_coords.numel():
                raise CoordinateGenerationFailed(self.log_info, 'No structures found with acceptable inter-atomic distances')

            debug_shapes('set_coords', 'good_all_coords')
            # only need to check base coord
            # good_coords = good_all_coords[:, :1, :]
            good_coords = good_all_coords

            if self.coords.numel():
                radii = torch.tensor([CovalentRadius.radius[el.symbol] for el in self.elems])
                coords = self.coords
                # print(self.elems, self.wsets, wset.multiplicity)
                self.log_info['num_distance_checks'] += coords.shape[0] * good_coords.shape[0] * coords.shape[1] * good_coords.shape[1]

                # shape [coords_batch, coords_num, good_batch, good_num]
                # shape [coords_batch, good_batch]

                min_cdists = pairwise_dist_ratio(good_coords, coords, radius, radii, self.lattice)
                debug_shapes('good_coords', 'coords', 'radius', 'radii', 'min_cdists')
                # min_cdists = cdists.permute((0, 2, 1, 3)).min(dim=-1)[0].min(dim=-1)[0]

                good_batch, coord_batch = min_cdists.shape
                min_cdists = min_cdists.flatten()
                k = min(min_cdists.numel(), self.config.n_parallel_structs)
                dists, inds = torch.topk(min_cdists, k, largest=True, sorted=False)
                inds = inds[dists >= self.MIN_DIST_RATIO]
                dists = dists[dists >= self.MIN_DIST_RATIO]
                if inds.numel() == 0:
                    raise CoordinateGenerationFailed(
                        self.log_info,
                        'No structures found with acceptable inter-atomic distances'
                    )

                # convert flat index i to original good_batch, coord_batch
                all_new = inds // coord_batch
                all_old = inds % coord_batch

                self.log_info['num_total_coords'].append(all_new.numel())

                old = self.coords[all_old]
                new = good_all_coords[all_new]
                debug_shapes('old', 'new')
                self.coords = torch.cat([old, new], dim=1)
                # self.coords.append(set_coords[torch.where(set_valid)[0][0]].unsqueeze(0))

            else:
                # no other coordinates to worry about, just add all found coordinates
                self.log_info['num_total_coords'].append(good_all_coords.shape[0])
                self.coords = good_all_coords

            self.log_info['num_filtered_coords'].append(self.coords.shape[0])
            self.elems.extend([elem] * wset.multiplicity)
            self.wsets.append(wset)

            # radii = torch.tensor([CovalentRadius.radius[el.symbol] for el in self.elems])
            # if self.coords.shape[1] > 1:
            #     for single_coords in self.coords:
            #         pdists = pairwise_dist_ratio(single_coords[1:].unsqueeze(0), single_coords[[0]].unsqueeze(0), radii[1:], radii[0], self.lattice)
            #         if pdists.min() <= self.MIN_DIST_RATIO:
            #             raise ValueError('Ah shit')


        self.log_info['num_outputs'] = self.coords.shape[0]


        return (self.coords, self.lattice, self.elems, self.wsets, self.sg)

    def to_structures(self) -> list[Structure]:
        np_coords = self.coords.detach().cpu().numpy()
        lattice = self.lattice_obj
        return [Structure(lattice, self.elems, coords) for coords in np_coords]

    def to_gen_coords(self) -> torch.Tensor:
        """Gets just the free coordinates."""
        curr_i = 0
        coords = []
        for wset in self.wsets:
            if wset.dof == 0:
                # no general coordinates to add
                curr_i += wset.multiplicity
                continue
            else:
                coords.append(wset.inverse(self.coords[..., curr_i, :]))
                curr_i += wset.multiplicity
        return torch.cat(coords, dim=-1)




if __name__ == '__main__':
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.progress import track
    from rich.logging import RichHandler
    from baysic.lattice import OrthorhombicLattice, HexagonalLattice
    from baysic.structure_evaluation import is_structure_valid
    from baysic.utils import struct_dist_ratio
    console = Console()
    logging.basicConfig(
        level="INFO", format='%(message)s', datefmt="[%X]", handlers=[RichHandler()]
    )
    mod = SystemStructureModel(
        LogConfig(),
        SearchConfig(order_positions_by_radius=False, wyckoff_strategy=WyckoffSelectionStrategy.sample_distinct,
                     rng_seed=1234),
        # Composition({'Mg': 8, 'Al': 16, 'O': 32}),
        # Composition.from_dict({'K': 8, 'Li': 4, 'Cr': 4, 'F': 24}),
        Composition('Ba2YCu3O7'),
        OrthorhombicLattice,
        force_group=16
    )

    rows = []
    for _ in track(range(100)):
        try:
            coords, lat, elems, wsets, sg = mod.forward()
            new_structs = mod.to_structures()
            for i, struct in enumerate(new_structs):
                dists = struct.distance_matrix
                radii = np.array([CovalentRadius.radius[site.specie.symbol] for site in struct.sites])
                rads = np.add.outer(radii, radii)
                ratios = dists / rads
                ratios += np.eye(len(rads)) * 100
                min_i = np.argmin(ratios)
                min_pair = (min_i // len(ratios), min_i % len(ratios))
                sites = [struct.sites[p] for p in min_pair]
                if ratios.min() <= 0.7:
                    raise ValueError('Uh oh!')

        except CoordinateGenerationFailed as e:
            # console.print(e)
            pass
        finally:
            rows.append(deepcopy(mod.log_info))

    df = pd.DataFrame(rows)
    df['num_checked'] = df['num_filtered_coords'].apply(sum)
    console.print(Markdown(df.to_markdown()))
    console.print(Markdown(df.describe(include=np.number).to_markdown()))
    console.print(Markdown(df.query('num_outputs == 0').describe(include=np.number).to_markdown()))
