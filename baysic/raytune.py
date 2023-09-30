from ray import tune, air
import numpy as np

import numpy as np
from pyxtal import Group
from pymatgen.core import Composition
from pymatgen.core import Structure, Lattice
from pyxtal import Wyckoff_position
from feature_space import FeatureSpace
import numpy as np

lat_params = {
    'cubic': 'a',
    'hexagonal': ('a', 'c'),
    'tetragonal': ('a', 'c'),
    'orthorhombic': ('a', 'b', 'c'),
    'triclinic': ('a', 'b', 'c', 'alpha', 'beta', 'gamma'),
    'monoclinic': ('a', 'b', 'c', 'beta'),    
}

lat_funcs = {k: getattr(Lattice, k) if k != 'triclinic' else Lattice.from_parameters for k in lat_params.keys()}

class RaytuneOptimizer:
    def __init__(self, sg: Group, composition: Composition, scale: int):
        self.sg = sg
        self.comp = composition
        n_els = np.array(list(composition.values()))
        self.combs = []
        self.scales = []
        for sc in [scale]:
            scale_combs, _has_freedom, _inds = sg.list_wyckoff_combinations(n_els * sc)
            self.combs.extend(scale_combs)
            self.scales.extend([scale] * len(scale_combs))        

        if not self.combs:
            raise ValueError('No valid Wyckoff combinations', self.sg, self.comp)

        self.lattice = {param: tune.uniform(2, 30) if param in 'abc' else tune.uniform(20, 160) for param in lat_params[sg.lattice_type]}

        def coord_space(config):
            comb = self.combs[config['wp_i']]    
            symbs = sum(comb, start=[])
            dof = sg.get_site_dof(symbs)
            return np.random.uniform(size=int(sum(dof)))

        wp_i = {'wp_i': tune.randint(0, len(self.combs))}
        coords = {'wp_xyz': tune.sample_from(coord_space)}

        self.config = {}
        for d in (self.lattice, wp_i, coords):
            self.config.update(d)    


    def fit(self, objective, **kwargs):
        def config_objective(config):
            return objective(self.make_struct(config))
        
        default_kwargs = dict(mode='min', num_samples=30, max_concurrent_trials=5)
        default_kwargs.update(kwargs)
        self.tuner = tune.Tuner(
            config_objective,
            param_space=self.config,
            tune_config = tune.TuneConfig(
                **default_kwargs
            ),
            run_config=air.RunConfig(verbose=0)
        )

        self.res = self.tuner.fit().get_best_result()
        return (self.make_struct(self.res.config), self.res.config)

    def make_struct(self, config):
        lat_vals = [config[param] for param in self.lattice]
        lat = lat_funcs[self.sg.lattice_type](*lat_vals)
        fs = FeatureSpace(self.sg.number)    
        flat_combs = []
        flat_elems = []
        
        comb = self.combs[config['wp_i']]
        for el_group, el in zip(comb, self.comp.elements):
            flat_combs.extend(el_group)
            flat_elems.extend([el] * len(el_group))

        dofs = fs.sg.get_site_dof(flat_combs)
        coord_i = 0
        coords = []
        elems = []        
        for letter, elem, dof in zip(flat_combs, flat_elems, dofs):
            dof = int(dof)
            wp = Wyckoff_position.from_group_and_letter(self.sg.number, letter)
            wp_pos = config['wp_xyz'][coord_i:coord_i+dof] if len(config['wp_xyz'].shape) > 0 else []
            positions = wp.get_all_positions(fs.from_free_transformed_xyz(wp_pos, wp))
            coords.append(positions)
            elems.extend([elem] * len(positions))
            coord_i += dof

        coords = np.vstack(coords)
        return Structure(lat, elems, coords)
