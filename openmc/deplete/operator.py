"""OpenMC transport operator

This module implements a transport operator for OpenMC so that it can be used by
depletion integrators. The implementation makes use of the Python bindings to
OpenMC's C API so that reading tally results and updating material number
densities is all done in-memory instead of through the filesystem.

"""

import copy
from copy import deepcopy
from collections import OrderedDict
import os
from warnings import warn
import re

import numpy as np
from uncertainties import ufloat

import openmc
import openmc.model
from openmc.checkvalue import check_value
from openmc.data import DataLibrary
from openmc.exceptions import DataError
import openmc.lib
from openmc.mpi import comm
from .abc import TransportOperator, OperatorResult
from .atom_number import AtomNumber
from .chain import _find_chain_file
from .reaction_rates import ReactionRates
from .results_list import ResultsList
from .helpers import (
    DirectReactionRateHelper, ChainFissionHelper, ConstantFissionYieldHelper,
    FissionYieldCutoffHelper, AveragedFissionYieldHelper, EnergyScoreHelper,
    SourceRateHelper, FluxCollapseHelper)


__all__ = ["Operator", "OperatorResult"]


def _distribute(items):
    """Distribute items across MPI communicator

    Parameters
    ----------
    items : list
        List of items of distribute

    Returns
    -------
    list
        Items assigned to process that called

    """
    min_size, extra = divmod(len(items), comm.size)
    j = 0
    for i in range(comm.size):
        chunk_size = min_size + int(i < extra)
        if comm.rank == i:
            return items[j:j + chunk_size]
        j += chunk_size


def _find_cross_sections(model):
    """Determine cross sections to use for depletion"""
    if model.materials and model.materials.cross_sections is not None:
        # Prefer info from Model class if available
        return model.materials.cross_sections

    # otherwise fallback to environment variable
    cross_sections = os.environ.get("OPENMC_CROSS_SECTIONS")
    if cross_sections is None:
        raise DataError(
            "Cross sections were not specified in Model.materials and "
            "the OPENMC_CROSS_SECTIONS environment variable is not set."
        )
    return cross_sections


class Operator(TransportOperator):
    """OpenMC transport operator for depletion.

    Instances of this class can be used to perform depletion using OpenMC as the
    transport operator. Normally, a user needn't call methods of this class
    directly. Instead, an instance of this class is passed to an integrator
    class, such as :class:`openmc.deplete.CECMIntegrator`.

    .. versionchanged:: 0.13.0
        The geometry and settings parameters have been replaced with a
        model parameter that takes an openmc.Model object

    Parameters
    ----------
    model : openmc.Model
        OpenMC model object
    chain_file : str, optional
        Path to the depletion chain XML file.  Defaults to the file
        listed under ``depletion_chain`` in
        :envvar:`OPENMC_CROSS_SECTIONS` environment variable.
    prev_results : ResultsList, optional
        Results from a previous depletion calculation. If this argument is
        specified, the depletion calculation will start from the latest state
        in the previous results.
    diff_burnable_mats : bool, optional
        Whether to differentiate burnable materials with multiple instances.
        Volumes are divided equally from the original material volume.
        Default: False.
    normalization_mode : {"energy-deposition", "fission-q", "source-rate"}
        Indicate how tally results should be normalized. ``"energy-deposition"``
        computes the total energy deposited in the system and uses the ratio of
        the power to the energy produced as a normalization factor.
        ``"fission-q"`` uses the fission Q values from the depletion chain to
        compute the  total energy deposited. ``"source-rate"`` normalizes
        tallies based on the source rate (for fixed source calculations).
    fission_q : dict, optional
        Dictionary of nuclides and their fission Q values [eV]. If not given,
        values will be pulled from the ``chain_file``. Only applicable
        if ``"normalization_mode" == "fission-q"``
    dilute_initial : float, optional
        Initial atom density [atoms/cm^3] to add for nuclides that are zero
        in initial condition to ensure they exist in the decay chain.
        Only done for nuclides with reaction rates.
        Defaults to 1.0e3.
    fission_yield_mode : {"constant", "cutoff", "average"}
        Key indicating what fission product yield scheme to use. The
        key determines what fission energy helper is used:

        * "constant": :class:`~openmc.deplete.helpers.ConstantFissionYieldHelper`
        * "cutoff": :class:`~openmc.deplete.helpers.FissionYieldCutoffHelper`
        * "average": :class:`~openmc.deplete.helpers.AveragedFissionYieldHelper`

        The documentation on these classes describe their methodology
        and differences. Default: ``"constant"``
    fission_yield_opts : dict of str to option, optional
        Optional arguments to pass to the helper determined by
        ``fission_yield_mode``. Will be passed directly on to the
        helper. Passing a value of None will use the defaults for
        the associated helper.
    reaction_rate_mode : {"direct", "flux"}, optional
        Indicate how one-group reaction rates should be calculated. The "direct"
        method tallies transmutation reaction rates directly. The "flux" method
        tallies a multigroup flux spectrum and then collapses one-group reaction
        rates after a transport solve (with an option to tally some reaction
        rates directly).

        .. versionadded:: 0.12.1
    reaction_rate_opts : dict, optional
        Keyword arguments that are passed to the reaction rate helper class.
        When ``reaction_rate_mode`` is set to "flux", energy group boundaries
        can be set using the "energies" key. See the
        :class:`~openmc.deplete.helpers.FluxCollapseHelper` class for all
        options.

        .. versionadded:: 0.12.1
    reduce_chain : bool, optional
        If True, use :meth:`openmc.deplete.Chain.reduce` to reduce the
        depletion chain up to ``reduce_chain_level``. Default is False.

        .. versionadded:: 0.12
    reduce_chain_level : int, optional
        Depth of the search when reducing the depletion chain. Only used
        if ``reduce_chain`` evaluates to true. The default value of
        ``None`` implies no limit on the depth.

    step_removal : list of dicts, optional
        If not empty, depletable nuclides concentrations are changed accordingly
        before solving bateman equation.

        .. versionadded:: 0.12

    Attributes
    ----------
    model : openmc.Model
        OpenMC model object
    geometry : openmc.Geometry
        OpenMC geometry object
    settings : openmc.Settings
        OpenMC settings object
    dilute_initial : float
        Initial atom density [atoms/cm^3] to add for nuclides that
        are zero in initial condition to ensure they exist in the decay
        chain. Only done for nuclides with reaction rates.
    output_dir : pathlib.Path
        Path to output directory to save results.
    round_number : bool
        Whether or not to round output to OpenMC to 8 digits.
        Useful in testing, as OpenMC is incredibly sensitive to exact values.
    number : openmc.deplete.AtomNumber
        Total number of atoms in simulation.
    nuclides_with_data : set of str
        A set listing all unique nuclides available from cross_sections.xml.
    chain : openmc.deplete.Chain
        The depletion chain information necessary to form matrices and tallies.
    reaction_rates : openmc.deplete.ReactionRates
        Reaction rates from the last operator step.
    burnable_mats : list of str
        All burnable material IDs
    heavy_metal : float
        Initial heavy metal inventory [g]
    local_mats : list of str
        All burnable material IDs being managed by a single process
    prev_res : ResultsList or None
        Results from a previous depletion calculation. ``None`` if no
        results are to be used.
    diff_burnable_mats : bool
        Whether to differentiate burnable materials with multiple instances
    cleanup_when_done : bool
        Whether to finalize and clear the shared library memory when the
        depletion operation is complete. Defaults to clearing the library.
    """
    _fission_helpers = {
        "average": AveragedFissionYieldHelper,
        "constant": ConstantFissionYieldHelper,
        "cutoff": FissionYieldCutoffHelper,
    }

    def __init__(self, model, chain_file=None, prev_results=None,
                 diff_burnable_mats=False, normalization_mode="fission-q",
                 fission_q=None, dilute_initial=1.0e3,
                 fission_yield_mode="constant", fission_yield_opts=None,
                 reaction_rate_mode="direct", reaction_rate_opts=None,
                 reduce_chain=False, reduce_chain_level=None,
                 step_removal=None, keff_control=None):
        # check for old call to constructor
        if isinstance(model, openmc.Geometry):
            msg = "As of version 0.13.0 openmc.deplete.Operator requires an " \
                "openmc.Model object rather than the openmc.Geometry and " \
                "openmc.Settings parameters. Please use the geometry and " \
                "settings objects passed here to create a model with which " \
                "to generate the depletion Operator."
            raise TypeError(msg)

        # Determine cross sections / depletion chain
        cross_sections = _find_cross_sections(model)
        if chain_file is None:
            chain_file = _find_chain_file(cross_sections)

        check_value('fission yield mode', fission_yield_mode,
                    self._fission_helpers.keys())
        check_value('normalization mode', normalization_mode,
                    ('energy-deposition', 'fission-q', 'source-rate'))
        if normalization_mode != "fission-q":
            if fission_q is not None:
                warn("Fission Q dictionary will not be used")
                fission_q = None
        super().__init__(chain_file, fission_q, dilute_initial, prev_results)
        self.round_number = False
        self.model = model
        self.settings = model.settings
        self.geometry = model.geometry

        # determine set of materials in the model
        if not model.materials:
            model.materials = openmc.Materials(
                model.geometry.get_all_materials().values()
            )
        self.materials = model.materials

        self.diff_burnable_mats = diff_burnable_mats
        self.cleanup_when_done = True

        # Reduce the chain before we create more materials
        if reduce_chain:
            all_isotopes = set()
            for material in self.materials:
                if not material.depletable:
                    continue
                for name, _dens_percent, _dens_type in material.nuclides:
                    all_isotopes.add(name)
            self.chain = self.chain.reduce(all_isotopes, reduce_chain_level)

        # Differentiate burnable materials with multiple instances
        if self.diff_burnable_mats:
            self._differentiate_burnable_mats()

        # Clear out OpenMC, create task lists, distribute
        openmc.reset_auto_ids()
        self.burnable_mats, volume, nuclides = self._get_burnable_mats()
        self.local_mats = _distribute(self.burnable_mats)

        # Generate map from local materials => material index
        self._mat_index_map = {
            lm: self.burnable_mats.index(lm) for lm in self.local_mats}

        if self.prev_res is not None:
            # Reload volumes into geometry
            prev_results[-1].transfer_volumes(self.model)

            # Store previous results in operator
            # Distribute reaction rates according to those tracked
            # on this process
            if comm.size == 1:
                self.prev_res = prev_results
            else:
                self.prev_res = ResultsList()
                mat_indexes = _distribute(range(len(self.burnable_mats)))
                for res_obj in prev_results:
                    new_res = res_obj.distribute(self.local_mats, mat_indexes)
                    self.prev_res.append(new_res)

        self.step_removal = step_removal
        self.keff_control = keff_control

        # Determine which nuclides have incident neutron data
        self.nuclides_with_data = self._get_nuclides_with_data(cross_sections)

        # Select nuclides with data that are also in the chain
        self._burnable_nucs = [nuc.name for nuc in self.chain.nuclides
                               if nuc.name in self.nuclides_with_data]

        # Extract number densities from the geometry / previous depletion run
        self._extract_number(self.local_mats, volume, nuclides, self.prev_res)

        # Create reaction rates array
        self.reaction_rates = ReactionRates(
            self.local_mats, self._burnable_nucs, self.chain.reactions)

        # Get classes to assist working with tallies
        if reaction_rate_mode == "direct":
            self._rate_helper = DirectReactionRateHelper(
                self.reaction_rates.n_nuc, self.reaction_rates.n_react)
        elif reaction_rate_mode == "flux":
            if reaction_rate_opts is None:
                reaction_rate_opts = {}

            # Ensure energy group boundaries were specified
            if 'energies' not in reaction_rate_opts:
                raise ValueError(
                    "Energy group boundaries must be specified in the "
                    "reaction_rate_opts argument when reaction_rate_mode is"
                    "set to 'flux'.")

            self._rate_helper = FluxCollapseHelper(
                self.reaction_rates.n_nuc,
                self.reaction_rates.n_react,
                **reaction_rate_opts
            )
        else:
            raise ValueError("Invalid reaction rate mode.")

        if normalization_mode == "fission-q":
            self._normalization_helper = ChainFissionHelper()
        elif normalization_mode == "energy-deposition":
            score = "heating" if self.settings.photon_transport else "heating-local"
            self._normalization_helper = EnergyScoreHelper(score)
        else:
            self._normalization_helper = SourceRateHelper()

        # Select and create fission yield helper
        fission_helper = self._fission_helpers[fission_yield_mode]
        fission_yield_opts = (
            {} if fission_yield_opts is None else fission_yield_opts)
        self._yield_helper = fission_helper.from_operator(
            self, **fission_yield_opts)

    def __call__(self, vec, source_rate):
        """Runs a simulation.

        Simulation will abort under the following circumstances:

            1) No energy is computed using OpenMC tallies.

        Parameters
        ----------
        vec : list of numpy.ndarray
            Total atoms to be used in function.
        source_rate : float
            Power in [W] or source rate in [neutron/sec]

        Returns
        -------
        openmc.deplete.OperatorResult
            Eigenvalue and reaction rates resulting from transport operator

        """
        # Reset results in OpenMC
        openmc.lib.reset()
        # Update the number densities regardless of the source rate
        self.number.set_density(vec)
        self._update_materials()

        # If the source rate is zero, return zero reaction rates without running
        # a transport solve
        if source_rate == 0.0:
            rates = self.reaction_rates.copy()
            rates.fill(0.0)
            return OperatorResult(ufloat(0.0, 0.0), rates)

        # Prevent OpenMC from complaining about re-creating tallies
        openmc.reset_auto_ids()

        # Update tally nuclides data in preparation for transport solve
        nuclides = self._get_tally_nuclides()
        self._rate_helper.nuclides = nuclides
        self._normalization_helper.nuclides = nuclides
        self._yield_helper.update_tally_nuclides(nuclides)

        # Run OpenMC
        #if self.keff_control is not None and "surf_id" in self.keff_control.keys():
        #    openmc.lib.init_geom() #init_geom defined in core.py
        openmc.lib.run(output=False)
        openmc.lib.reset_timers()

        # Extract results
        op_result = self._unpack_tallies_and_normalize(source_rate)
        return copy.deepcopy(op_result)

    @staticmethod
    def write_bos_data(step):
        """Write a state-point file with beginning of step data

        Parameters
        ----------
        step : int
            Current depletion step including restarts
        """
        openmc.lib.statepoint_write(
            "openmc_simulation_n{}.h5".format(step),
            write_source=False)

    def _differentiate_burnable_mats(self):
        """Assign distribmats for each burnable material

        """

        # Count the number of instances for each cell and material
        self.geometry.determine_paths(instances_only=True)

        # Extract all burnable materials which have multiple instances
        distribmats = set(
            [mat for mat in self.materials
             if mat.depletable and mat.num_instances > 1])

        for mat in distribmats:
            if mat.volume is None:
                raise RuntimeError("Volume not specified for depletable "
                                    "material with ID={}.".format(mat.id))
            mat.volume /= mat.num_instances

        if distribmats:
            # Assign distribmats to cells
            for cell in self.geometry.get_all_material_cells().values():
                if cell.fill in distribmats:
                    mat = cell.fill
                    cell.fill = [mat.clone()
                                 for i in range(cell.num_instances)]

    def _get_burnable_mats(self):
        """Determine depletable materials, volumes, and nuclides

        Returns
        -------
        burnable_mats : list of str
            List of burnable material IDs
        volume : OrderedDict of str to float
            Volume of each material in [cm^3]
        nuclides : list of str
            Nuclides in order of how they'll appear in the simulation.

        """

        burnable_mats = set()
        model_nuclides = set()
        volume = OrderedDict()

        self.heavy_metal = 0.0

        # Iterate once through the geometry to get dictionaries
        for mat in self.materials:
            for nuclide in mat.get_nuclides():
                model_nuclides.add(nuclide)
            if mat.depletable:
                burnable_mats.add(str(mat.id))
                if mat.volume is None:
                    raise RuntimeError("Volume not specified for depletable "
                                       "material with ID={}.".format(mat.id))
                volume[str(mat.id)] = mat.volume
                self.heavy_metal += mat.fissionable_mass

        # Make sure there are burnable materials
        if not burnable_mats:
            raise RuntimeError(
                "No depletable materials were found in the model.")

        # Sort the sets
        burnable_mats = sorted(burnable_mats, key=int)
        model_nuclides = sorted(model_nuclides)

        # Construct a global nuclide dictionary, burned first
        nuclides = list(self.chain.nuclide_dict)
        for nuc in model_nuclides:
            if nuc not in nuclides:
                nuclides.append(nuc)

        return burnable_mats, volume, nuclides

    def make_keff_control(self, x, step_index):
        """Perform a separate search_for_keff in between depletion iterations
￼        and returned some optimezd paramters.
￼        The aim is for example to maintain a reactor critical (k_eff = 1)
         during a depletion analysis, parametrizing some user-defined variables,
         such as geometrical surfaces or cells (mimic for example the extraction
         of control rods) or changing some material composition (reproducing for
￼        example the effect of a nuclear refueling).

        Parameters
        ---------
        x : list of float
            Nuclides concentration from previous iteration
        step_index: int
            Depletion step index
        Returns
        -------
        x : list
            Nuclides concentration after search_for_keff

        """
        _model = deepcopy(self.model)

        #Create atoms vector dictionary for easier use
        atoms_vec = list()
        for idx,burn_id in enumerate(self.burnable_mats):
            atoms = OrderedDict()
            for id_nuc,nuc in enumerate(self.number.burnable_nuclides):
                atoms[nuc] = x[idx][id_nuc]
            atoms_vec.append(atoms)

        def keff_control_material(x, mat_id, range, bracketed_method, mat_comp,
                                  tol, target, density_limit,*args):

            if str(mat_id) not in self.burnable_mats:
                raise Exception(f'Mat_id: {mat_id} is not a valid depletable material id')

            def _model_builder(param):
                # In case of refueling, we need to modify the geometry before
                if args:
                    surf_id, res = args
                    # Set optimized output param back to the init value
                    for surf in _model.geometry.get_all_surfaces().items():
                        if surf[1].id == surf_id:
                            keys = list(surf[1].coefficients.keys())
                            if len(keys) == 1:
                                setattr(surf[1],keys[0],res)
                            else:
                                raise Exception(f'Surface coefficients {keys} are not one')

                for idx,burn_id in enumerate(self.burnable_mats):
                    for nuc,val in atoms_vec[idx].items():
                        if nuc not in mat_comp.keys():
                            if val < density_limit or nuc not in self._burnable_nucs:
                                _model.materials[int(burn_id)-1].remove_nuclide(nuc)
                            else:
                                _model.materials[int(burn_id)-1].remove_nuclide(nuc)
                                _model.materials[int(burn_id)-1].add_nuclide(nuc,val,'ao')
                        else:
                            if _model.materials[int(burn_id)-1].id == int(mat_id):
                                _model.materials[int(burn_id)-1].remove_nuclide(nuc)
                                # convert grams into atoms
                                atoms = param/openmc.data.atomic_mass(nuc)*openmc.data.AVOGADRO*mat_comp[nuc]
                                _model.materials[int(burn_id)-1].add_nuclide(nuc,val+atoms,'ao')
                            else:
                                _model.materials[int(burn_id)-1].remove_nuclide(nuc)
                                _model.materials[int(burn_id)-1].add_nuclide(nuc,val,'ao')
                    # ensure density is set
                    density = _model.materials[int(burn_id)-1].get_mass_density()
                    _model.materials[int(burn_id)-1].set_density('g/cm3',density)
                _model.export_to_xml()
                return _model

            res = None
            lower_range = range[0]
            upper_range = range[1]

            while res == None:
                search = openmc.search_for_keff(_model_builder,
                                bracket = [lower_range,upper_range], tol = tol,
                                bracketed_method = bracketed_method, target = target,
                                print_iterations = True)

                if len(search) == 3:
                    res, guesses, k = search

                elif len(search) == 2:
                    print ("Invalid range")
                    guesses, k = search

                    if np.array(k).prod() < target:
                        # If k is close enought to target (below 0.2%),get directly that value
                        if (target - np.array(k).max()) < 0.002:
                            index = [idx for idx,i in enumerate(k) if i == np.array(k).max()][0]
                            res =  guesses[index]
                        else:
                            lower_range = upper_range
                            upper_range *= 5
                    else:
                        if (np.array(k).min() -target)  < 0.002:
                            index = [idx for idx,i in enumerate(k) if i == np.array(k).min()][0]
                            res = guesses[index]
                        else:
                            upper_range = lower_range
                            lower_range /= 5
                else:
                    raise Exception(f'search_for_keff output not contemplated')

            # Update X vector with res
            diff = {}
            for idx,burn_id in enumerate(self.burnable_mats):
                atoms_per_mol = 0 #to calculate the new volume
                for id_nuc,nuc in enumerate(self.number.burnable_nuclides):
                    if nuc in mat_comp.keys() and int(burn_id) == int(mat_id):
                        # Convert res grams into atoms
                        res_atoms = res / openmc.data.atomic_mass(nuc) * openmc.data.AVOGADRO * mat_comp[nuc]
                        diff[nuc] = x[idx][id_nuc] - res_atoms
                        x[idx][id_nuc] += res_atoms
                    atoms_per_mol += x[idx][id_nuc]*openmc.data.atomic_mass(nuc)
                # Calculate new volume and assign
                vol = atoms_per_mol/openmc.data.AVOGADRO/self.materials[int(self.burnable_mats[idx])-1].get_mass_density()
                self.number.volume[idx] = vol
            return x, diff

        def keff_control_geometrical(x, surf_id, range, bracketed_method,
                                     init_param, tol, target, density_limit):

            for surf in self.geometry.get_all_surfaces().items():
                    if surf[1].id == surf_id:
                        keys = list(surf[1].coefficients.keys())
                        if len(keys) == 1:
                            guess = getattr(surf[1],keys[0])
                        else:
                            raise Exception(f'Surface coefficients {keys} are not one')

            def _model_builder(param):
                for surf in _model.geometry.get_all_surfaces().items():
                    if surf[1].id == surf_id:
                        keys = list(surf[1].coefficients.keys())
                        if len(keys) == 1:
                            setattr(surf[1],keys[0],param)
                        else:
                            raise Exception(f'Surface coefficients {keys} are not one')

                for idx,burn_id in enumerate(self.burnable_mats):
                    for nuc in _model.materials[int(burn_id)-1].get_nuclides():
                        _model.materials[int(burn_id)-1].remove_nuclide(nuc)
                    # add new nuclides and new volume in one go
                    atoms_per_mol = 0
                    for nuc,val in atoms_vec[idx].items():
                        if val > density_limit and nuc in self._burnable_nucs:
                            _model.materials[int(burn_id)-1].add_nuclide(nuc,val,'ao')
                            atoms_per_mol += val*openmc.data.atomic_mass(nuc)
                    #ensure density is set
                    density = _model.materials[int(burn_id)-1].get_mass_density()
                    _model.materials[int(burn_id)-1].set_density('g/cm3',density)
                    # calculate new volume and assign
                    vol = atoms_per_mol/openmc.data.AVOGADRO/density
                    self.number.volume[idx] = vol
                _model.export_to_xml()
                return _model

            res = None
            lower_range = range[0]
            upper_range = range[1]

            #normalize Tolerance for search_for_keff
            if -1.0 < guess < 1.0:
                tolerance = tol/2
            else:
                tolerance = abs(tol/guess)

            while res == None:
                if self.keff_control['refine_search'] and guess >= abs(range[2])/2:
                    check_brackets = True
                else:
                    check_brackets = False
                search = openmc.search_for_keff(_model_builder,
                                bracket=[guess+lower_range,guess+upper_range],
                                tol=tolerance, bracketed_method=bracketed_method,
                                target=target, print_iterations=True,
                                check_brackets = check_brackets)

                if len(search) == 3:
                    _res, guesses, k = search
                    # Further check, in case upper limit get hit
                    if _res <= range[2]:
                        res = _res
                    else:
                        if self.keff_control['refuel']:
                            print(f'Upper limit reached, level brought down to {init_param}cm and \
                            start refueling...')
                            res = init_param
                            break
                        else:
                            raise Exception(f'Upper limit reached, stopping depletion')

                elif len(search) == 2:
                    guesses, k = search

                    if guesses[-1] > range[2]:
                        if self.keff_control['refuel']:
                            print(f'Upper limit reached, level brought down to {init_param}cm and \
                            start refueling...')
                            res = init_param
                            break
                        else:
                            raise Exception(f'Upper limit reached, stopping depletion')

                    if np.array(k).prod() < target:
                        print ("Invalid range, below target")
                        if (target - np.array(k).prod()) <= 0.02:
                            lower_range = upper_range - 3
                        elif 0.02 < (target - np.array(k).prod()) < 0.03:
                            lower_range = upper_range - 1
                        else:
                            lower_range = upper_range - 0.5
                        upper_range += abs(range[1])/2

                    else:
                        print ("Invalid range, above target")
                        upper_range = lower_range + 2
                        lower_range -= abs(range[0])

                else:
                    raise Exception(f'search_for_keff output not contemplated')

            if 'plot' in self.keff_control.keys():
                from shutil import move
                if 'plots' not in os.listdir(os.getcwd()):
                    os.mkdir('plots')
                openmc.plot_geometry(output=False)
                for plot in os.listdir(os.getcwd()):
                    if plot.startswith('plot_'):
                        name = '_'.join([plot.split('_')[0],str(step_index)])
                        suffix = plot.split('.')[1]
                        rename = '.'.join([name,suffix])
                        os.rename(plot, rename)
                        move(rename,'plots')

            # Set res in the self geometry model
            for surf in self.geometry.get_all_surfaces().items():
                if surf[1].id == surf_id:
                    keys = list(surf[1].coefficients.keys())
                    if len(keys) == 1:
                        setattr(surf[1],keys[0],res)
                    else:
                        msg = (f'Surface coefficients {keys} are not one')
                        raise Exception(msg)
            print(f'res: {res}, guess: {guess}')
            diff = res - guess

            # In case of hitting top limit, refuel
            if res == init_param:
                refuel = self.keff_control['refuel']
                x, diff_mat = keff_control_material(x, refuel['mat_id'], refuel['range'],
                            refuel['bracketed_method'], refuel['mat_comp'],
                            refuel['tol'], target, density_limit, surf_id, res)
            return x, diff

        if "mat_id" in self.keff_control.keys():
            x, diff = keff_control_material(x, self.keff_control['mat_id'],
                        self.keff_control['range'], self.keff_control['bracketed_method'],
                        self.keff_control['mat_comp'],
                        self.keff_control['tol'], self.keff_control['target'],
                        self.keff_control['density_limit'])

        elif "surf_id" in self.keff_control.keys():
            x, diff = keff_control_geometrical(x, self.keff_control['surf_id'],
                        self.keff_control['range'], self.keff_control['bracketed_method'],
                        self.keff_control['init_param'],
                        self.keff_control['tol'], self.keff_control['target'],
                        self.keff_control['density_limit'])
            openmc.lib.init_geom() # reinitilize geometry in memory

        elif "mat_id" and "surf_id" in self.keff_control.keys():
            raise Exception("Too many keys")

        else:
            raise Exception(f'keff_search depletion Keys are not recognized')

        return x, diff

    def _extract_number(self, local_mats, volume, nuclides, remove_nuc, prev_res=None):
        """Construct AtomNumber using geometry

        Parameters
        ----------
        local_mats : list of str
            Material IDs to be managed by this process
        volume : OrderedDict of str to float
            Volumes for the above materials in [cm^3]
        nuclides : list of str
            Nuclides to be used in the simulation.
        prev_res : ResultsList, optional
            Results from a previous depletion calculation

        """
        self.number = AtomNumber(local_mats, nuclides, volume, len(self.chain))

        if self.dilute_initial != 0.0:
            for nuc in self._burnable_nucs:
                self.number.set_atom_density(np.s_[:], nuc, self.dilute_initial)

        # Now extract and store the number densities
        # From the geometry if no previous depletion results
        if prev_res is None:
            for mat in self.materials:
                if str(mat.id) in local_mats:
                    self._set_number_from_mat(mat)

        # Else from previous depletion results
        else:
            for mat in self.materials:
                if str(mat.id) in local_mats:
                    self._set_number_from_results(mat, prev_res, remove_nuc)

    def _set_number_from_mat(self, mat):
        """Extracts material and number densities from openmc.Material

        Parameters
        ----------
        mat : openmc.Material
            The material to read from

        """
        mat_id = str(mat.id)

        for nuclide, density in mat.get_nuclide_atom_densities().values():
            number = density * 1.0e24
            self.number.set_atom_density(mat_id, nuclide, number)

    def _set_number_from_results(self, mat, prev_res):
        """Extracts material nuclides and number densities.

        If the nuclide concentration's evolution is tracked, the densities come
        from depletion results. Else, densities are extracted from the geometry
        in the summary.

        Parameters
        ----------
        mat : openmc.Material
            The material to read from
        prev_res : ResultsList
            Results from a previous depletion calculation

        """
        mat_id = str(mat.id)

        # Get nuclide lists from geometry and depletion results
        depl_nuc = prev_res[-1].nuc_to_ind
        geom_nuc_densities = mat.get_nuclide_atom_densities()
        # Merge lists of nuclides, with the same order for every calculation
        geom_nuc_densities.update(depl_nuc)
        for nuclide in geom_nuc_densities.keys():
            if nuclide in depl_nuc:
                concentration = prev_res.get_atoms(mat_id, nuclide)[1][-1]
                volume = prev_res[-1].volume[mat_id]
                number = concentration / volume
            else:
                density = geom_nuc_densities[nuclide][1]
                number = density * 1.0e24

            self.number.set_atom_density(mat_id, nuclide, number)

    def make_step_removal(self, x):
        """ Change depletable material nuclide concentrations

        If step_removal attribute is defined and passed to the
        operator class, this method will modify depletable nuclide
        concentration accordingly, before solving bateman equation.

        Parameters
        ----------
        x : list of float
            Nuclides concentration from previous iteration

        Returns
        -------
        x : list of float
            Nuclides concentration after change, based of step_removal
            attribute

        """

        # to split Alphanumeric string (ex. U233)
        regex = re.compile(r'(\d+|\s+)')
        # get info from get_results_info method
        mat_with_vol, nucs, burns, full_burns  = self.get_results_info()
        # Iterate list of dicts
        for item in self.step_removal:
            # find self.keff_control['mat_id'] index
            m = [idx for idx, mat in enumerate(mat_with_vol.keys())
                 if mat == str(item['mat_id'])][0]
            # Iterate all nuclides in depletable materials
            for i,nuc in enumerate(nucs):
                # Nuclide can either match itsel or the element they correspond,
                # depending on the assignment
                if ("nuc" in item.keys() and nuc in item["nuc"]) or (
                    "elem" in item.keys() and regex.split(nuc)[0] in item["elem"]):
                    # Remove element from concentration vector, but add few atoms
                    # for the decay chain
                    if item['action'] == 'remove':
                        x[m][i] -= x[m][i] * item['efficiency'] / 100
                        x[m][i] += 1.0e3
                    # Swap nuclide from one material to another
                    elif item['action'] == 'add':
                        # find index of material from which to add
                        a = [idx for idx, mat in enumerate(mat_with_vol.keys())
                            if mat == str(item['from'])][0]
                        x[m][i] += x[a][i] * item['efficiency'] / 100
                        x[a][i] -= x[a][i] * item['efficiency'] / 100
                        x[a][i] += 1.0e3
                    else:
                        msg = (f'Action in {item} is neither remove nor add')
                        raise Exception(msg)

        return x

    def initial_condition(self):
        """Performs final setup and returns initial condition.

        Returns
        -------
        list of numpy.ndarray
            Total density for initial conditions.
        """

        # Create XML files
        if comm.rank == 0:
            self.geometry.export_to_xml()
            self.settings.export_to_xml()
            self._generate_materials_xml()

        # Initialize OpenMC library
        comm.barrier()
        if not openmc.lib.is_initialized:
            openmc.lib.init(intracomm=comm)

        # Generate tallies in memory
        materials = [openmc.lib.materials[int(i)]
                     for i in self.burnable_mats]
        self._rate_helper.generate_tallies(materials, self.chain.reactions)
        self._normalization_helper.prepare(
            self.chain.nuclides, self.reaction_rates.index_nuc)
        # Tell fission yield helper what materials this process is
        # responsible for
        self._yield_helper.generate_tallies(
            materials, tuple(sorted(self._mat_index_map.values())))

        # Return number density vector
        return list(self.number.get_mat_slice(np.s_[:]))

    def finalize(self):
        """Finalize a depletion simulation and release resources."""
        if self.cleanup_when_done:
            openmc.lib.finalize()

    def _update_materials(self):
        """Updates material compositions in OpenMC on all processes."""

        for rank in range(comm.size):
            number_i = comm.bcast(self.number, root=rank)

            for mat in number_i.materials:
                nuclides = []
                densities = []
                for nuc in number_i.nuclides:
                    if nuc in self.nuclides_with_data:
                        val = 1.0e-24 * number_i.get_atom_density(mat, nuc)

                        # If nuclide is zero, do not add to the problem.
                        if val > 0.0:
                            if self.round_number:
                                val_magnitude = np.floor(np.log10(val))
                                val_scaled = val / 10**val_magnitude
                                val_round = round(val_scaled, 8)

                                val = val_round * 10**val_magnitude

                            nuclides.append(nuc)
                            densities.append(val)
                        else:
                            # Only output warnings if values are significantly
                            # negative. CRAM does not guarantee positive values.
                            if val < -1.0e-21:
                                print("WARNING: nuclide ", nuc, " in material ", mat,
                                      " is negative (density = ", val, " at/barn-cm)")
                            number_i[mat, nuc] = 0.0

                # Update densities on C API side
                mat_internal = openmc.lib.materials[int(mat)]
                mat_internal.set_densities(nuclides, densities)
                #print(f'Mat {mat_internal.name} -> density: {mat_internal.get_density(units="g/cm3")} ')
                #TODO Update densities on the Python side, otherwise the
                # summary.h5 file contains densities at the first time step

    def _generate_materials_xml(self):
        """Creates materials.xml from self.number.

        Due to uncertainty with how MPI interacts with OpenMC API, this
        constructs the XML manually.  The long term goal is to do this
        through direct memory writing.

        """
        # Sort nuclides according to order in AtomNumber object
        nuclides = list(self.number.nuclides)
        for mat in self.materials:
            mat._nuclides.sort(key=lambda x: nuclides.index(x[0]))

        self.materials.export_to_xml()

    def _get_tally_nuclides(self):
        """Determine nuclides that should be tallied for reaction rates.

        This method returns a list of all nuclides that have neutron data and
        are listed in the depletion chain. Technically, we should tally nuclides
        that may not appear in the depletion chain because we still need to get
        the fission reaction rate for these nuclides in order to normalize
        power, but that is left as a future exercise.

        Returns
        -------
        list of str
            Tally nuclides

        """
        nuc_set = set()

        # Create the set of all nuclides in the decay chain in materials marked
        # for burning in which the number density is greater than zero.
        for nuc in self.number.nuclides:
            if nuc in self.nuclides_with_data:
                if np.sum(self.number[:, nuc]) > 0.0:
                    nuc_set.add(nuc)

        # Communicate which nuclides have nonzeros to rank 0
        if comm.rank == 0:
            for i in range(1, comm.size):
                nuc_newset = comm.recv(source=i, tag=i)
                nuc_set |= nuc_newset
        else:
            comm.send(nuc_set, dest=0, tag=comm.rank)

        if comm.rank == 0:
            # Sort nuclides in the same order as self.number
            nuc_list = [nuc for nuc in self.number.nuclides
                        if nuc in nuc_set]
        else:
            nuc_list = None

        # Store list of tally nuclides on each process
        nuc_list = comm.bcast(nuc_list)
        return [nuc for nuc in nuc_list if nuc in self.chain]

    def _unpack_tallies_and_normalize(self, source_rate):
        """Unpack tallies from OpenMC and return an operator result

        This method uses OpenMC's C API bindings to determine the k-effective
        value and reaction rates from the simulation. The reaction rates are
        normalized by a helper class depending on the method being used.

        Parameters
        ----------
        source_rate : float
            Power in [W] or source rate in [neutron/sec]

        Returns
        -------
        openmc.deplete.OperatorResult
            Eigenvalue and reaction rates resulting from transport operator

        """
        rates = self.reaction_rates
        rates.fill(0.0)

        # Get k and uncertainty
        k_combined = ufloat(*openmc.lib.keff())

        # Extract tally bins
        nuclides = self._rate_helper.nuclides

        # Form fast map
        nuc_ind = [rates.index_nuc[nuc] for nuc in nuclides]
        react_ind = [rates.index_rx[react] for react in self.chain.reactions]

        # Keep track of energy produced from all reactions in eV per source
        # particle
        self._normalization_helper.reset()
        self._yield_helper.unpack()

        # Store fission yield dictionaries
        fission_yields = []

        # Create arrays to store fission Q values, reaction rates, and nuclide
        # numbers, zeroed out in material iteration
        number = np.empty(rates.n_nuc)

        fission_ind = rates.index_rx.get("fission")

        # Extract results
        for i, mat in enumerate(self.local_mats):
            # Get tally index
            mat_index = self._mat_index_map[mat]

            # Zero out reaction rates and nuclide numbers
            number.fill(0.0)

            # Get new number densities
            for nuc, i_nuc_results in zip(nuclides, nuc_ind):
                number[i_nuc_results] = self.number[mat, nuc]

            tally_rates = self._rate_helper.get_material_rates(
                mat_index, nuc_ind, react_ind)

            # Compute fission yields for this material
            fission_yields.append(self._yield_helper.weighted_yields(i))

            # Accumulate energy from fission
            if fission_ind is not None:
                self._normalization_helper.update(tally_rates[:, fission_ind])

            # Divide by total number and store
            rates[i] = self._rate_helper.divide_by_adens(number)

        # Scale reaction rates to obtain units of reactions/sec
        rates *= self._normalization_helper.factor(source_rate)

        # Store new fission yields on the chain
        self.chain.fission_yields = fission_yields

        return OperatorResult(k_combined, rates)

    def _get_nuclides_with_data(self, cross_sections):
        """Loads cross_sections.xml file to find nuclides with neutron data"""
        nuclides = set()
        data_lib = DataLibrary.from_xml(cross_sections)
        for library in data_lib.libraries:
            if library['type'] != 'neutron':
                continue
            for name in library['materials']:
                if name not in nuclides:
                    nuclides.add(name)

        return nuclides

    def get_results_info(self):
        """Returns volume list, material lists, and nuc lists.

        Returns
        -------
        volume : dict of str float
            Volumes corresponding to materials in full_burn_dict
        nuc_list : list of str
            A list of all nuclide names. Used for sorting the simulation.
        burn_list : list of int
            A list of all material IDs to be burned.  Used for sorting the simulation.
        full_burn_list : list
            List of all burnable material IDs

        """
        nuc_list = self.number.burnable_nuclides
        burn_list = self.local_mats

        volume = {}
        for i, mat in enumerate(burn_list):
            volume[mat] = self.number.volume[i]

        # Combine volume dictionaries across processes
        volume_list = comm.allgather(volume)
        volume = {k: v for d in volume_list for k, v in d.items()}

        return volume, nuc_list, burn_list, self.burnable_mats
