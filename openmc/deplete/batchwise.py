from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from warnings import warn
from numbers import Real

import numpy as np
import h5py
import os
import re

from openmc.checkvalue import check_type, check_value, check_less_than, \
check_iterable_type, check_length
from openmc import Materials, Material, Cell
from openmc.search import _SCALAR_BRACKETED_METHODS, search_for_keff
from openmc.data import atomic_mass, AVOGADRO, ELEMENT_SYMBOL
import openmc.lib
from openmc.mpi import comm


class Batchwise(ABC):
    """Abstract Base Class for implementing depletion batchwise classes.

    Users should instantiate:
    :class:`openmc.deplete.batchwise.BatchwiseGeom` or
    :class:`openmc.deplete.batchwise.BatchwiseMat` rather than this class.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    bracket : list of float
        Bracketing range around the guess value to search for the solution as
        list of float.
        This is equivalent to the `bracket` parameter of the `search_for_keff`.
    bracket_limit : list of float
        Upper and lower limits for the search_for_keff. If search_for_keff root
        or guesses fall above the range, the closest limit will be taken and
        set as new result.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Defaults to 'brentq'.
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0.
    print_iterations : Bool, Optional
        Wheter or not to print `search_for_keff` iterations.
        Default to True
    search_for_keff_output : Bool, Optional
        Wheter or not to print transport iterations during  `search_for_keff`.
        Default to False
    atom_density_limit : float, Optional
        If set only nuclides with atom density greater than limit are passed
        to next transport.
        Default to 0.0 atoms/b-cm
    interrupt : bool
        To be used during a restart from a batchwise simulation that was
        interrupted.
        Default to False
    Attributes
    ----------
    burn_mats : list of str
        List of burnable materials ids
    """
    def __init__(self, operator, model, bracket, bracket_limit,
                 bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, interrupt=False):

        self.operator = operator
        self.burn_mats = operator.burnable_mats
        self.local_mats = operator.local_mats
        self.model = model

        check_iterable_type('bracket', bracket, Real)
        check_length('bracket', bracket, 2)
        check_less_than('bracket values', bracket[0], bracket[1])
        self.bracket = bracket

        check_iterable_type('bracket_limit', bracket_limit, Real)
        check_length('bracket_limit', bracket_limit, 2)
        check_less_than('bracket limit values',
                         bracket_limit[0], bracket_limit[1])

        self.bracket_limit = bracket_limit

        self.bracketed_method = bracketed_method
        self.tol = tol
        self.target = target
        self.print_iterations = print_iterations
        self.search_for_keff_output = search_for_keff_output
        self.atom_density_limit = atom_density_limit
        self.interrupt = interrupt

    @property
    def bracketed_method(self):
        return self._bracketed_method

    @bracketed_method.setter
    def bracketed_method(self, value):
        check_value('bracketed_method', value, _SCALAR_BRACKETED_METHODS)
        if value != 'brentq':
            warn('brentq bracketed method is recomended here')
        self._bracketed_method = value

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        check_type("tol", value, Real)
        self._tol = value

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        check_type("target", value, Real)
        self._target = value

    @property
    def atom_density_limit(self):
        return self._atom_density_limit

    @atom_density_limit.setter
    def atom_density_limit(self, value):
        check_type("Atoms density limit", value, Real)
        if value < 0.0:
            raise ValueError(f'Cannot set negative value')
        else:
            self._atom_density_limit = value

    @abstractmethod
    def _model_builder(self, param):
        """
        Builds the parametric model to be passed to `search_for_keff`.
        Callable function which builds a model according to a passed
        parameter. This function must return an openmc.model.Model object.
        Parameters
        ----------
        param : parameter
            model function variable
        Returns
        -------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
    def _msr_search_for_keff(self, val):
        """
        Perform the criticality search for a given parametric model.
        It supports geometrical or material based `search_for_keff`.
        Parameters
        ----------
        val : float
            Previous result value
        Returns
        -------
        res : float
            Estimated value of the variable parameter where keff is the
            targeted value
        """
        _bracket = deepcopy(self.bracket)
        _tol = self.tol
        # Normalize search_for_keff tolerance with guess value
        if not -1.0 < val < 1.0:
            _tol = self.tol / abs(val)

        # Run until a search_for_keff root is found or ouf ot limits
        res = None
        while res == None:
            search = search_for_keff(self._model_builder,
                    bracket = [_bracket[0]+val, _bracket[1]+val],
                    tol = _tol,
                    bracketed_method = self.bracketed_method,
                    target = self.target,
                    print_iterations = self.print_iterations,
                    run_args = {'output': self.search_for_keff_output})

            # if len(search) is 3 search_for_keff was successful
            if len(search) == 3:
                _res, _, _ = search
                #Check if root is within bracket limits
                if self.bracket_limit[0] < _res <  self.bracket_limit[1]:
                    res = _res
                else:
                    # Set res with the closest limit and continue
                    arg_min = abs(np.array(self.bracket_limit)-_res).argmin()
                    warn('WARNING: Search_for_keff returned root out of '\
                         'bracket limit. Set root to {:.2f} and continue.'
                         .format(self.bracket_limit[arg_min]))
                    res = self.bracket_limit[arg_min]

            elif len(search) == 2:
                guess, k = search
                #Check if all guesses are within bracket limits
                if all(self.bracket_limit[0] < g < self.bracket_limit[1] \
                    for g in guess):
                    #Simple method to iteratively adapt the bracket
                    print('INFO: Function returned values below or above ' \
                           'target. Adapt bracket...')

                    # if the bracket ends up being smaller than the std of the
                    # keff's closer value to target, no need to continue-
                    if all(_k <= max(k).s for _k in k):
                        arg_min = abs(self.target-np.array(guess)).argmin()
                        res = guess[arg_min]

                    # Calculate gradient as ratio of delta bracket and delta k
                    grad = abs(np.diff(_bracket) / np.diff(k))[0].n
                    # Move the bracket closer to presumed keff root.
                    # 2 cases: both k are below or above target
                    if np.mean(k) < self.target:
                        # direction of moving bracket: +1 is up, -1 is down
                        if guess[np.argmax(k)] > guess[np.argmin(k)]:
                            dir = 1
                        else:
                            dir = -1
                        _bracket[np.argmin(k)] = _bracket[np.argmax(k)]
                        _bracket[np.argmax(k)] += grad * (self.target - \
                                                  max(k).n) * dir
                    else:
                        if guess[np.argmax(k)] > guess[np.argmin(k)]:
                            dir = -1
                        else:
                            dir = 1
                        _bracket[np.argmax(k)] = _bracket[np.argmin(k)]
                        _bracket[np.argmin(k)] += grad * (min(k).n - \
                                                  self.target) * dir

                else:
                    # Set res with closest limit and continue
                    arg_min = abs(np.array(self.bracket_limit)-guess).argmin()
                    warn('WARNING: Adaptive iterative bracket went off '\
                         'bracket limits. Set root to {:.2f} and continue.'
                         .format(self.bracket_limit[arg_min]))
                    res = self.bracket_limit[arg_min]

            else:
                raise ValueError(f'ERROR: Search_for_keff output is not valid')

        return res

    def _save_res(self, type, step_index, res):
        """
        Save results to msr_results.h5 file.
        Parameters
        ----------
        type : str
            String to characterize geometry and material results
        step_index : int
            depletion time step index
        res : float or dict
             Root of the search_for_keff function
        """
        filename = 'msr_results.h5'
        kwargs = {'mode': "a" if os.path.isfile(filename) else "w"}

        if comm.rank == 0:
            with h5py.File(filename, **kwargs) as h5:
                name = '_'.join([type, str(step_index)])
                if name in list(h5.keys()):
                    last = sorted([int(re.split('_',i)[1]) for i in h5.keys()])[-1]
                    step_index = last + 1
                h5.create_dataset('_'.join([type, str(step_index)]), data=res)

    def _update_volumes_after_depletion(self, x):
        """
        After a depletion step, both material volume and density change, due to
        transmutation reactions and continuous removal operations, if present.
        At present we lack any implementation to calculate density and volume
        changes due to different molecules speciation. Therefore, the assumption
        we make is to consider the density constant and let the material volume
        vary with the change in the nuclide concentrations.
        This method uses the nuclide concentrations coming from the previous Bateman
        solution and calculates a new volume, keeping the mass density of the material
        constant. It will then assign the volume to the AtomNumber class instance.

        Parameters
        ----------
        x : list of numpy.ndarray
            Total atom concentrations
        """
        self.operator.number.set_density(x)

        for rank in range(comm.size):
            number_i = comm.bcast(self.operator.number, root=rank)

            for i, mat in enumerate(number_i.materials):
                # Total nuclides density
                dens = 0
                vals = []
                for nuc in number_i.nuclides:
                    # total number of atoms
                    val = number_i[mat, nuc]
                    # obtain nuclide density in atoms-g/mol
                    dens +=  val * atomic_mass(nuc)
                # Get mass dens from beginning, intended to be held constant
                rho = openmc.lib.materials[int(mat)].get_density('g/cm3')
                #rho = [m.get_mass_density() for m in self.model.materials if
                #        m.id == int(mat)][0]

                #In the CA version we assign the new volume to AtomNumber
                number_i.volume[i] = dens / AVOGADRO / rho

class BatchwiseGeom(Batchwise):
    """ Batchwise geoemtrical parent class

    Instances of this class can be used to define geometrical based criticality
    actions during a transport-depletion calculation.
    Currently only translation is supported as a child class, but rotation
    can be easily added if needed.
    In this case a geometrical cell translation coefficient will be used as
    parametric variable.
    The user should remember to fill the cell with a Universe.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    cell_id_or_name : Openmc.Cell or int or str
        Identificative of parametric cell
    bracket : list of float
        Bracketing range around the guess value to search for the solution as
        list of float in cm.
        In this case the guess guess value is the translation coefficient result
        of the previous depletion step
    bracket_limit : list of float
        Upper and lower limits in cm. If search_for_keff root
        or guesses fall above the range, the closest limit will be taken and
        set as new result. In this case the limit should coincide with the
        cell geometrical boundary conditions.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Default to 'brentq'
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0
    print_iterations : bool, Optional
        Wether or not to print root finder interations
        Deafult to true
    search_for_keff_output : bool, optional
        Wether or not to print keff run output
        Default to False
    atom_density_limit : float, optional
        If set only nuclides with atom density greater than limit are passed
        to next transport.
        Default to 0.0 atoms/b-cm
    interrupt : bool, optional
        whether or not ro restart from an interrupted simulation (crashed).
        Default to False

    Attributes
    ----------
    cell_id : openmc.Cell or int or str
        Identificative of parametric cell
    """
    def __init__(self, operator, model, cell_id_or_name, bracket,
                 bracket_limit, bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, interrupt=False):

        super().__init__(operator, model, bracket, bracket_limit,
                         bracketed_method, tol, target, print_iterations,
                         search_for_keff_output, atom_density_limit, interrupt)

        self.cell_id = self._get_cell_id(cell_id_or_name)


    def _get_cell_id(self, val):
        """Helper method for getting cell id from cell instance or cell name.
        Parameters
        ----------
        val : Openmc.Cell or str or int representing Cell
        Returns
        -------
        id : str
            Cell id
        """
        if isinstance(val, Cell):
            check_value('Cell id', val.id, [cell.id for cell in \
                                self.model.geometry.get_all_cells().values()])
            val = val.id

        elif isinstance(val, str):
            if val.isnumeric():
                check_value('Cell id', val, [str(cell.id) for cell in \
                                self.model.geometry.get_all_cells().values()])
                val = int(val)
            else:
                check_value('Cell name', val, [cell.name for cell in \
                                self.model.geometry.get_all_cells().values()])

                val = [cell.id for cell in \
                    self.model.geometry.get_all_cells().values() \
                    if cell.name == val][0]

        elif isinstance(val, int):
            check_value('Cell id', val, [cell.id for cell in \
                                self.model.geometry.get_all_cells().values()])

        else:
            ValueError(f'Cell: {val} is not recognized')

        return val

    def _get_cell_attrib(self):
        """
        Get cell attribute coefficient.
        Returns
        -------
        coeff : float
            cell coefficient
        """

    def _set_cell_attrib(self, val, attrib_name):
        """
        Set cell attribute to the cell instance.
        Attributes are only applied to cells filled with a universe
        Parameters
        ----------
        var : float
            Surface coefficient to set
        geometry : openmc.model.geometry
            OpenMC geometry model
        attrib_name : str
            Currently only translation is implemented
        """

    def _update_materials(self, x):
        """
        Assign concentration vectors from Bateman solution at previous
        timestep to the in-memory model materials, after having recalculated the
        material volume.

        Parameters
        ----------
        x : list of numpy.ndarray
            Total atom concentrations
        """
        super()._update_volumes_after_depletion(x)

        for rank in range(comm.size):
            number_i = comm.bcast(self.operator.number, root=rank)

            for mat in number_i.materials:
                nuclides = []
                densities = []

                for nuc in number_i.nuclides:
                    # get atom density in atoms/b-cm
                    val = 1.0e-24 * number_i.get_atom_density(mat, nuc)
                    if nuc in self.operator.nuclides_with_data:
                        if val > self.atom_density_limit:
                            nuclides.append(nuc)
                            densities.append(val)

                #set nuclide densities to model in memory (C-API)
                openmc.lib.materials[int(mat)].set_densities(nuclides, densities)

    def _model_builder(self, param):
        """
        Builds the parametric model that is passed to the `msr_search_for_keff`
        function by setting the parametric variable to the geoemetrical cell.
        Parameters
        ----------
        param : model parametricl variable
            for examlple: cell translation coefficient
        Returns
        -------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        self._set_cell_attrib(param)
        return self.model

    def msr_search_for_keff(self, x, step_index):
        """
        Perform the criticality search on the parametric geometrical model.
        Will set the root of the `search_for_keff` function to the cell
        attribute.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        # Get cell attribute from previous iteration
        val = self._get_cell_attrib()
        check_type('Cell coeff', val, Real)

        # Update volume and concentration vectors before performing the search_for_keff
        self._update_materials(x)

        # Calculate new cell attribute
        res = super()._msr_search_for_keff(val)

        # set results value as attribute in the geometry
        self._set_cell_attrib(res)
        print('UPDATE: old value: {:.2f} cm --> ' \
              'new value: {:.2f} cm'.format(val, res))

        #Store results
        super()._save_res('geometry', step_index, res)

        return x

class BatchwiseGeomTrans(BatchwiseGeom):
    """ Batchwise geometric translation child class, inherited from parent class
    BatchwiseGeom.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    cell_id_or_name : Openmc.Cell or int or str
        Identificative of parametric cell
    axis : int
        cell translation direction axis, where 0 is 'x', 1 is 'y' and 2 'z'.
    bracket : list of float
        Bracketing range around the guess value to search for the solution as
        list of float in cm.
        In this case the guess guess value is the translation coefficient result
        of the previous depletion step
    bracket_limit : list of float
        Upper and lower limits in cm. If search_for_keff root
        or guesses fall above the range, the closest limit will be taken and
        set as new result. In this case the limit should coincide with the
        cell geometrical boundary conditions.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Default to 'brentq'
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0
    Attributes
    ----------
    cell_id : openmc.Cell or int or str
        Identificative of parametric cell
    axis : int
        cell translation direction axis, where 0 is 'x', 1 is 'y' and 2 'z'.
    vector : numpy.array
        translation vector
    """
    def __init__(self, operator, model, cell_id_or_name, axis, bracket,
                 bracket_limit, bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, interrupt=False):

        super().__init__(operator, model, cell_id_or_name, bracket, bracket_limit,
                         bracketed_method, tol, target, print_iterations,
                         search_for_keff_output, atom_density_limit, interrupt)

        #index of cell translation direction axis
        check_value('axis', axis, [0,1,2])
        self.axis = axis

        # Initialize translation vector
        self.vector = np.zeros(3)

    def _get_cell_attrib(self):
        """
        Get cell translation coefficient.
        The translation is only applied to cells filled with a universe
        Returns
        -------
        coeff : float
            cell coefficient
        """
        for cell in openmc.lib.cells.values():
            if cell.id == self.cell_id:
                return cell.translation[self.axis]

    def _set_cell_attrib(self, val, attrib_name='translation'):
        """
        Set translation coeff to the cell in memeory.
        The translation is only applied to cells filled with a universe
        Parameters
        ----------
        var : float
            Surface coefficient to set
        geometry : openmc.model.geometry
            OpenMC geometry model
        attrib_name : str
            Currently only translation is implemented
            Default to 'translation'
        """
        self.vector[self.axis] = val
        for cell in openmc.lib.cells.values():
            if cell.id == self.cell_id or cell.name == self.cell_id:
                setattr(cell, attrib_name, self.vector)

class BatchwiseMat(Batchwise):
    """ Batchwise material class parent class.

    Instances of this class can be used to define material based criticality
    actions during a transport-depletion calculation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    mats_id_or_name : openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        nuclides and values fractions.
        E.g., mat_vector = {'U235':0.3,'U238':0.7}
    bracket : list of float
        Bracketing range quantity of material to add in grams.
    bracket_limit : list of float
        Upper and lower limits of material to add in grams.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Default to 'brentq'
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0

    Attributes
    ----------
    mats_id_or_name : List of openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        the nuclides str and values are the composition fractions.
    """

    def __init__(self, operator, model, mats_id_or_name, mat_vector, bracket,
                 bracket_limit, bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, restart_level=0.0, interrupt=False):

        super().__init__(operator, model, bracket, bracket_limit,
                         bracketed_method, tol, target, print_iterations,
                         search_for_keff_output, atom_density_limit, interrupt)

        self.mats_id = [self._get_mat_id(i) for i in mats_id_or_name]

        check_type("material vector", mat_vector, dict, str)
        for nuc in mat_vector.keys():
            check_value("check nuclide exists", nuc, self.operator.nuclides_with_data)

        if round(sum(mat_vector.values()), 2) != 1.0:
            raise ValueError('Refuel vector fractions {} do not sum up to 1.0'
                             .format(mat_vector.values()))
        self.mat_vector = mat_vector

        if not isinstance(restart_level, (float, int)):
            raise ValueError(f'{restart_level} is of type {type(restart_level)},'
                             ' while it should be int or float')
        else:
            self.restart_level = restart_level

    def _check_nuclides(self, nucs):
        """Checks if nuclides vector exist in the parametric material.
        Parameters
        ----------
        nucs : list of str
            nuclides vector
        """
        for nuc in nucs:
            check_value("check nuclids exists in mat", nuc,
                [mat.nuclides for mat_id, mat in openmc.lib.materials.items() \
                if mat_id in self.mats_id][0])

    def _get_mat_id(self, val):
        """Helper method for getting material id from Material instance or
        material name.
        Parameters
        ----------
        val : Openmc,Material or str or int representing material name/id
        Returns
        -------
        id : str
            Material id
        """
        if isinstance(val, Material):
            check_value('Material id', str(val.id), self.burn_mats)
            val = val.id

        elif isinstance(val, str):
            if val.isnumeric():
                check_value('Material id', val, self.burn_mats)
                val = int(val)
            else:
                check_value('Material name', val,
                   [mat.name for mat in self.model.materials if mat.depletable])
                val = [mat.id for mat in self.model.materials \
                        if mat.name == val][0]

        elif isinstance(val, int):
            check_value('Material id', str(val), self.burn_mats)

        return val

    def _model_builder(self, param):
        """
        Builds the parametric model that is passed to the `msr_search_for_keff`
        function by updating the material densities and setting the parametric
        variable as function of the nuclides vector. Since this is a paramteric
        material addition (or removal), we can parametrize the volume as well.
        Parameters
        ----------
        param :
            Model material function variable
        Returns
        -------
        _model :  openmc.model.Model
            Openmc parametric model
        """

    def _update_x_vector_and_volumes(self, x, res):
        """
        Updates and returns the total atoms concentrations vector with the root
        from the `search_for_keff`. It also calculates the new total volume
        in cc from the nuclides atom densities, assuming constant material mass
        density. The volume is then passed to the `openmc.deplete.AtomNumber`
        class to renormalize the atoms vector in the model in memory before
        running the next transport solver.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        res : float
            Root of the search_for_keff function
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """

    def msr_search_for_keff(self, x, step_index):
        """
        Perform the criticality search on the parametric material model.
        Will set the root of the `search_for_keff` function to the atoms
        concentrations vector.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        # Update AtomNumber with new conc vectors. Materials are updated
        # when building the model for the search_for_keff

        super()._update_volumes_after_depletion(x)

        self._check_nuclides(self.mat_vector.keys())

        # Solve search_for_keff and find new value
        res = super()._msr_search_for_keff(0)
        print('UPDATE: material new value --> {:.2f} --> '.format(res))

        #Update concentration vector and volumes with new value
        x = self._update_x_vector_and_volumes(x, res)

        #Store results
        super()._save_res('material', step_index, res)
        super()._save_res('geometry', step_index, self.restart_level)
        return  x

class BatchwiseMatRefuel(BatchwiseMat):
    """
    Batchwise material refuel child class, inherited from BatchwiseMat parent
    class.

    Instances of this class can be used to define material based criticality
    actions during a transport-depletion calculation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    mats_id_or_name : openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        nuclides and values fractions.
        E.g., mat_vector = {'U235':0.3,'U238':0.7}
    bracket : list of float
        Bracketing range quantity of material to add in grams.
    bracket_limit : list of float
        Upper and lower limits of material to add in grams.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Default to 'brentq'
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0

    Attributes
    ----------
    mats_id_or_name : List of openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        the nuclides str and values are the composition fractions.
    """

    def __init__(self, operator, model, mats_id_or_name, mat_vector, bracket,
                 bracket_limit, bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, restart_level=0.0, interrupt=False):

        super().__init__(operator, model, mats_id_or_name, mat_vector, bracket, bracket_limit,
                         bracketed_method, tol, target, print_iterations,
                         search_for_keff_output, atom_density_limit,
                         restart_level, interrupt)

    def _model_builder(self, param):
        """
        Builds the parametric model that is passed to the `msr_search_for_keff`
        function by updating the material densities and setting the parametric
        variable to the material nuclides to add. Here we fix the total number
        of atoms per material and try to conserve this quantity. Both the
        material volume and density are let free to vary.
        Parameters
        ----------
        param :
            Model function variable, fraction of total atoms to dilute
        Returns
        -------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        for rank in range(comm.size):
            number_i = comm.bcast(self.operator.number, root=rank)
            #for i, mat in enumerate(self.burn_mats):
            for i,mat in enumerate(number_i.materials):
                nuclides = []
                densities = []

                if int(mat) in self.mats_id:
                    # parametrize volume, keeping mass density constant
                    vol = number_i.volume[i] + param / \
                        [m.get_mass_density() for m in self.model.materials
                            if m.id == int(mat)][0]
                        #openmc.lib.materials[int(mat)].get_density('g/cm3')

                    for nuc in number_i.nuclides:
                        #assuming nuclides in material vector are always present
                        # in cross sections data
                        if nuc in self.mat_vector:
                            # units [#atoms/cm-b]
                            val = 1.0e-24 * number_i.get_atom_density(mat,nuc)
                            # parametrize concentration
                            # need to convert params [grams] into [#atoms/cm-b]
                            val += 1.0e-24 * param / atomic_mass(nuc) * \
                                   AVOGADRO * self.mat_vector[nuc] / vol

                            if val > self.atom_density_limit:
                                nuclides.append(nuc)
                                densities.append(val)
                        else:
                            if nuc in self.operator.nuclides_with_data:
                                # get normalized atoms density in [atoms/b-cm]
                                val = 1.0e-24 * number_i.get_atom_density(mat,
                                      nuc) * number_i.volume[i] / vol

                                if val > self.atom_density_limit:
                                    nuclides.append(nuc)
                                    densities.append(val)

                else:
                    # for all other materials, still check atom density limits
                    for nuc in number_i.nuclides:
                        if nuc in self.operator.nuclides_with_data:
                            # get normalized atoms density in [atoms/b-cm]
                            val = 1.0e-24 * number_i.get_atom_density(mat, nuc)
                            if val > self.atom_density_limit:
                                nuclides.append(nuc)
                                densities.append(val)

                #set nuclides and densities to the in-memory model
                openmc.lib.materials[int(mat)].set_densities(nuclides, densities)

        # alwyas need to return a model
        return self.model

    def _update_x_vector_and_volumes(self, x, res):
        """
        Updates and returns the total atoms concentrations vector with the root
        from the `search_for_keff`. It also calculates the new total volume
        in cc from the nuclides atom densities, assuming constant material mass
        density. The volume is then passed to the `openmc.deplete.AtomNumber`
        class to renormalize the atoms vector in the model in memory before
        running the next transport solver.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        res : float
            Root of the search_for_keff function
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        # Don't broadcast here
        number_i = self.operator.number

        for mat_id in self.mats_id:
            if str(mat_id) in self.local_mats:
                mat_idx = self.local_mats.index(str(mat_id))
                old_vol = number_i.volume[mat_idx]

                # update volume, keeping mass density constant
                number_i.volume[mat_idx] += res / \
                    [m.get_mass_density() for m in self.model.materials
                     if m.id == mat_id][0]
                    #openmc.lib.materials[mat_id].get_density('g/cm3')

                # update all concentration data with the new updated volumes
                for nuc, dens in zip(openmc.lib.materials[mat_id].nuclides,
                                     openmc.lib.materials[mat_id].densities):

                    if nuc in number_i.burnable_nuclides:
                        nuc_idx = number_i.burnable_nuclides.index(nuc)
                        # convert [#atoms/b-cm] into [#atoms]
                        x[mat_idx][nuc_idx] = dens / 1.0e-24 * \
                                            number_i.volume[mat_idx]
                    # when the nuclide is not in depletion chain update the AtomNumber
                    else:
                        #Atom density needs to be in [#atoms/cm3]
                        number_i.set_atom_density(mat_idx, nuc,
                                                              dens / 1.0e-24)

                # Normalize nuclides in x vector without cross section data
                for nuc in number_i.burnable_nuclides:
                    if nuc not in self.operator.nuclides_with_data:
                        nuc_idx = number_i.burnable_nuclides.index(nuc)
                        # normalzie with new volume
                        x[mat_idx][nuc_idx] *= old_vol / number_i.volume[mat_idx]

        return x

class BatchwiseMatDilute(BatchwiseMat):
    """
    Batchwise material dilution child class, inherited from BatchwiseMat parent
    class.

    Instances of this class can be used to define material based criticality
    actions during a transport-depletion calculation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    mats_id_or_name : openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        nuclides and values fractions.
        E.g., mat_vector = {'U235':0.3,'U238':0.7}
    bracket : list of float
        Bracketing range quantity of material to add in grams.
    bracket_limit : list of float
        Upper and lower limits of material to add in grams.
    bracketed_method : {'brentq', 'brenth', 'ridder', 'bisect'}, optional
        Solution method to use.
        This is equivalent to the `bracket_method` parameter of the
        `search_for_keff`.
        Default to 'brentq'
    tol : float
        Tolerance for search_for_keff method.
        This is equivalent to the `tol` parameter of the `search_for_keff`.
        Default to 0.01
    target : Real, optional
        This is equivalent to the `target` parameter of the `search_for_keff`.
        Default to 1.0

    Attributes
    ----------
    mats_id_or_name : List of openmc.Material or int or str
        Identificative of parametric material
    mat_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        the nuclides str and values are the composition fractions.

    """
    def __init__(self, operator, model, mats_id_or_name, mat_vector, bracket,
                 bracket_limit, bracketed_method='brentq', tol=0.01, target=1.0,
                 print_iterations=True, search_for_keff_output=False,
                 atom_density_limit=0.0, restart_level=0.0, interrupt=False):

        super().__init__(operator, model, mats_id_or_name, mat_vector, bracket,
                         bracket_limit, bracketed_method, tol, target, print_iterations,
                         search_for_keff_output, atom_density_limit,
                         restart_level, interrupt)

    def _model_builder(self, param):
        """
        Builds the parametric model that is passed to the `msr_search_for_keff`
        function by updating the material densities and setting the parametric
        variable to the material nuclides to add. Here we fix the total number
        of atoms per material and try to conserve this quantity. Both the
        material volume and density are let free to vary.
        Parameters
        ----------
        param :
            Model function variable, fraction of total atoms to dilute
        Returns
        -------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        for rank in range(comm.size):
            number_i = comm.bcast(self.operator.number, root=rank)

            for i, mat in enumerate(number_i.materials):
                nuclides = []
                densities = []

                if int(mat) in self.mats_id:
                    # Sum all atoms present in [#atoms/b-cm]
                    tot_atoms = 1.0e-24 * sum(number_i.number[i]) / number_i.volume[i]

                    for nuc in number_i.nuclides:
                        # Dilute nuclides with cross sections (with data)
                        if nuc in self.operator.nuclides_with_data:
                            # [#atoms/b-cm]
                            val = 1.0e-24 * number_i.get_atom_density(mat,nuc)
                            # Build parametric function, where param is the
                            # dilute fraction to replace.
                            # it assumes all nuclides in material vector have
                            # cross sections data
                            if nuc in self.mat_vector:
                                val = (1-param) * val + param * \
                                      self.mat_vector[nuc] * tot_atoms
                            else:
                                val *= (1-param)

                            #just making sure we are not adding any negative values
                            if val > self.atom_density_limit:
                                nuclides.append(nuc)
                                densities.append(val)

                # For all other materials, still check density limit
                else:
                    for nuc in number_i.nuclides:
                        if nuc in self.operator.nuclides_with_data:
                            # get atoms density [atoms/b-cm]
                            val = 1.0e-24 * number_i.get_atom_density(mat, nuc)
                            if val > self.atom_density_limit:
                                nuclides.append(nuc)
                                densities.append(val)

                openmc.lib.materials[int(mat)].set_densities(nuclides, densities)

        return self.model

    def _update_x_vector_and_volumes(self, x, res):
        """
        Updates and returns the total atoms concentrations vector with the root
        from the `search_for_keff`. No volume update is computed here.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        res : float
            Root of the search_for_keff function
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        # Don't broadcast here
        number_i = self.operator.number

        for mat_id in self.mats_id:
            if str(mat_id) in self.local_mats:
                mat_idx = self.local_mats.index(str(mat_id))

                for nuc, dens in zip(openmc.lib.materials[mat_id].nuclides,
                                     openmc.lib.materials[mat_id].densities):
                    if nuc in number_i.burnable_nuclides:
                        nuc_idx = number_i.burnable_nuclides.index(nuc)
                        # convert [#atoms/b-cm] into [#atoms]
                        x[mat_idx][nuc_idx] = dens / 1.0e-24 * \
                                              number_i.volume[mat_idx]
                    else:
                        #Atom density needs to be in [#atoms/cm3]
                        number_i.set_atom_density(mat_idx, nuc, dens / 1.0e-24)

                for nuc in number_i.burnable_nuclides:
                    if nuc not in self.operator.nuclides_with_data:
                        nuc_idx = number_i.burnable_nuclides.index(nuc)
                        x[mat_idx][nuc_idx] *= (1-res)

        return x

class BatchwiseWrap1():
    """
    Batchwise wrapper class, it wraps BatchwiseGeom and BatchwiseMat instances,
    with some user defined logic.

    This class should probably not be defined here, but we can keep it now for
    convenience

    The loop logic of this wrapper class is the following:

    1. Run BatchwiseGeom and return geometrical coefficient
    2. check if geometrical coeff hits bracket upper geometrical limit
    3.1 if not, update geometry
    3.2 if yes, set geometrical coefficient to user-defined restart level and
    run BatchwiseMat

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    bw_geom : BatchwiseGeom
        openmc.deplete.batchwise.BatchwiseGeom object
    bw_mat : BatchwiseMat
        openmc.deplete.batchwise.BatchwiseMat object
    """

    def __init__(self, bw_geom, bw_mat=None, interrupt=False):

        if not isinstance(bw_geom, BatchwiseGeom):
            raise ValueError(f'{bw_geom} is not a valid instance of'
                              ' BatchwiseGeom class')
        else:
            self.bw_geom = bw_geom

        if not isinstance(bw_mat, BatchwiseMat):
            raise ValueError(f'{bw_mat} is not a valid instance of'
                              ' BatchwiseMat class')
        else:
            self.bw_mat = bw_mat

        self.interrupt = interrupt

    def _update_volumes_after_depletion(self, x):
        """
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atom concentrations
        """
        self.bw_geom._update_volumes_after_depletion(x)

    def msr_search_for_keff(self, x, step_index):
        """
        Perform the criticality search on the parametric material model.
        Will set the root of the `search_for_keff` function to the atoms
        concentrations vector.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        #Start by doing a geometrical parametrization
        x = self.bw_geom.msr_search_for_keff(x, step_index)
        #check if upper geometrical limit gets hit
        if self.bw_geom._get_cell_attrib() >= self.bw_geom.bracket_limit[1]:
            # Restart level and add material
            if self.bw_mat is not None:
                self.bw_geom._set_cell_attrib(self.bw_mat.restart_level)
                x = self.bw_mat.msr_search_for_keff(x, step_index)
            # in case not material parametrization is defined, touch and exit-
            else:
                from pathlib import Path
                print(f'Reached maximum of {self.bw_geom.bracket_limit[1]} cm'
                       ' exit..')
                Path('sim.done').touch()
                exit()
        return x

class BatchwiseWrap2():
    """
    Batchwise wrapper class, it wraps BatchwiseGeom and BatchwiseMat instances,
    with some user defined logic.

    This class should probably not be defined here, but we can keep it now for
    convenience

    The loop logic of this wrapper class is the following:

    1. Run BatchwiseGeom and return geometrical coefficient
    2. check if step index equals user definde dilute interval
    3.1 if not, update geometry
    3.2 if yes, set geometrical coefficient to user-defined restart level and
    run BatchwiseMatDilute

    In this case if the bracket upper geometrical limit is hitted,
    simply stop the simulation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    bw_geom : BatchwiseGeom
        openmc.deplete.batchwise.BatchwiseGeom object
    bw_mat : BatchwiseMat
        openmc.deplete.batchwise.BatchwiseMat object
    dilute_interval : int
        Frequency of dilution in number of timesteps
    first_dilute : int or None
        Timestep index for first dilution, to be used during restart simulation
        Default to None
    """

    def __init__(self, bw_geom, bw_mat, dilute_interval, first_dilute=None,
                 interrupt=False):

        if not isinstance(bw_geom, BatchwiseGeom):
            raise ValueError(f'{bw_geom} is not a valid instance of'
                              ' BatchwiseGeom class')
        else:
            self.bw_geom = bw_geom

        if not isinstance(bw_mat, BatchwiseMat):
            raise ValueError(f'{bw_mat} is not a valid instance of'
                              ' BatchwiseMat class')
        else:
            self.bw_mat = bw_mat

        #TODO check these values
        self.first_dilute = first_dilute
        self.step_interval = dilute_interval

        # if first dilute is set, the dilute interval needs to be updated
        if self.first_dilute is not None:
            self.dilute_interval = dilute_interval + self.first_dilute
        else:
            self.dilute_interval = dilute_interval
        self.interrupt = interrupt

    def _update_volumes_after_depletion(self, x):
        """
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atom concentrations
        """
        self.bw_geom._update_volumes_after_depletion(x)

    def msr_search_for_keff(self, x, step_index):
        """
        Perform the criticality search on the parametric material model.
        Will set the root of the `search_for_keff` function to the atoms
        concentrations vector.
        Parameters
        ----------
        x : list of numpy.ndarray
            Total atoms concentrations
        Returns
        -------
        x : list of numpy.ndarray
            Updated total atoms concentrations
        """
        #Check if index lies in dilution timesteps
        if step_index in [self.first_dilute, self.dilute_interval]:
            # restart level and perform dilution
            self.bw_geom._set_cell_attrib(self.bw_mat.restart_level)
            x = self.bw_mat.msr_search_for_keff(x, step_index)
            #update dulution interval
            if step_index == self.dilute_interval:
                self.dilute_interval += self.step_interval

        else:
            x = self.bw_geom.msr_search_for_keff(x, step_index)
            # in this case if upper limit gets hit, stop directly
            if self.bw_geom._get_cell_attrib() >= self.bw_geom.bracket_limit[1]:
                from pathlib import Path
                print(f'Reached maximum of {self.bw_geom.bracket_limit[1]} cm'
                       ' exit..')
        return x
