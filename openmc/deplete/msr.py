from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np

from openmc import Materials, Material
from openmc.search import search_for_keff
from openmc.data import atomic_mass, AVOGADRO
from openmc.lib import init_geom

class MsrContinuous:
    """Class defining Molten salt reactor (msr) elements (fission products)
    removal, based on removal rates and cycle time concepts.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.
    Parameters
    ----------
    local_mats : openmc.Material
        openmc.Material

    Attributes
    ----------
    local_mats : openmc.Material
        All local material defined in the model.
    burn_mats : list of str
        List of depletable material id
    ordr_burn : OrderedDict of int and int
        OrderedDict of depletable material id and enuemerated indeces
    removal_rates : OrderedDict of str and OrderedDict
        OrderedDict of depletable material id and OrderedDict to fill
    """

    def __init__(self, local_mats):

        if not isinstance(local_mats, Materials):
            raise ValueError(f'{local_mats} is not a valid openmc Material')
        else:
            self.local_mats = local_mats

        self.burn_mats = [mat.id for mat in self.local_mats if mat.depletable]
        self.index_msr = [(i, i) for i in range(self.n_burn)]

        self.ord_burn = self._order_burn_mats()
        self.removal_rates = self._initialize_removal_rates()

    @property
    def n_burn(self):
        return len(self.burn_mats)

    def _get_mat_index(self, mat):
        """Helper method for getting material index"""
        if isinstance(mat, Material):
            mat = str(mat.id)
        return self.burn_mats[mat] if isinstance(mat, str) else mat

    def _order_burn_mats(self):
        """Order depletable material id
        Returns
        ----------
        OrderedDict of int and int
            OrderedDict of depletable material id and enuemrated indeces
        """
        return OrderedDict((int(id), i) for i, id in enumerate(self.burn_mats))

    def _initialize_removal_rates(self):
        """Initialize removal rates container
        Returns
        ----------
        OrderedDict of str and OrderedDict
            OrderedDict of depletable material id and OrderedDict to fill
        """
        return OrderedDict((id, OrderedDict()) for id in self.burn_mats)

    def transfer_matrix_index(self):
        """Get transfer matrices indeces
        Returns
        ----------
        list of tuples :
            List of tuples pairs to order the transfer matrices
            when building the coupled matrix
        """
        transfer_index = OrderedDict()
        for id, val in self.removal_rates.items():
            if val:
                for elm, [tr, mat] in val.items():
                    if mat is not None:
                        j = self.ord_burn[id]
                        i = self.ord_burn[mat]
                        transfer_index[(i,j)] = None
        return list(transfer_index.keys())

    def get_removal_rate(self, mat, element):
        """Extract removal rates
        Parameters
        ----------
        mat : Openmc,Material or int
            Depletable material
        element : str
            Element to extract removal rate value
        Returns:
        ----------
        removal_rate : float
            Removal rate value
        """
        mat = self._get_mat_index(mat)
        return self.removal_rates[mat][element][0]

    def get_destination_mat(self, mat, element):
        """Extract destination material
        Parameters
        ----------
        mat : Openmc,Material or int
            Depletable material
        element : str
            Element to extract removal rate value
        Returns:
        ----------
        destination_mat : str
            Depletable material id to where elements get transferred
        """
        mat = self._get_mat_index(mat)
        if element in self.removal_rates[mat]:
            return self.removal_rates[mat][element][1]


    def get_elements(self, mat):
        """Extract removing elements for a given material
        Parameters
        ----------
        mat : Openmc,Material or int
            Depletable material
        Returns:
        ----------
        elements : list
            List of elements
        """
        mat = self._get_mat_index(mat)
        elements=[]
        for k, v in self.removal_rates.items():
            if k == mat:
                for elm, _ in v.items():
                    elements.append(elm)
        return elements

    def set_removal_rate(self, mat, elements, removal_rate, dest_mat=None,
                         units='1/s'):
        """Set removal rates to depletable material
        Parameters
        ----------
        mat : Openmc,Material or int
            Depletable material to where add removal rates
        elements : list[str]
            List of strings of elements
        removal_rate : float
            Removal rate coefficient
        dest_mat : Openmc,Material or int, Optional
            Destination material if transfer or elements tracking is wanted
        units: str, optional
            Removal rates units (not set yet)
            Default : '1/s'
        """
        mat = self._get_mat_index(mat)
        if dest_mat is not None:
            dest_mat = self._get_mat_index(dest_mat)
        for element in elements:
            self.removal_rates[mat][element] = [removal_rate, dest_mat]


class MsrBatchwise(ABC):
    """Abstract Base Class for implementing msr batchwise classes.

    Users should instantiate
    :class:`openmc.deplete.msr.MsrBatchwiseGeom` or
    :class:`openmc.deplete.msr.MsrBatchwiseMat` rather than this class.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    range : list of float
        List of floats defining bracketed range for search_for_keff
    bracketed_method : string
        Bracketed method for search_for_keff
        Default to 'brentq' --> more robuts
    tol : float
        Tolerance for search_for_keff method
        Default to 0.01
    target : float
        Search_for_keff function target
        Default to 1.0
    atom_density_limit : float, optional
        Atom density limit below which nuclides are excluded from
        search_for_keff, to be used for speed up.
        Default to 0.0
    Attributes

    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    burn_mats : list of str
        List of burnable materials ids
    burn_nucs : list of str
        List of nuclides with available data for burnup
    model : openmc.model.Model
        OpenMC model object
        range : list of float
        list of floats defining bracketed range for search_for_keff
    bracketed_method : string
        Bracketed method for search_for_keff
    tol : float
        Tolerance for search_for_keff method
    target : float
        Search_for_keff function target
    atom_density_limit : float
        Atom density limit below which nuclides are excluded from
        search_for_keff, to be used for speed up.
    """
    def __init__(self, operator, model, range=None, bracketed_method='brentq',
                 tol=0.01, target=1.0, atom_density_limit=0.0):

        self.operator = operator
        self.burn_mats = operator.burnable_mats

        self.model = model

        self.range = range
        self.bracketed_method = bracketed_method
        self.tol = tol
        self.target = target
        self.atom_density_limit = atom_density_limit

    def _order_vector(self, x):
        """
        Order concentrations vector into a list of nuclides ordered
        dictionaries for each depletable material for easier extraction.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function.
        Returns
        ------------
        ord_x : list of OrderedDict
            List of OrderedDict of nuclides for each depletable material
        """
        ord_x = list()
        for i, mat in enumerate(self.burn_mats):
            nucs = OrderedDict()
            for j, nuc in enumerate(self.operator.number.burnable_nuclides):
                nucs[nuc] = x[i][j]
            ord_x.append(nucs)
        return ord_x

    @abstractmethod
    def msr_criticality_search(self, x):
        """
        Perform the criticality search for a given parametric model.
        It can either be a geometrical or material search.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function for normalization
        Returns
        ------------
        res : float
            root of search_for_keff function
        diff : float
            difference from previous result
        """
        pass

    @abstractmethod
    def _build_parametric_model(self, param):
        """
        Builds the parametric model to be passed to `msr_criticality_search`
        Parameters
        ------------
        param :
            model function variable
        Returns
        ------------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        pass

    def finalize(self, var):
        """
        Finilize the geometry by setting the search_for_keff function root to
        the geometry model defined in memory
        Parameters
        ------------
        var : float
            geometrical variable to re-initialize geometry in memory
        """
        pass

class MsrBatchwiseGeom(MsrBatchwise):
    """ MsrBatchwise geoemtrical class

    Instances of this class can be used to define geometry-based  criticality
    actions during a transport-depletion calculation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    geom_id : int
        Parametric goemetrical feature id from the geometry model
    type : string, optional
        Parametric goemetrical feature type. Right now only 'surface' is
        supported
        Default to 'surface'
    range : list of float
        List of floats defining bracketed range for search_for_keff
    bracketed_method : string
        Bracketed method for search_for_keff
        Default to 'brentq' --> more robuts
    tol : float
        Tolerance for search_for_keff method
        Default to 0.01
    target : float
        Search_for_keff function target
        Default to 1.0
    atom_density_limit : float, optional
        Atom density limit below which nuclides are excluded from
        search_for_keff, to be used for speed up.
        Default to 0.0
    refine_search : bool, optional
        Whether or not to include search_for_keff statistical refinement
        when close to a geometrical upper limit.
        Default to True
    refuel : bool, Optional
        Whether or not to include a refule, whenever the upper limit gets hit
        by the geometrical search_for_keff. A call from the abc class will
        Default to False
    start_param : float, optional
        Restarting geometrical variable coefficient after a refuel event.
        Deafult to 0.0

    Attributes
    ----------
    geom_id : int
        Parametric goemetrical feature id from the geometry model
    type : string
        Parametric goemetrical feature type. Right now only 'surface' is
        supported
    refine_search : bool
        Whether or not to include search_for_keff statistical refinement
        when close to a geometrical upper limit.
    refuel : bool
        Whether or not to include a refule, whenever the upper limit gets hit
        by the geometrical search_for_keff. A call from the abc class will
    start_param : float
        Restarting geometrical variable coefficient after a refuel event.

    """
    def __init__(self, operator, model, geom_id, type='surface', range=None,
                    bracketed_method='brentq', tol=0.01, target=1.0,
                    atom_density_limit=0.0, refine_search=True, refuel=True,
                    start_param=0.0):

        super().__init__(operator, model, range, bracketed_method, tol, target,
                         atom_density_limit)

        self.geom_id = geom_id
        if len(range) != 3:
            raise ValueError("Range: {range} lenght is not 3. Must provide"
                             "bracketed range and upper limit")

        self.refine_search = refine_search
        self.refuel = refuel
        self.start_param = start_param

    def _extract_geom_coeff(self):
        """
        Extract surface coefficient from surface id. Right now only
        surfaces are supported, but this could be extended to other geometrical
        features.
        Returns
        ------------
        coeff : float
            surface coefficient
        """
        for surf in self.model.geometry.get_all_surfaces().items():
            if surf[1].id == self.geom_id:
                keys = list(surf[1].coefficients.keys())
                if len(keys) == 1:
                    coeff = getattr(surf[1], keys[0])
                else:
                    msg=(f'Surface coefficients {keys} are more than one')
                    raise Exception(msg)
        return coeff

    def _set_geom_coeff(self, var, geometry):
        """
        Set surface coefficient
        Parameters
        ------------
        var : float
            Surface coefficient to set
        geometry : openmc.model.geometry
            OpenMC geometry model
        """
        for surf in geometry.get_all_surfaces().items():
            if surf[1].id == self.geom_id:
                keys = list(surf[1].coefficients.keys())
                if len(keys) == 1:
                    setattr(surf[1], keys[0], var)
                else:
                    msg = (f'Surface coefficients {keys} are more than one')
                    raise Exception(msg)

    def _normalize_nuclides(self, x):
        """
        Export a meterial xml to be red by the search_for_keff function
        based on available crosse section nuclides with data and
        `atom_density_limit` argument, if defined. This function also calculates
        the new total volume in [cc] from nuclides atom densities, assuming
        constant material density as set by the user in the initial material
        definition.
        The new volume is then passed to the `openmc.deplete.AtomNumber` class
        to renormalize the atoms vector in the model in memory before running
        the next transport iteration.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function.
        """
        _materials = deepcopy(self.model.materials)
        nucs = super()._order_vector(x)

        for i, mat in enumerate(self.burn_mats):

            atoms_gram_per_mol = 0
            for nuc, val in nucs[i].items():
                if val < self.atom_density_limit or \
                nuc not in self.operator.nuclides_with_data:
                    _materials[int(mat)-1].remove_nuclide(nuc)
                else:
                    _materials[int(mat)-1].remove_nuclide(nuc)
                    _materials[int(mat)-1].add_nuclide(nuc, val)
                atoms_gram_per_mol += val * atomic_mass(nuc)

            #Assign new volume to AtomNumber
            self.operator.number.volume[i] = atoms_gram_per_mol / AVOGADRO /\
                                    _materials[int(mat)-1].get_mass_density()
        _materials.export_to_xml()

    def _build_parametric_model(self, param):
        """
        Builds the parametric model to be passed to `msr_criticality_search`
        Parameters
        ------------
        param :
            model function variable: geometrical coefficient
        Returns
        ------------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        _model = deepcopy(self.model)
        self._set_geom_coeff(param, _model.geometry)
        _model.geometry.export_to_xml()
        return _model

    def _finalize(self, var):
        """
        Finilize the geometry by setting the search_for_keff function root to
        the geometry model defined in memory
        Parameters
        ------------
        var : float
            geometry coefficient
        """
        self._set_geom_coeff(var, self.model.geometry)
        self.model.geometry.export_to_xml()
        init_geom()

    def msr_criticality_search(self, x):
        """
        Perform the criticality search on the parametric geometrical model.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function for normalization
        Returns
        ------------
        res : float
            geometrical coefficient root of search_for_keff function
        diff : float
            difference from previous result
        """
        self._normalize_nuclides(x)
        coeff = self._extract_geom_coeff()

        low_range = self.range[0]
        up_range = self.range[1]

        if -1.0 < coeff < 1.0:
            _tol = self.tol / 2
        else:
            _tol = self.tol / abs(coeff)

        res = None
        while res == None:

            if self.refine_search and coeff >= abs(self.range[2])/2:
                check_brackets = True
            else:
                check_brackets = False

            search = search_for_keff(self._build_parametric_model,
                    bracket=[coeff + low_range, coeff + up_range],
                    tol=_tol,
                    bracketed_method=self.bracketed_method,
                    target=self.target,
                    print_iterations=True,
                    run_args={'output':False},
                    check_brackets=check_brackets)

            if len(search) == 3:
                _res, _, _ = search
                # Further check, in case upper limit gets hit
                if _res <= self.range[2]:
                    res = _res
                else:
                    if self.refuel:
                        msg = 'INFO: Hit upper limit {:.2f} cm. Update geom' \
                        'coeff to {:.2f} and start refuel'.format(self.range[2],
                                                               self.start_param)
                        print(msg)
                        res = self.start_param
                        break
                    else:
                        msg = 'STOP: Hit upper limit and no further criteria' \
                               'defined'
                        raise Exception(msg)

            elif len(search) == 2:
                guesses, k = search

                if guesses[-1] > self.range[2]:
                    if self.refuel:
                        msg = 'INFO: Hit upper limit {:.2f} cm. Update geom' \
                        'coeff to {:.2f} and start refuel'.format(self.range[2],
                                                               self.start_param)
                        print(msg)
                        res = self.start_param
                        break
                    else:
                        msg = 'STOP: Hit upper limit and no further criteria ' \
                               'defined'
                        raise Exception(msg)

                if np.array(k).prod() < self.target:
                    print ('INFO: Function returned values BELOW target, ' \
                           'adapting bracket range...')
                    if (self.target - np.array(k).prod()) <= 0.02:
                        low_range = up_range - 3
                    elif 0.02 < (self.target - np.array(k).prod()) < 0.03:
                        low_range = up_range - 1
                    else:
                        low_range = up_range - 0.5
                    up_range += abs(self.range[1])/2

                else:
                    print ('INFO: Function returned values ABOVE target, ' \
                           'adapting bracket range...')
                    up_range = low_range + 2
                    low_range -= abs(self.range[0])

            else:
                raise ValueError(f'ERROR: search_for_keff output not valid')

        self._finalize(res)
        msg = 'UPDATE: old coeff: {:.2f} cm --> ' \
              'new coeff: {:.2f} cm'.format(coeff, res)
        print(msg)
        diff = res - coeff
        return res, diff

class MsrBatchwiseMat(MsrBatchwise):
    """ MsrBatchwise material class

    Instances of this class can be used to define material-based criticality
    actions during a transport-depletion calculation.

    An instance of this class can be passed directly to an instance of the
    integrator class, such as :class:`openmc.deplete.CECMIntegrator`.

    Parameters
    ----------
    operator : openmc.deplete.Operator
        OpenMC operator object
    model : openmc.model.Model
        OpenMC model object
    mat_id : int
        Material id to perform actions on
    refuel_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        the nuclides and values the fractions
    range : list of float
        List of floats defining bracketed range for search_for_keff
    bracketed_method : string
        Bracketed method for search_for_keff
        Default to 'brentq' --> more robuts
    tol : float
        Tolerance for search_for_keff method
        Default to 0.01
    target : float
        Search_for_keff function target
        Default to 1.0
    atom_density_limit : float, optional
        Atom density limit below which nuclides are excluded from
        search_for_keff, to be used for speed up.
        Default to 0.0

    Attributes
    ----------
    mat_id : int
        Material id to perform actions on
    refuel_vector : dict
        Refueling material nuclides composition in form of dict, where keys are
        the nuclides str and values are the composition fractions.
    """
    def __init__(self, operator, model, mat_id, refuel_vector, range=None,
                 bracketed_method='brentq', tol=0.01, target=1.0,
                 atom_density_limit=0.0):

        super().__init__(operator, model, range, bracketed_method, tol, target,
                         atom_density_limit)

        if str(mat_id) not in self.burn_mats:
                msg = 'Mat_id: {} is not a valid depletable material id'\
                      .format(mat_id)
                raise ValueError(msg)
        else:
            self.mat_id = mat_id

        if len(range) != 2:
            raise ValueError("Range: {range} lenght is not 2. Must provide "
                             "bracketed range only")

        self.refuel_vector = refuel_vector


    def _build_parametric_model(self, param):
        """
        Builds the parametric model to be passed to `msr_criticality_search`
        Parameters
        ------------
        param :
            Model function variable, mass of material to refuel in grams
        Returns
        ------------
        _model :  openmc.model.Model
            OpenMC parametric model
        """
        _model = deepcopy(self.model)

        for i, mat in enumerate(self.burn_mats):
            for nuc, val in self.nucs[i].items():
                if nuc not in self.refuel_vector.keys():
                    if val < self.atom_density_limit or \
                    nuc not in self.operator.nuclides_with_data:
                        _model.materials[int(mat)-1].remove_nuclide(nuc)
                    else:
                        _model.materials[int(mat)-1].remove_nuclide(nuc)
                        _model.materials[int(mat)-1].add_nuclide(nuc, val)
                else:
                    if _model.materials[int(mat)-1].id == self.mat_id:
                        _model.materials[int(mat)-1].remove_nuclide(nuc)
                        # convert grams into atoms
                        atoms = param / atomic_mass(nuc) * AVOGADRO * \
                                self.refuel_vector[nuc]
                        _model.materials[int(mat)-1].add_nuclide(nuc, val+atoms)
                    else:
                        _model.materials[int(mat)-1].remove_nuclide(nuc)
                        _model.materials[int(mat)-1].add_nuclide(nuc, val)

        _model.export_to_xml()
        return _model

    def _update_x_vector_and_volumes(self, x, res):
        """
        Updates and returns total atoms with the root from the search_for_keff
        function. This function also calculates the new total volume
        in cc from nuclides atom densities, assuming constant material density
        as set by the user in the initial material definition.
        The new volume is then passed to the `openmc.deplete.AtomNumber` class
        to renormalize the atoms vector in the model in memory before running
        the next transport.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function
        res : float
            Root of the search_for_keff function
        Returns
        ------------
        x : list of numpy.ndarray
            Updated total atoms to be used in function
        diff : float
            Difference from previous iteration
        """
        diff = dict()

        for i, mat in enumerate(self.burn_mats):
            atoms_gram_per_mol = 0
            for j, nuc in enumerate(self.operator.number.burnable_nuclides):
                if nuc in self.refuel_vector.keys() and int(mat) == self.mat_id:
                    # Convert res grams into atoms
                    res_atoms = res / atomic_mass(nuc) * AVOGADRO * \
                                self.refuel_vector[nuc]
                    diff[nuc] = x[i][j] - res_atoms
                    x[i][j] += res_atoms
                atoms_gram_per_mol += x[i][j] * atomic_mass(nuc)

            # Calculate new volume and assign it in memory
            self.operator.number.volume[i] = atoms_gram_per_mol / AVOGADRO / \
                            self.model.materials[int(mat)-1].get_mass_density()

        return x, diff

    def msr_criticality_search(self, x):
        """
        Perform the criticality search on the parametric material model.
        Parameters
        ------------
        x : list of numpy.ndarray
            Total atoms to be used in function for normalization
        Returns
        ------------
        res : float
            Material mass root of search_for_keff function
        diff : float
            Difference from previous result
        """
        self.nucs = super()._order_vector(x)

        low_range = self.range[0]
        up_range = self.range[1]
        res = None
        while res == None:
            search = search_for_keff(self._build_parametric_model,
                    bracket=[low_range, up_range],
                    tol=self.tol,
                    bracketed_method=self.bracketed_method,
                    target=self.target,
                    print_iterations=True,
                    run_args={'output':False})

            if len(search) == 3:
                res, _, _ = search

            elif len(search) == 2:
                guesses, k = search
                if np.array(k).prod() < self.target:
                    print ('INFO: Function returned values BELOW target, ' \
                           'adapting bracket range...')
                    if (self.target - np.array(k).max()) < 0.001:
                        print ('INFO: Max value is close enough')
                        res =  guesses[np.array(k).argmax()]
                    else:
                        low_range = up_range
                        up_range *= 5
                else:
                    print ('INFO: Function returned values ABOVE target, ' \
                           'adapting bracket range...')
                    if (np.array(k).min() - self.target)  < 0.001:
                        print ('INFO: Min value is close enough')
                        res = guesses[np.array(k).argmin()]
                    else:
                        up_range = low_range
                        low_range /= 5

            else:
                raise ValueError(f'ERROR: search_for_keff output not valid')

        msg = 'UPDATE: Refueling: {:.2f} g --> ' \
              .format(res)
        print(msg)
        return  self._update_x_vector_and_volumes(x, res)
