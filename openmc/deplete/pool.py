"""Dedicated module containing depletion function

Provided to avoid some circular imports
"""
from itertools import repeat, starmap
from multiprocessing import Pool
import functools
import copy
from scipy.sparse import coo_matrix, bmat
import numpy as np
# Configurable switch that enables / disables the use of
# multiprocessing routines during depletion
USE_MULTIPROCESSING = True


def deplete(func, chain, x, rates, dt, eql0d, matrix_func=None):
    """Deplete materials using given reaction rates for a specified time

    Parameters
    ----------
    func : callable
        Function to use to get new compositions. Expected to have the
        signature ``func(A, n0, t) -> n1``
    chain : openmc.deplete.Chain
        Depletion chain
    x : list of numpy.ndarray
        Atom number vectors for each material
    rates : openmc.deplete.ReactionRates
        Reaction rates (from transport operator)
    dt : float
        Time in [s] to deplete for
    maxtrix_func : callable, optional
        Function to form the depletion matrix after calling
        ``matrix_func(chain, rates, fission_yields)``, where
        ``fission_yields = {parent: {product: yield_frac}}``
        Expected to return the depletion matrix required by
        ``func``

    Returns
    -------
    x_result : list of numpy.ndarray
        Updated atom number vectors for each material

    """

    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(x):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(x)))

    idx_mat = [(v,int(k)) for k,v in rates.index_mat.items() ]
    eql0d_list = [[(mat[0],mat[0]),None] for mat in idx_mat]
    if eql0d is None:
        if matrix_func is None:
            matrices = map(chain.form_matrix, rates, eql0d_list, fission_yields)
        else:
            matrices = map(matrix_func, repeat(chain), rates, fission_yields)
        inputs = zip(matrices, x, repeat(dt))
        if USE_MULTIPROCESSING:
            with Pool() as pool:
                x_result = list(pool.starmap(func, inputs))
        else:
            x_result = list(starmap(func, inputs))

    else:
        rates_list = [rate for rate in rates]
        null_rate = copy.deepcopy(rates)[0]
        null_rate.fill(0)
        null_fy=copy.deepcopy(fission_yields)[0]
        for product, y in null_fy.items():
                y.yields.fill(0)

        for item in eql0d:
            i = [idx[0] for idx in idx_mat if idx[1]==item['mat_id']][0]
            eql0d_list[i][1] = item['transfer']
            for group in item['transfer']:
                rates_list.append(null_rate)
                fission_yields.append(null_fy)
                j = [idx[0] for idx in idx_mat if idx[1]==group['to']][0]
                eql0d_list.append([(j,i),group])
        rates = rates_list

        if matrix_func is None:
            matrices = map(chain.form_matrix, rates, eql0d_list, fission_yields)
        else:
            matrices = map(matrix_func, repeat(chain), rates, fission_yields)

        matrices_list = list(matrices)
        n=len(idx_mat)
        array_list = []
        for raw in range(n):
            raw_list =[None for d in range(n)]
            for col in range(n):
                for i,m in zip(eql0d_list,matrices_list):
                    if i[0] == (raw,col):
                        raw_list[col]=m
            array_list.append(raw_list)

        matrix = bmat(array_list)
        x = np.concatenate([_x for _x in x])
        x_result = func(matrix,x,dt)
        split_index = np.cumsum([i.shape[0] for i in matrices_list[:n]]).tolist()[:-1]
        x_result = np.split(x_result,split_index)

    return x_result
