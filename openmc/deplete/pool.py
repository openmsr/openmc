"""Dedicated module containing depletion function

Provided to avoid some circular imports
"""
from itertools import repeat, starmap
from multiprocessing import Pool

import numpy as np
from scipy.sparse import hstack

from openmc.mpi import comm
from .._sparse_compat import csc_array

# Configurable switch that enables / disables the use of
# multiprocessing routines during depletion
USE_MULTIPROCESSING = True

# Allow user to override the number of worker processes to use for depletion
# calculations
NUM_PROCESSES = None


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

def deplete(func, chain, n, rates, dt, current_timestep=None, matrix_func=None,
            transfer_rates=None, external_source_rates=None, substeps=1,
            *matrix_args):
    """Deplete materials using given reaction rates for a specified time

    Parameters
    ----------
    func : callable
        Function to use to get new compositions. Expected to have the signature
        ``func(A, n0, t, substeps=1) -> n1``.
    chain : openmc.deplete.Chain
        Depletion chain
    n : list of numpy.ndarray
        List of atom number arrays for each material. Each array in the list
        contains the number of [atom] of each nuclide.
    rates : openmc.deplete.ReactionRates
        Reaction rates (from transport operator)
    dt : float
        Time in [s] to deplete for
    current_timestep : int
        Current timestep index
    maxtrix_func : callable, optional
        Function to form the depletion matrix after calling ``matrix_func(chain,
        rates, fission_yields)``, where ``fission_yields = {parent: {product:
        yield_frac}}`` Expected to return the depletion matrix required by
        ``func``
    transfer_rates : openmc.deplete.TransferRates, Optional
        Transfer rates for continuous removal/feed.

        .. versionadded:: 0.14.0
    external_source_rates : openmc.deplete.ExternalSourceRates, Optional
        External source rates for continuous removal/feed.

        .. versionadded:: 0.15.3
    substeps : int, optional
        Number of substeps to pass to solvers that support substepping.
    matrix_args: Any, optional
        Additional arguments passed to matrix_func

    Returns
    -------
    n_result : list of numpy.ndarray
        Updated list of atom number arrays for each material. Each array in the
        list contains the number of [atom] of each nuclide.

    """

    fission_yields = chain.fission_yields
    if len(fission_yields) == 1:
        fission_yields = repeat(fission_yields[0])
    elif len(fission_yields) != len(n):
        raise ValueError(
            "Number of material fission yield distributions {} is not "
            "equal to the number of compositions {}".format(
                len(fission_yields), len(n)))

    if matrix_func is None:
        matrices = map(chain.form_matrix, rates, fission_yields)
    else:
        matrices = map(matrix_func, repeat(chain), rates, fission_yields,
                       *matrix_args)

    if (transfer_rates is not None and
        current_timestep in transfer_rates.external_timesteps):
        # Calculate transfer rate terms as diagonal matrices
        transfers = map(chain.form_rr_term, repeat(transfer_rates),
                        repeat(current_timestep), transfer_rates.local_mats)

        # Subtract transfer rate terms from Bateman matrices
        matrices = [matrix - transfer for (matrix, transfer) in zip(matrices,
                                                                    transfers)]

        if transfer_rates.redox:
            for mat_idx, mat_id in enumerate(transfer_rates.local_mats):
                if mat_id in transfer_rates.redox:
                    matrices[mat_idx] = chain.add_redox_term(matrices[mat_idx],
                                                transfer_rates.redox[mat_id][0],
                                                transfer_rates.redox[mat_id][1])

        if current_timestep in transfer_rates.index_transfer:
            # ── Distributed block Jacobi via pool.starmap ────────────────────
            # Each MPI rank keeps its own local materials; no gather/broadcast
            # is needed.  The coupling from other materials is incorporated as
            # a constant source column (augmented-matrix approach, identical to
            # ExternalSourceRates) and updated each Jacobi iteration using the
            # previous full-material iterate.  Ranks exchange composed vectors
            # via comm.allgather between iterations.
            mat_idx = {mat_id: i for i, mat_id in enumerate(transfer_rates.burnable_mats)}
            local_mat_idx = {mat_id: i for i, mat_id in
                             enumerate(transfer_rates.local_mats)}

            # recv_from[local_i] = [(global_j, T_ij), ...] for every source
            # material j that transfers nuclides into local material i.
            recv_from = [[] for _ in range(len(transfer_rates.local_mats))]
            for mat_pair in transfer_rates.index_transfer[current_timestep]:
                dest, src = mat_pair
                if dest not in local_mat_idx:
                    continue
                transfer_matrix = chain.form_rr_term(transfer_rates,
                                              current_timestep, mat_pair)
                if dest in transfer_rates.redox:
                    transfer_matrix = chain.add_redox_term(
                        transfer_matrix,
                        transfer_rates.redox[dest][0],
                        transfer_rates.redox[dest][1])
                recv_from[local_mat_idx[dest]].append((mat_idx[src], transfer_matrix))

            local_indices = [mat_idx[mat_id] for mat_id in transfer_rates.local_mats]

            # Precompute global-index mapping for MPI allgather.
            if comm.size > 1:
                all_local_indices = comm.allgather(local_indices)
            
            # Step 0: uncoupled solve for each local material.
            inputs = zip(matrices, n, repeat(dt), repeat(substeps))
            if USE_MULTIPROCESSING:
                with Pool(NUM_PROCESSES) as pool:
                    x = list(pool.starmap(func, inputs))
            else:
                x = list(starmap(func, inputs))

            # Jacobi iterations: re-solve with the coupling from the
            # previous iterate added as an augmented constant source.
            for _ in range(transfer_rates.max_jacobi_iter):
                # Build x_lookup: global_j -> composition from prev iter.
                if comm.size > 1:
                    all_x = comm.allgather(x)
                    x_lookup = {
                        g: xv
                        for r_idx, r_x in zip(all_local_indices, all_x)
                        for g, xv in zip(r_idx, r_x)
                    }
                else:
                    x_lookup = dict(zip(local_indices, x))

                # Build (matrix, n0, dt, substeps) tuples.  Materials that
                # receive transfers get an augmented matrix with the coupling
                # vector appended as an extra column (same pattern as ESR).
                coupled = []
                aug_inputs = []
                for i, (A, ni) in enumerate(zip(matrices, n)):
                    if recv_from[i]:
                        coupling = sum(t @ x_lookup[j]
                                       for j, t in recv_from[i])
                        A_aug = hstack(
                            [A, csc_array(coupling.reshape(-1, 1))])
                        A_aug.resize(A_aug.shape[1], A_aug.shape[1])
                        aug_inputs.append(
                            (A_aug, np.append(ni, 1.0), dt, substeps))
                        coupled.append(True)
                    else:
                        aug_inputs.append((A, ni, dt, substeps))
                        coupled.append(False)

                if USE_MULTIPROCESSING:
                    with Pool(NUM_PROCESSES) as pool:
                        x_raw = list(pool.starmap(func, aug_inputs))
                else:
                    x_raw = list(starmap(func, aug_inputs))

                # Strip the dummy trailing component from augmented solves.
                x_new = [xr[:-1] if c else xr
                         for xr, c in zip(x_raw, coupled)]

                # Convergence: relative change in all receiving materials.
                local_conv = all(
                    not recv_from[i] or
                    np.linalg.norm(x_new[i] - x[i]) <=
                    transfer_rates.jacobi_tol * np.linalg.norm(x_new[i])
                    for i in range(len(transfer_rates.local_mats))
                )
                if comm.size > 1:
                    converged = (
                        sum(comm.allgather(int(not local_conv))) == 0)
                else:
                    converged = local_conv

                x = x_new
                if converged:
                    break

            # n_result is already local — no bcast/distribute needed.
            return x           


 

    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        # Calculate external source term vectors
        sources = map(chain.form_ext_source_term, repeat(external_source_rates),
                      repeat(current_timestep), external_source_rates.local_mats)

        # stack vector column at the end of the matrix
        matrices = [
            hstack([matrix, source])
            for matrix, source in zip(matrices, sources)
        ]

        # Add a last row of zeroes to the matrices and append 1 to the last row
        # of the nuclide vectors
        for i, matrix in enumerate(matrices):
            if not np.equal(*matrix.shape):
                matrix.resize(matrix.shape[1], matrix.shape[1])
                n[i] = np.append(n[i], 1.0)

    inputs = zip(matrices, n, repeat(dt), repeat(substeps))

    if USE_MULTIPROCESSING:
        with Pool(NUM_PROCESSES) as pool:
            n_result = list(pool.starmap(func, inputs))
    else:
        n_result = list(starmap(func, inputs))

    # Remove extra value at the end of the nuclide vectors
    if (external_source_rates is not None and
        current_timestep in external_source_rates.external_timesteps):
        external_source_rates.reformat_nuclide_vectors(n)
        external_source_rates.reformat_nuclide_vectors(n_result)

    return n_result
