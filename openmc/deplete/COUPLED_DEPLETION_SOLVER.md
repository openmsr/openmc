# Coupled Depletion Solver — Mathematics and Algorithm

> Internal design document for the coupled depletion system in `openmc.deplete`.
> Covers the Bateman equations, transfer rates, external source rates, and the
> block Jacobi solver implemented in `pool.py`.

---

## 1. Single-material depletion (baseline)

The time evolution of nuclide densities $\mathbf{N}(t) \in \mathbb{R}^n$ in a
single depletable material is governed by the Bateman equations:

$$
\frac{d\mathbf{N}}{dt} = A\,\mathbf{N}, \qquad \mathbf{N}(0) = \mathbf{N}_0
$$

where $A \in \mathbb{R}^{n \times n}$ is the sparse transmutation matrix built
from reaction rates and radioactive decay constants ($n \approx 1750$ nuclides
in a typical chain).  The formal solution is the matrix exponential
$\mathbf{N}(t) = e^{At}\mathbf{N}_0$.

### 1.1 IPF-CRAM solver

The matrix exponential is approximated using the Incomplete Partial Fraction
(IPF) Chebyshev Rational Approximation Method (CRAM) of order $p$ (16 or 48):

$$
e^{At} \approx \alpha_0 \prod_{k=1}^{p/2}\left(I + 2\,\mathrm{Re}\!\left[
    \alpha_k\,(A\,t - \theta_k I)^{-1}
\right]\right)
$$

where $\{\theta_k\} \subset \mathbb{C}$ are the poles and
$\{\alpha_k\} \subset \mathbb{C}$ are the residues of the rational
approximation.  Evaluation proceeds iteratively:

```
y ← N₀
for each pole (αₖ, θₖ):
    y ← y + 2 Re[αₖ · (At − θₖI)⁻¹ y]
y ← α₀ · y
```

Each inner step requires solving one complex sparse linear system
$(At - \theta_k I)\,\mathbf{x} = \mathbf{y}$ via sparse LU factorisation
(`scipy.sparse.linalg.splu`).

**Substeps.** When `substeps > 1` the interval $[0, \Delta t]$ is split into
$s$ equal sub-intervals of length $\delta t = \Delta t / s$.  LU
factorisations of $(A\,\delta t - \theta_k I)$ are computed once and reused
across all sub-intervals.

**Implementation:** `IPFCramSolver.__call__` in `openmc/deplete/cram.py`.

---

## 2. Transfer rates — single material (no destination)

A continuous removal or feed of nuclide $i$ at rate $t_i\ [\text{s}^{-1}]$
adds a proportional loss term to the Bateman equation:

$$
\frac{dN_i}{dt} = \sum_j A_{ij} N_j - t_i N_i
$$

The transfer term is subtracted from the diagonal of $A$, yielding a modified
matrix $\tilde{A} = A - T$ where $T = \mathrm{diag}(t_1, \ldots, t_n)$ is a
diagonal matrix with nonzero entries only for the transferred nuclides
(typically $\ll n$ entries).  The resulting system is solved identically to the
uncoupled case.

**Implementation:** `chain.form_rr_term` builds $T$; `pool.deplete` subtracts
it from the Bateman matrix before calling the solver.

---

## 3. Transfer rates — coupled materials (`destination_material`)

When nuclides are transferred *from* material $j$ *into* material $i$, the
two depletion equations must be solved simultaneously.  For $M$ materials the
system becomes:

$$
\frac{d}{dt}
\begin{pmatrix} \mathbf{N}_1 \\ \vdots \\ \mathbf{N}_M \end{pmatrix}
=
\underbrace{
\begin{pmatrix}
  B_{11} & \cdots & T_{1M} \\
  \vdots & \ddots & \vdots \\
  T_{M1} & \cdots & B_{MM}
\end{pmatrix}
}_{B}
\begin{pmatrix} \mathbf{N}_1 \\ \vdots \\ \mathbf{N}_M \end{pmatrix}
$$

where:

* **Diagonal blocks** $B_{ii} = A_{ii} - \sum_{j \neq i} T_{ji}^{\text{out}}$
  — the Bateman matrix for material $i$ minus the diagonal removal terms for
  all transfers leaving $i$.
* **Off-diagonal blocks** $T_{ij}$ ($i \neq j$) — diagonal matrices whose
  nonzero entries equal the transfer rates of nuclides flowing from $j$ into
  $i$.  At most a few dozen nonzeros out of $n^2$ possible entries.

> **Structural note.** Each $T_{ij}$ is strictly diagonal (entry `matrix[k,k]`
> only, as built by `chain.form_rr_term`).  The block matrix $B$ is therefore
> very sparse in the off-diagonal blocks, even if $M$ is large.

---

## 4. External source rates

An external feed or removal of nuclide $i$ at a constant rate $S_i\
[\text{atoms/s}]$ adds a non-homogeneous term:

$$
\frac{dN_i}{dt} = \sum_j A_{ij} N_j + S_i
$$

The non-homogeneous system is recast as homogeneous by augmenting the state
vector with a dummy component fixed at unity:

$$
\frac{d}{dt}
\begin{pmatrix} \mathbf{N} \\ 1 \end{pmatrix}
=
\underbrace{
\begin{bmatrix}
  A      & \mathbf{S} \\
  \mathbf{0}^T & 0
\end{bmatrix}
}_{\tilde{A}}
\begin{pmatrix} \mathbf{N} \\ 1 \end{pmatrix}
$$

where $\mathbf{S} \in \mathbb{R}^n$ is the source rate vector.  The augmented
system $\dot{\tilde{\mathbf{N}}} = \tilde{A}\tilde{\mathbf{N}}$ has the same
form as the standard Bateman system and is solved by the same CRAM call.  After
the solve the trailing component (which remains $\approx 1$) is stripped.

**Rates** are specified in mass per time (default `g/s`) and converted to
`atoms/s` via $S_i = \dot{m} \cdot w_i \cdot N_A / M_i$ where $w_i$ is the
weight fraction of nuclide $i$ in the feed stream.

**Implementation:** `chain.form_ext_source_term` builds $\mathbf{S}`;
`pool.deplete` performs the augmentation with `hstack` / `resize`.

---

## 5. Coupled block Jacobi solver

### 5.1 Motivation

A naive coupled solve would assemble the full $Mn \times Mn$ block matrix $B$
and call the CRAM solver once on it (serial, runs only on rank 0).  For large
$M$ this is a bottleneck: the LU factorisation cost scales as $M$ times the
single-material cost, and no parallelism is exploited.

The block Jacobi approach avoids both issues by treating the off-diagonal
coupling blocks as a *source term* that is updated iteratively.  Each iteration
involves only per-material CRAM solves that are fully independent and can be
dispatched via `multiprocessing.Pool`.  MPI ranks keep their local materials
throughout; no gather/broadcast is required.

### 5.2 Derivation — augmented-matrix formulation

At each Jacobi iteration $k$, the coupling from source materials $j$ into
receiving material $i$ is treated as a *known* constant source vector:

$$
\mathbf{c}_i^{(k)} = \sum_{j \neq i} T_{ij}\,\mathbf{x}_j^{(k)}
$$

where $\mathbf{x}_j^{(k)}$ is the composition of material $j$ computed in the
previous iteration.  The ODE for material $i$ then reads:

$$
\frac{d\mathbf{N}_i}{dt} = B_{ii}\,\mathbf{N}_i + \mathbf{c}_i^{(k)},
\qquad \mathbf{N}_i(0) = \mathbf{N}_i^{(0)}
$$

This is a non-homogeneous linear ODE with a constant source.  Its exact
solution over $[0, \Delta t]$ is given by the variation-of-constants formula:

$$
\mathbf{x}_i^{(k+1)}
= e^{B_{ii}\Delta t}\,\mathbf{N}_i^{(0)}
+ \underbrace{\int_0^{\Delta t} e^{B_{ii}(\Delta t - s)}\,ds}_{\displaystyle A_{ii}^{-1}(e^{B_{ii}\Delta t} - I)}
  \mathbf{c}_i^{(k)}
$$

This integral is evaluated **exactly** using the same augmented-matrix trick as
for external source rates (Section 4).  The coupling vector $\mathbf{c}_i^{(k)}$
is appended as an extra column to the Bateman matrix, the system is resized to
square by adding a zero row, and the initial state is extended with a trailing
$1$:

$$
\tilde{A}_i =
\begin{bmatrix}
  B_{ii} & \mathbf{c}_i^{(k)} \\
  \mathbf{0}^T & 0
\end{bmatrix},
\qquad
\tilde{\mathbf{N}}_i =
\begin{pmatrix} \mathbf{N}_i^{(0)} \\ 1 \end{pmatrix}
$$

The augmented solve $e^{\tilde{A}_i \Delta t}\,\tilde{\mathbf{N}}_i$ yields
$\mathbf{x}_i^{(k+1)}$ in its first $n$ components.  The trailing component
(always $\approx 1$) is discarded afterwards.

### 5.3 Convergence

The iteration converges when the spectral radius of the coupling operator is
less than 1.  A sufficient condition is:

$$
\Delta t \cdot \max_{i,j,\text{nuclide}} t_{ij} \ll 1
$$

Typical MSR transfer rates ($t \sim 10^{-4}\text{–}10^{-7}\ \text{s}^{-1}$)
and daily timesteps give $\Delta t \cdot t \sim 10^{-2}\text{–}10^{-5}$, so
**1–2 iterations** are sufficient in practice.  For fast transfer rates
($t \sim 10^{-2}\ \text{s}^{-1}$) with long timesteps more iterations (or
smaller timesteps) may be needed.

The convergence criterion used in the code is a relative change check on each
receiving material:

$$
\frac{\|\mathbf{x}_i^{(k+1)} - \mathbf{x}_i^{(k)}\|}{\|\mathbf{x}_i^{(k+1)}\|}
\leq \varepsilon
$$

where $\varepsilon$ = `jacobi_tol`.  Global convergence (MPI) is declared only
when all ranks have converged locally.

### 5.4 Algorithm

```
Input: matrices [B₀,…,B_{M-1}],  n [N₀,…,N_{M-1}],
       recv_from[i] = [(global_j, T_ij), ...],
       dt, substeps, max_jacobi_iter, jacobi_tol

# Step 0: uncoupled solve (exact for non-receiving materials)
x ← pool.starmap(func, [(Bᵢ, Nᵢ, dt, substeps) for each local i])

# Jacobi iterations
for k = 1..max_jacobi_iter:

    # MPI exchange: share x vectors across all ranks
    if MPI:
        x_lookup ← comm.allgather(x)   # global_j -> xⱼ^(k)
    else:
        x_lookup ← {local_indices[i]: x[i]}

    # Build augmented inputs for receiving materials
    for each local material i:
        if recv_from[i] is not empty:
            cᵢ ← Σⱼ  Tᵢⱼ · x_lookup[j]       # coupling vector (sparse)
            Ã_i ← [[Bᵢ, cᵢ], [0…, 0]]         # augmented (n+1)×(n+1) matrix
            ñᵢ  ← [Nᵢ; 1.0]                   # extended initial state
        else:
            Ã_i ← Bᵢ,  ñᵢ ← Nᵢ               # no coupling, unchanged

    # Solve all materials in parallel
    x_raw ← pool.starmap(func, [(Ã_i, ñᵢ, dt, substeps) for each local i])

    # Strip trailing component from augmented solves
    x_new[i] ← x_raw[i][:-1]  if recv_from[i] else x_raw[i]

    # Check convergence
    local_conv ← all ‖x_new[i] − x[i]‖ ≤ jacobi_tol · ‖x_new[i]‖
                  for i with recv_from[i] non-empty
    if MPI:  converged ← comm.allgather(local_conv).all()
    else:    converged ← local_conv

    x ← x_new
    if converged: break

Output: x  (local compositions, already distributed — no bcast needed)
```

### 5.5 Implementation details

- **`recv_from[local_i]`** — list of `(global_j, T_ij)` pairs for every
  material that sends nuclides into local material `i`.  Built once before the
  iteration loop from `transfer_rates.index_transfer`.
- **Augmentation pattern** — identical to the ESR path: `hstack([A, col])`,
  then `A.resize(n+1, n+1)`, then `np.append(n0, 1.0)`.  No CRAM-internal
  changes are required; `func` is called unmodified.
- **`pool.starmap`** — both the zeroth-order solve and each Jacobi iteration
  dispatch all per-material calls through the existing `Pool(NUM_PROCESSES)`
  infrastructure, giving shared-memory parallelism for free.
- **MPI distribution** — each rank retains its local materials throughout.
  Between iterations, `comm.allgather(x)` exchanges composition vectors across
  all ranks so that each rank can compute the coupling term for its local
  receiving materials.  Global convergence is checked via
  `sum(comm.allgather(int(not local_conv))) == 0`.
- **Jacobi vs Gauss–Seidel** — pure Jacobi is used (each iteration reads
  `x^(k)` from all materials, not the partially-updated `x^(k+1)`).  This
  ensures that all per-material solves within one iteration are truly
  independent.
- **Implementation:** `pool.deplete`, coupled-transfer branch in
  `openmc/deplete/pool.py`.

---

## 6. Configuration API

```python
integrator = openmc.deplete.PredictorIntegrator(
    op, timesteps, power,
    max_jacobi_iter = 2,    # Jacobi iterations per timestep
    jacobi_tol      = 1e-8, # relative convergence tolerance
)
integrator.add_transfer_rate('fuel', ['Xe'], 0.1,
                              destination_material='storage')
```

The parameters are propagated as:

```
Integrator.__init__
  → self.max_jacobi_iter / self.jacobi_tol
    ↓ synced in add_transfer_rate
  → TransferRates.max_jacobi_iter / .jacobi_tol
    ↓ read in pool.deplete Jacobi loop
  → augmented-matrix Jacobi iterations (pool.starmap)
```

---

## 7. Completed and planned work

- [x] **Parallel per-material solves** — `pool.starmap(func, ...)` is used for
  both the zeroth-order step and each Jacobi correction, giving shared-memory
  parallelism via `multiprocessing.Pool` without any CRAM-internal changes.
- [x] **MPI distribution** — each rank keeps its local materials; coupling
  vectors are exchanged via `comm.allgather` between iterations; no
  gather/broadcast of full result arrays is needed.
- [ ] **DAG fast path** — detect when the transfer graph is a directed acyclic
  graph (no cycles); in that case the system can be solved exactly in one pass
  through the topological order (block forward substitution), with no iteration
  needed.
