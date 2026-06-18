# High-Order Matrix-Free Solver for Advection–Diffusion–Reaction Problems

A parallel, matrix-free finite element solver for the stationary
advection–diffusion–reaction (ADR) equation in 3D, built on
[deal.II](https://www.dealii.org/). The operator action is recomputed on the
fly with sum-factorization instead of assembling a global sparse matrix,
making the solver memory-bandwidth efficient and highly scalable on modern CPUs.

> **Course project** — *Numerical Methods for PDEs*, A.Y. 2025/2026.
> See the accompanying report (`report/`) for the full mathematical formulation,
> the SUPG stabilization, the preconditioning strategy, and the numerical results.

---

## Features

- **Matrix-free operator evaluation** via `MatrixFree` / `FEEvaluation`, with
  sum-factorization on tensor-product (hexahedral) elements.
- **Arbitrary polynomial degree** (continuous `FE_Q`); instantiated for
  degrees 1–7.
- **Distributed-memory parallelism** through
  `parallel::distributed::Triangulation` (MPI + SIMD vectorization).
- **Geometric Multigrid (GMG) preconditioner** in **mixed precision**:
  `double` on the active level, `float` on the multigrid hierarchy.
- **Chebyshev polynomial smoother** (matrix-free: needs only mat-vec products
  and the operator diagonal).
- **GMRES** outer solver (the ADR operator is non-symmetric due to advection).
- **SUPG stabilization** for the advection-dominated regime, toggleable
  (see [Configuration](#configuration)).
- **Inhomogeneous Dirichlet** boundary conditions via lifting; Neumann faces
  supported.
- **Convergence study** with `L2` and `H1`-seminorm errors against a
  manufactured solution, reported in a convergence table.

---

## Prerequisites

- **deal.II ≥ 9.3.1**, configured **with MPI and p4est** (required for
  `parallel::distributed::Triangulation`).
- An **MPI** implementation (e.g. OpenMPI).
- **CMake ≥ 3.13** and a **C++17** compiler.

At PoliMi the dependencies are available through the `mk` modules; the build
honors the `mkDealiiPrefix` / `DEAL_II_DIR` environment variables. Typically:

```bash
module load gcc-glibc dealii      # or your local module setup
```

Alternatively, point CMake to a manual deal.II install with
`-DDEAL_II_DIR=/path/to/dealii`.

---

## Building

```bash
# from the project root
mkdir build && cd build
cmake ..
make -j
```

Useful options:

```bash
# Debug build (enables -DBUILD_TYPE_DEBUG and deal.II debug checks)
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Override the vectorization target (default: native)
cmake -DTARGET_ARCH=skylake-avx512 ..   # or core-avx2, etc.
```

The default build type is `Release` (`-O3 -march=native`).

> **Note:** the `out/` directory must exist before running if VTU output is
> enabled, since deal.II does not create it. Create it with `mkdir out` at the
> appropriate level (output is written to `../out/` relative to the working
> directory).

---

## Running

```bash
# from the build directory
mpirun -np <N> ./<executable>
```

where `<N>` is the number of MPI processes and `<executable>` is the target name
defined in the top-level `CMakeLists.txt` (`add_executable(...)`).

The program runs a sequence of refinement cycles, printing per-cycle DoF counts,
solver iterations, timings and a memory summary, and finishes with the
convergence table (`L2` / `H1` errors and rates).

---

## Configuration

The main knobs are compile-time and live in the source:

| What | Where |
|------|-------|
| Spatial dimension `dim` and polynomial degree `fe_degree` | `main.cpp` |
| Number of refinement cycles `n_cycles` | `SolverClass::run()` |
| **SUPG on/off** | `ADR_Operator::compute_tau()` — returns `0` (off) by default; uncomment the `return 1/sqrt(...)` line to enable. The matching toggle is in `SolverClass::assemble_rhs()`. |
| Mesh, geometry and boundary IDs | `SolverClass::init_mesh()` |
| Problem data (μ, β, γ, f, BCs, exact solution) | `test/ |
| VTU output | disabled by default — uncomment `output_results(...)` in `run()` |

> Changing `dim` or `fe_degree` requires the corresponding explicit template
> instantiation to exist at the bottom of `MatrixFree.cpp` and
> `ADROperator.cpp` (degrees 1–7 in 3D are already provided).

---

## Repository structure

```
.
├── CMakeLists.txt          # top-level build script (includes cmake-common.cmake)
├── cmake-common.cmake      # shared compiler/deal.II/MPI configuration
├── matrix-based/            # matrix based solver sources
├── matrix-free/            # matrix free solver sources
│   ├── main.cpp            # entry point: sets dim/degree, runs the solver
│   ├── MatrixFree.hpp      # SolverClass: setup, multigrid, solve, post-processing
│   ├── MatrixFree.cpp
│   ├── ADROperator.hpp     # matrix-free ADR operator
│   └── ADROperator.cpp     # local_apply, SUPG (compute_tau), diagonal
├── test/                   # problem definition
│   ├── homogeneousManufactured.hpp    #homogeneous Dirichlet BCs
|   ├── nonHomogeneousManufactured.hpp    #no homogeneous Dirichlet BCs    
│   |── noAdvection.hpp    #test without advection term
|    └── dominantAdvection.hpp    #test with high advection term
├── out/                    # VTU output (runtime)
```

---

## Authors

Davide Cutrupi · Luca Di Profio · Simone De Carlo · Davide Di Tanna

*Numerical Methods for PDEs — A.Y. 2025/2026*
