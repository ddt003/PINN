# FEM vs PINN: Solving Poisson and Allen–Cahn Equations

This repository contains two independent sets of solvers for the same collection of
PDE problems — the Poisson equation in 1D, 2D and 3D, and the (time-dependent)
Allen–Cahn equation in 1D:

- **`fem/`** — classical Finite Element solvers written in C++ with
  [deal.II](https://www.dealii.org/), used to produce reference/ground-truth solutions.
- **`pinn/`** — Physics-Informed Neural Network (PINN) solvers written in Python with
  [DeepXDE](https://deepxde.readthedocs.io/), used to solve the same problems with a
  neural-network ansatz and compare accuracy/timing against the FEM reference.

## Repository layout

```
.
├── fem/                     # C++ / deal.II finite element solvers
│   ├── common/
│   │   └── cmake-common.cmake   # shared CMake settings (C++17, MPI, Boost, deal.II)
│   ├── poisson1d/                # 1D Poisson, Dirichlet on both ends
│   ├── poisson2d/                # 2D Poisson, mixed Dirichlet/Neumann
│   ├── poisson3d/                # 3D Poisson, homogeneous Dirichlet
│   └── allencahn/                 # 1D Allen–Cahn, parallel (MPI + Trilinos), Newton solver
│       └── src/{Parabolic.hpp,Parabolic.cpp,parabolic_main.cpp}
├── pinn/                    # Python / DeepXDE PINN solvers
│   ├── common.py             # shared CLI parser, training loop, saving/plotting helpers
│   ├── poisson_1d.py
│   ├── poisson_2d.py
│   ├── poisson_3d.py
│   ├── allen_cahn_1d.py
│   └── job.sh                # sample PBS job script for a GPU cluster
├── requirements.txt          # Python dependencies for the PINN scripts
├── results/                  # PINN outputs — git-ignored 
└── plots/                    # solution representations 
```

## The test cases

| Case | PDE | Boundary conditions | FEM | PINN |
|---|---|---|---|---|
| Poisson 1D | u'' = (4x³−6x)e⁻ˣ² on (0,1) | u(0)=0, u(1)=e⁻¹ | `fem/poisson1d` | `pinn/poisson_1d.py` |
| Poisson 2D | Δu = f(x,y) on (0,1)² | Neumann on x=0, x=1, y=1; Dirichlet on y=0 | `fem/poisson2d` | `pinn/poisson_2d.py` |
| Poisson 3D | Δu = −3π²sin(πx)sin(πy)sin(πz) on (0,1)³ | homogeneous Dirichlet | `fem/poisson3d` | `pinn/poisson_3d.py` |
| Allen–Cahn 1D | uₜ = ε u_xx − (2/ε)u(1−u)(1−2u), ε=0.01, t∈[0,0.05] | periodic in x | `fem/allencahn` | `pinn/allen_cahn_1d.py` |

All Poisson cases have a closed-form exact solution (used to compute L2/H1 errors).
The Allen–Cahn case has no analytical solution, so the FEM run on a fine mesh
(`fem/allencahn`) is meant to serve as the reference against which the PINN
prediction (`pinn/allen_cahn_1d.py`) is compared.

## Prerequisites

**FEM (C++)**
- CMake ≥ 3.12, a C++17 compiler
- MPI
- Boost ≥ 1.72.0 (`filesystem`, `iostreams`, `serialization`)
- deal.II ≥ 9.3.1 — the `allencahn` case additionally needs deal.II built with
  Trilinos support (it uses `TrilinosWrappers` matrices/vectors and a distributed
  triangulation)

**PINN (Python)**
- Python 3
- packages listed in `requirements.txt`: `deepxde`, `numpy`, `matplotlib`, and one
  backend for DeepXDE (the repo uses `torch`)

## Building and running the FEM solvers

Each Poisson case builds the same way:

```bash
cd fem/poisson1d      # or poisson2d, poisson3d
mkdir build && cd build
cmake ..
make
./elliptic
```

A couple of build-time switches live at the top of each case's `Elliptic.hpp`:

- `#define CONVERGENCE` — when enabled, `elliptic_main.cpp` loops over a list of
  mesh sizes `N`, solves the problem at each one, computes the L2/H1 error against
  the exact solution, writes `convergence.csv`, and prints a convergence-rate table.
  When disabled, it runs a single simulation at a fixed `N`.
- `#define NEUMANN` — enabled only in `poisson2d`, to assemble the Neumann
  contribution on the relevant boundary faces.

Every run writes `mesh-<N>.vtk` and `output-<N>.vtk` (viewable in ParaView/VisIt)
and prints timing information via deal.II's `TimerOutput`.

The Allen–Cahn case is parallel and uses a Newton solver at each time step:

```bash
cd fem/allencahn
mkdir build && cd build
cmake ..
make
./parabolic
```

It loops over mesh sizes N = 32, 128, 512, 2048
and writes `output.<step>.*.vtu` / `.pvtu` files for each time step.

## Running the PINN solvers

Set up the environment once:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

All four PINN scripts share a common CLI (defined in `pinn/common.py`):

| Flag | Meaning | Default |
|---|---|---|
| `--arch` | hidden layer sizes, e.g. `--arch 60 60 60` | script-specific |
| `--adam-iters` | number of Adam iterations | script-specific |
| `--lr` | Adam learning rate | script-specific |
| `--lbfgs-iters` | max iterations for the L-BFGS refinement step | 15000 |
| `--no-lbfgs` | skip the L-BFGS refinement | off |
| `--seed` | random seed | 1234 |
| `--outdir` | output directory | `results` |
| `--dist` | collocation-point sampling (`Hammersley`, `LHS`, `pseudo`) | `LHS` |
| `--resample-period` | epochs between resampling of collocation points (`0` = never) | 100 |
| `--bc` | `soft` (boundary conditions as an extra loss term) or `hard` (built into the network output) | `soft` |

Per-script defaults: `poisson_1d.py` uses `--arch 20 20 20 --adam-iters 15000 --lr 1e-4`;
`poisson_2d.py` and `poisson_3d.py` use `--arch 60 60 60 --adam-iters 20000 --lr 1e-3`;
`allen_cahn_1d.py` uses `--arch 100 100 100 100 --adam-iters 50000 --lr 1e-4` and adds
`--pretrain-iters` (Adam iterations spent on the initial condition alone before full
training, default 7000). `poisson_2d.py` and `poisson_3d.py` also add `--eval-n` to set
the resolution of the evaluation grid.

Examples:

```bash
# Poisson 1D with default settings
python pinn/poisson_1d.py --outdir results/poisson_1d

# Poisson 1D, boundary conditions built into the network instead of a loss term
python pinn/poisson_1d.py --bc hard --outdir results/poisson_1d

# Poisson 3D, LHS sampling, coarser 64^3 evaluation grid
python pinn/poisson_3d.py --dist LHS --eval-n 64 --outdir results/poisson_3d

# Allen-Cahn 1D (the heaviest case)
python pinn/allen_cahn_1d.py --arch 100 100 100 100 \
    --resample-period 100 --dist LHS --outdir results/allen_cahn
```

Each run writes, inside `--outdir`:
- `<case>_pred.csv` — PINN prediction on an evaluation grid (same columns as the
  matching FEM export, to ease comparison)
- `<case>_info.json` — architecture, training/evaluation time, errors
- `<case>.png` — solution and error plots
- `loss.dat`, `train.dat` — DeepXDE's loss history and sampled training points
