"""
Allen-Cahn 1D

    u_t = eps * u_xx - (2/eps) * u (1-u) (1-2u),   x in [0,1], t in [0, 0.05]
    u(t,0) = u(t,1)                                (periodic boundary)
    u(0,x) = 0.25 sin(2 pi x) + 0.25 sin(16 pi x) + 0.5

with eps = 0.01. There is no analytical solution: using ground
truth an FEM on a very fine mesh (7993 nodes, dt = 1e-4/3). This script
exports the PINN prediction on a regular (t, x) grid, to be compared
with the FEM solution calculated separately.

PINN setup:
  - Nf = 20000, Ng = 250 on the boundary (periodic),
    Nh = 500 for the initial condition
    - weighted loss: initial-condition term multiplied by 1000
    - pre-training: Adam lr = 1e-4 for 7000 iterations ONLY on the
        initial-condition loss, then Adam lr = 1e-4 for 50000 iterations
        on the complete loss, and finally L-BFGS
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import deepxde as dde

from common import (get_parser, train_pinn, timed_predict, save_run)

EPS = 0.01 # diffusion coefficient in the PDE
T_FINAL = 0.05


def init_cond(X):
    x = X[:, 0:1]
    return 0.25 * np.sin(2 * np.pi * x) + 0.25 * np.sin(16 * np.pi * x) + 0.5


def pde(X, u):
    """Residual: u_t - eps u_xx + (2/eps) u (1-u) (1-2u)."""
    u_t = dde.grad.jacobian(u, X, i=0, j=1)
    u_xx = dde.grad.hessian(u, X, i=0, j=0)
    return u_t - EPS * u_xx + (2.0 / EPS) * u * (1 - u) * (1 - 2 * u)


def main():
    parser = get_parser(__doc__, default_arch=[100, 100, 100, 100],
                        default_adam=50000, default_lr=1e-4)
    parser.add_argument("--pretrain-iters", type=int, default=7000,
                        help="Adam iterations on the initial condition only")
    args = parser.parse_args()

    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, T_FINAL)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # periodic boundary: u(t,0) = u(t,1)
    bc = dde.icbc.PeriodicBC(geomtime, 0,
                             lambda X, on_boundary: on_boundary,
                             derivative_order=0, component=0)
    ic = dde.icbc.IC(geomtime, init_cond,
                     lambda X, on_initial: on_initial)

    nf, ng, nh = (20000, 250, 500)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                            num_domain=nf, num_boundary=ng,
                            num_initial=nh, train_distribution=args.dist)

    net = dde.nn.FNN([2] + args.arch + [1], "tanh", "Glorot normal")

    # Loss weights in the order [PDE residual, periodic BC, IC]:
    # the paper weights the initial-condition term by 1000.
    weights_full = [1, 1, 1000]
    weights_ic_only = [0, 0, 1]

    # training: pre-training on the initial condition, then full training, then L-BFGS
    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        loss_weights=weights_full,
        pretrain=(args.pretrain_iters, args.lr, weights_ic_only),
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period)

    # evaluation
    # (the same grid to use when exporting the reference FEM solution)
    nx, nt = 512, 51
    xs = np.linspace(0, 1, nx)
    ts = np.linspace(0, T_FINAL, nt)
    XX, TT = np.meshgrid(xs, ts, indexing="ij")
    X = np.stack([XX.ravel(), TT.ravel()], axis=1)
    y_pred, t_eval = timed_predict(model, X)

    name = "allen_cahn_1d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"note": "error to be calculated against FEM on a fine mesh "
                      "(no analytical solution)"},
             X, y_pred, ["x", "t", "u_pinn"],
             losshistory=losshistory, train_state=train_state)

    # plot: heatmap + profiles at t = 0, T/2, T
    U = y_pred.reshape(nx, nt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].pcolormesh(TT, XX, U, shading="auto")
    axes[0].set(xlabel="t", ylabel="x", title="Allen-Cahn 1D - PINN u(t,x)")
    fig.colorbar(im, ax=axes[0])
    for j, lbl in [(0, "t=0"), (nt // 2, "t=T/2"), (nt - 1, "t=T")]:
        axes[1].plot(xs, U[:, j], label=lbl)
    axes[1].plot(xs, init_cond(xs[:, None]), "k:", label="initial condition")
    axes[1].set(xlabel="x", ylabel="u")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Plot saved to: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
