"""
Poisson 1D

    Delta u(x) = (4x^3 - 6x) e^{-x^2},   x in (0, 1)
    u(0) = 0,   u(1) = e^{-1}

Exact solution:  u(x) = x e^{-x^2}

Setup PINN:
  - N = 256 collocation points, Latin Hypercube, resample every 100 epochs (paper: every epoch)
  - FNN, tanh activation, Glorot normal initialization
  - Adam: 15000 iterations, lr = 1e-4, then L-BFGS refinement
  - evaluation on 512 points equispaced in [0, 1]
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import deepxde as dde
from deepxde import backend as bkd

from common import (get_parser, train_pinn, timed_predict, rel_l2, save_run)


def exact(x):
    return x * np.exp(-x ** 2)


U0, U1 = 0.0, float(np.exp(-1))  # u(0)=0, u(1)=e^{-1}


def pde(x, u):
    """Residual: u_xx - (4x^3 - 6x) e^{-x^2}."""
    u_xx = dde.grad.hessian(u, x)
    f = (4 * x ** 3 - 6 * x) * bkd.exp(-x ** 2)
    return u_xx - f


def main():
    args = get_parser(__doc__, default_arch=[20, 20, 20],
                      default_adam=15000, default_lr=1e-4).parse_args()
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Interval(0, 1)

    if args.bc == "soft":
        bc = dde.icbc.DirichletBC(geom, exact, lambda x, on_boundary: on_boundary)
        bcs = [bc]
        num_boundary = 2
    else:
        bcs = []
        num_boundary = 0

    data = dde.data.PDE(geom, pde, bcs, num_domain=256, num_boundary=num_boundary,
                        train_distribution=args.dist,
                        solution=exact, num_test=512)

    # layer_size, activation, initializer
    net = dde.nn.FNN([1] + args.arch + [1], "tanh", "Glorot normal")
    if args.bc == "hard":
        net.apply_output_transform(lambda x, y: (1 - x[:, 0:1]) * U0 + x[:, 0:1] * U1 + x[:, 0:1] * (1 - x[:, 0:1]) * y)

    # training: [pre-training] -> Adam -> L-BFGS
    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period, metrics=["l2 relative error"])

    # evaluation
    X = np.linspace(0, 1, 512)[:, None]
    y_pred, t_eval = timed_predict(model, X)
    y_true = exact(X)
    err = rel_l2(y_pred, y_true)

    name = f"poisson_1d_{'-'.join(map(str, args.arch))}_bc-{args.bc}"
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"l2_relative": err},
             X, y_pred, ["x", "u_pinn", "u_exact"], y_true,
             losshistory, train_state,
             extra_info={"bc_mode": args.bc,
                         "method": f"PINN with {args.bc} boundary conditions"})

    # graph
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(X, y_true, "k-", label="exact")
    axes[0].plot(X, y_pred, "r--", label="PINN")
    axes[0].set(xlabel="x", ylabel="u", title="Poisson 1D")
    axes[0].legend()
    axes[1].semilogy(X, np.abs(y_pred - y_true) + 1e-16)
    axes[1].set(xlabel="x", ylabel="|u_PINN - u_exact|",
                title=f"L2 rel. = {err:.2e}")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Graph saved in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
