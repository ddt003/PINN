"""
3D Poisson equation on the unit cube

    Delta u(x,y,z) = -3 pi^2 sin(pi x) sin(pi y) sin(pi z),  (x,y,z) in (0,1)^3
    u = 0 on the entire boundary (homogeneous Dirichlet condition)

Exact solution:  u(x,y,z) = sin(pi x) sin(pi y) sin(pi z)

PINN setup:
    - Nf = 1000 collocation points in the domain, Ng = 100 points per face group
        (here: 600 total boundary points)
    - fully connected tanh network
    - Adam: 20000 iterations, lr = 1e-3, followed by L-BFGS
    - evaluation: the paper uses a 150^3 grid (here the default is 64^3,
        configurable with --eval-n 150)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import deepxde as dde
from deepxde import backend as bkd

from common import (get_parser, train_pinn, timed_predict, rel_l2, save_run)

PI = np.pi

def exact(X):
    return (np.sin(PI * X[:, 0:1]) * np.sin(PI * X[:, 1:2])
            * np.sin(PI * X[:, 2:3]))


def pde(X, u):
    """Residual: Delta u + 3 pi^2 sin(pi x) sin(pi y) sin(pi z)."""
    lap = (dde.grad.hessian(u, X, i=0, j=0)
           + dde.grad.hessian(u, X, i=1, j=1)
           + dde.grad.hessian(u, X, i=2, j=2))
    f = -3 * PI ** 2 * (bkd.sin(PI * X[:, 0:1]) * bkd.sin(PI * X[:, 1:2])
                        * bkd.sin(PI * X[:, 2:3]))
    return lap - f


def main():
    parser = get_parser(__doc__, default_arch=[60, 60, 60],
                        default_adam=20000, default_lr=1e-3)
    args = parser.parse_args()

    parser.add_argument("--eval-n", type=int, default=64,
                        help="Evaluation grid eval-n^3 (paper: 150)")
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])

    if args.bc == "soft":
        bc = dde.icbc.DirichletBC(geom, exact, lambda x, on_boundary: on_boundary)
        bcs = [bc]
        num_boundary = 600
    else:
        bcs = []
        num_boundary = 0

    data = dde.data.PDE(geom, pde, bcs, num_domain=1000, num_boundary=num_boundary,
                        train_distribution=args.dist,
                        solution=exact, num_test=5000)

    net = dde.nn.FNN([3] + args.arch + [1], "tanh", "Glorot normal")

    def output_transform_3d(x, y):
        x0 = x[:, 0:1]  # x-axis
        x1 = x[:, 1:2]  # y-axis
        x2 = x[:, 2:3]  # z-axis
        
        # Distance function: evaluates to 0 on all 6 faces of the [0, 1]^3 unit cube
        dist = x0 * (1 - x0) * x1 * (1 - x1) * x2 * (1 - x2)
        return dist * y
         
    if args.bc == "hard":
        net.apply_output_transform(output_transform_3d)

    # train
    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period, metrics=["l2 relative error"])

    # evaluation
    n = args.eval_n
    xs = np.linspace(0, 1, n)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)
    y_pred, t_eval = timed_predict(model, X)
    y_true = exact(X)
    err = rel_l2(y_pred, y_true)

    name = "poisson_3d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"l2_relative": err},
             X, y_pred, ["x", "y", "z", "u_pinn", "u_exact"], y_true,
             losshistory, train_state)

    # plot: section at z = 0.5
    k = n // 2
    U_pred = y_pred.reshape(n, n, n)[:, :, k]
    U_true = y_true.reshape(n, n, n)[:, :, k]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, U, title in zip(
            axes, [U_true, U_pred, np.abs(U_pred - U_true)],
            ["exact (z=0.5)", "PINN (z=0.5)",
             f"|error| (relative L2 = {err:.2e})"]):
        im = ax.pcolormesh(XX[:, :, k], YY[:, :, k], U, shading="auto")
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Plot saved to: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
