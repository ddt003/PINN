"""
Poisson 3D sul cubo unitario (Sezione 4.3 del paper)

    Delta u(x,y,z) = -3 pi^2 sin(pi x) sin(pi y) sin(pi z),  (x,y,z) in (0,1)^3
    u = 0  su tutto il bordo (Dirichlet omogenea)

Soluzione esatta:  u(x,y,z) = sin(pi x) sin(pi y) sin(pi z)

Setup PINN del paper:
  - Nf = 1000 collocation nel dominio, Ng = 100 punti per gruppo di facce
    (qui: 600 punti totali sul bordo)
  - rete fully-connected tanh
  - Adam: 20000 iterazioni, lr = 1e-3, poi L-BFGS
  - architetture del paper: [20,20], [60,60], [20,20,20], [60,60,60],
      [20]*4, [60]*4, [20]*5, [60]*5
  - valutazione: il paper usa una griglia 150^3 (qui default 64^3,
    configurabile con --eval-n 150)

Esempio:  python poisson_3d.py --arch 60 60 60
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
from deepxde import backend as bkd

from pinn.common import (get_parser, apply_fast, train_pinn, timed_predict,
                    rel_l2, save_run)

PI = np.pi


def exact(X):
    return (np.sin(PI * X[:, 0:1]) * np.sin(PI * X[:, 1:2])
            * np.sin(PI * X[:, 2:3]))


def pde(X, u):
    """Residuo: Delta u + 3 pi^2 sin(pi x) sin(pi y) sin(pi z)."""
    lap = (dde.grad.hessian(u, X, i=0, j=0)
           + dde.grad.hessian(u, X, i=1, j=1)
           + dde.grad.hessian(u, X, i=2, j=2))
    f = -3 * PI ** 2 * (bkd.sin(PI * X[:, 0:1]) * bkd.sin(PI * X[:, 1:2])
                        * bkd.sin(PI * X[:, 2:3]))
    return lap - f


def main():
    parser = get_parser(__doc__, default_arch=[60, 60, 60],
                        default_adam=20000, default_lr=1e-3)
    parser.add_argument("--eval-n", type=int, default=64,
                        help="Griglia di valutazione eval-n^3 (paper: 150)")
    args = apply_fast(parser.parse_args())
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Cuboid([0, 0, 0], [1, 1, 1])
    bc = dde.icbc.DirichletBC(geom, lambda X: 0,
                              lambda X, on_boundary: on_boundary)

    data = dde.data.PDE(geom, pde, bc, num_domain=1000, num_boundary=600,
                        train_distribution=args.dist,
                        solution=exact, num_test=5000)

    net = dde.nn.FNN([3] + args.arch + [1], "tanh", "Glorot normal")

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period, metrics=["l2 relative error"])

    # --- Valutazione su griglia regolare n^3 ---------------------------------
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

    # --- Grafico: sezione a z = 0.5 (come le slice in Figura 5 del paper) ----
    k = n // 2
    U_pred = y_pred.reshape(n, n, n)[:, :, k]
    U_true = y_true.reshape(n, n, n)[:, :, k]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, U, title in zip(
            axes, [U_true, U_pred, np.abs(U_pred - U_true)],
            ["esatta (z=0.5)", "PINN (z=0.5)",
             f"|errore| (L2 rel = {err:.2e})"]):
        im = ax.pcolormesh(XX[:, :, k], YY[:, :, k], U, shading="auto")
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
