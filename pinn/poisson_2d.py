"""
Poisson 2D con condizioni al bordo miste (Sezione 4.2 del paper)

    Delta u(x,y) = f(x,y),   (x,y) in (0,1)^2
    f(x,y) = 2( x^4(3y-2) + x^3(4-6y) + x^2(6y^3-12y^2+9y-2)
                - 6x(y-1)^2 y + (y-1)^2 y )

    Neumann:    du/dn = 0  su  x=0, x=1, y=1
    Dirichlet:  u = 0      su  y=0

Soluzione esatta:  u(x,y) = x^2 (x-1)^2 y (y-1)^2

Setup PINN del paper:
  - Nf = 2000 collocation nel dominio, Ng = 250 sul bordo (LHS, ricampionati)
  - rete fully-connected tanh
  - Adam: 20000 iterazioni, lr = 1e-3, poi L-BFGS
  - architetture del paper: [20], [60], [20,20], [60,60], [20,20,20],
      [60,60,60], [20]*4, [60]*4, [20]*5, [60]*5, [120]*5

Esempio:  python poisson_2d.py --arch 60 60 60
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

from pinn.common import (get_parser, apply_fast, train_pinn, timed_predict,
                    rel_l2, save_run)


def exact(X):
    x, y = X[:, 0:1], X[:, 1:2]
    return x ** 2 * (x - 1) ** 2 * y * (y - 1) ** 2


def rhs(x, y):
    return 2 * (x ** 4 * (3 * y - 2)
                + x ** 3 * (4 - 6 * y)
                + x ** 2 * (6 * y ** 3 - 12 * y ** 2 + 9 * y - 2)
                - 6 * x * (y - 1) ** 2 * y
                + (y - 1) ** 2 * y)


def pde(X, u):
    """Residuo: u_xx + u_yy - f(x, y)."""
    u_xx = dde.grad.hessian(u, X, i=0, j=0)
    u_yy = dde.grad.hessian(u, X, i=1, j=1)
    x, y = X[:, 0:1], X[:, 1:2]
    return u_xx + u_yy - rhs(x, y)


def on_dirichlet(X, on_boundary):          # y = 0
    return on_boundary and np.isclose(X[1], 0)


def on_neumann(X, on_boundary):            # x = 0, x = 1, y = 1
    return on_boundary and (np.isclose(X[0], 0) or np.isclose(X[0], 1)
                            or np.isclose(X[1], 1))


def main():
    parser = get_parser(__doc__, default_arch=[60, 60, 60],
                        default_adam=20000, default_lr=1e-3)
    parser.add_argument("--eval-n", type=int, default=512,
                        help="Griglia di valutazione eval-n x eval-n "
                             "(il paper usa mesh fino a 2000x2000)")
    args = apply_fast(parser.parse_args())
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    bc_d = dde.icbc.DirichletBC(geom, lambda X: 0, on_dirichlet)
    bc_n = dde.icbc.NeumannBC(geom, lambda X: 0, on_neumann)

    data = dde.data.PDE(geom, pde, [bc_d, bc_n],
                        num_domain=2000, num_boundary=250,
                        train_distribution=args.dist,
                        solution=exact, num_test=5000)

    net = dde.nn.FNN([2] + args.arch + [1], "tanh", "Glorot normal")

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period, metrics=["l2 relative error"])

    # --- Valutazione su griglia regolare -------------------------------------
    n = args.eval_n
    xs = np.linspace(0, 1, n)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    X = np.stack([XX.ravel(), YY.ravel()], axis=1)
    y_pred, t_eval = timed_predict(model, X)
    y_true = exact(X)
    err = rel_l2(y_pred, y_true)

    name = "poisson_2d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"l2_relative": err},
             X, y_pred, ["x", "y", "u_pinn", "u_exact"], y_true,
             losshistory, train_state)

    # --- Grafico -------------------------------------------------------------
    U_pred = y_pred.reshape(n, n)
    U_true = y_true.reshape(n, n)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, U, title in zip(
            axes, [U_true, U_pred, np.abs(U_pred - U_true)],
            ["esatta", "PINN", f"|errore| (L2 rel = {err:.2e})"]):
        im = ax.pcolormesh(XX, YY, U, shading="auto")
        ax.set(xlabel="x", ylabel="y", title=title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
