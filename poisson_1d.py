"""
Poisson 1D (Sezione 4.1 del paper)

    Delta u(x) = (4x^3 - 6x) e^{-x^2},   x in (0, 1)
    u(0) = 0,   u(1) = e^{-1}

Soluzione esatta:  u(x) = x e^{-x^2}

Setup PINN del paper:
  - N = 256 collocation points, Latin Hypercube, ricampionati ad ogni epoca
  - rete fully-connected, attivazione tanh
  - Adam: 15000 iterazioni, lr = 1e-4, poi rifinitura L-BFGS
  - valutazione su 512 punti equispaziati in [0, 1]
  - architetture provate nel paper (nodi dei layer nascosti):
      [1], [2], [5], [10], [20], [40],
      [5,5], [10,10], [20,20], [40,40],
      [5,5,5], [10,10,10], [20,20,20], [40,40,40]

Esempio:  python poisson_1d.py --arch 20 20 20
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
from deepxde import backend as bkd

from common import (get_parser, apply_fast, train_pinn, timed_predict,
                    rel_l2, save_run)


def exact(x):
    return x * np.exp(-x ** 2)


def pde(x, u):
    """Residuo: u_xx - (4x^3 - 6x) e^{-x^2}."""
    u_xx = dde.grad.hessian(u, x)
    f = (4 * x ** 3 - 6 * x) * bkd.exp(-x ** 2)
    return u_xx - f


def main():
    args = get_parser(__doc__, default_arch=[20, 20, 20],
                      default_adam=15000, default_lr=1e-4).parse_args()
    apply_fast(args)
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Interval(0, 1)
    # La funzione esatta fornisce direttamente i valori al bordo:
    # u(0) = 0 e u(1) = e^{-1}
    bc = dde.icbc.DirichletBC(geom, exact, lambda x, on_boundary: on_boundary)

    data = dde.data.PDE(geom, pde, bc, num_domain=256, num_boundary=2,
                        train_distribution=args.dist,
                        solution=exact, num_test=512)

    net = dde.nn.FNN([1] + args.arch + [1], "tanh", "Glorot normal")

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period, metrics=["l2 relative error"])

    # --- Valutazione su 512 punti (come nel paper) --------------------------
    X = np.linspace(0, 1, 512)[:, None]
    y_pred, t_eval = timed_predict(model, X)
    y_true = exact(X)
    err = rel_l2(y_pred, y_true)

    name = "poisson_1d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"l2_relative": err},
             X, y_pred, ["x", "u_pinn", "u_exact"], y_true,
             losshistory, train_state)

    # --- Grafico -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(X, y_true, "k-", label="esatta")
    axes[0].plot(X, y_pred, "r--", label="PINN")
    axes[0].set(xlabel="x", ylabel="u", title="Poisson 1D")
    axes[0].legend()
    axes[1].semilogy(X, np.abs(y_pred - y_true) + 1e-16)
    axes[1].set(xlabel="x", ylabel="|u_PINN - u_esatta|",
                title=f"errore L2 rel. = {err:.2e}")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
