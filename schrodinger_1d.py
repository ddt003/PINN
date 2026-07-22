"""
Schroedinger semilineare 1D (Sezione 6.1 del paper, come in Raissi et al. 2019)

    i h_t = -0.5 h_xx - |h|^2 h,     x in [-5, 5], t in [0, pi/2]
    h(0, x) = 2 sech(x)
    h(t,-5) = h(t,5),   h_x(t,-5) = h_x(t,5)    (bordo periodico)

h e' complessa: la rete ha 2 output, h = u + i v (parte reale e immaginaria).
Residui (separando parte reale e immaginaria):

    f_u = u_t + 0.5 v_xx + (u^2 + v^2) v
    f_v = v_t - 0.5 u_xx - (u^2 + v^2) u

Non esiste soluzione analitica: il paper usa come ground truth un FEM su
mesh fine (7993 celle, dt = 1e-4/3). Lo script esporta u, v e |h| su una
griglia (t, x) regolare per il confronto.

Setup PINN del paper:
  - Nf = 20000 collocation, Ng = 50 sul bordo, Nh = 50 sull'istante iniziale
  - rete fully-connected tanh, output di dimensione 2
  - Adam: 50000 iterazioni, lr = 1e-4, poi L-BFGS
  - architetture del paper: [20,20,20], [100,100,100], [20]*4, [100]*4,
      [20]*5, [100]*5, [20]*6, [100]*6

Esempio:  python schrodinger_1d.py --arch 100 100 100 100
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

from common import (get_parser, apply_fast, train_pinn, timed_predict,
                    save_run)

X_MIN, X_MAX = -5.0, 5.0
T_FINAL = np.pi / 2


def pde(X, h):
    """Residui della PDE separati in parte reale (u) e immaginaria (v)."""
    u, v = h[:, 0:1], h[:, 1:2]
    u_t = dde.grad.jacobian(h, X, i=0, j=1)
    v_t = dde.grad.jacobian(h, X, i=1, j=1)
    u_xx = dde.grad.hessian(h, X, component=0, i=0, j=0)
    v_xx = dde.grad.hessian(h, X, component=1, i=0, j=0)
    mod2 = u ** 2 + v ** 2
    f_u = u_t + 0.5 * v_xx + mod2 * v
    f_v = v_t - 0.5 * u_xx - mod2 * u
    return [f_u, f_v]


def main():
    args = get_parser(__doc__, default_arch=[100, 100, 100, 100],
                      default_adam=50000, default_lr=1e-4).parse_args()
    apply_fast(args, adam=2000, lbfgs=200)
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Interval(X_MIN, X_MAX)
    timedomain = dde.geometry.TimeDomain(0, T_FINAL)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    on_b = lambda X, on_boundary: on_boundary
    # periodicita' di u e v e delle loro derivate prime in x
    bcs = [
        dde.icbc.PeriodicBC(geomtime, 0, on_b, derivative_order=0, component=0),
        dde.icbc.PeriodicBC(geomtime, 0, on_b, derivative_order=0, component=1),
        dde.icbc.PeriodicBC(geomtime, 0, on_b, derivative_order=1, component=0),
        dde.icbc.PeriodicBC(geomtime, 0, on_b, derivative_order=1, component=1),
    ]
    on_init = lambda X, on_initial: on_initial
    ics = [
        dde.icbc.IC(geomtime, lambda X: 2 / np.cosh(X[:, 0:1]), on_init,
                    component=0),
        dde.icbc.IC(geomtime, lambda X: 0, on_init, component=1),
    ]

    # Nf=20000, Ng=50, Nh=50 come nel paper (ridotti in modalita' --fast)
    nf = 2000 if args.fast else 20000
    data = dde.data.TimePDE(geomtime, pde, bcs + ics,
                            num_domain=nf, num_boundary=50,
                            num_initial=50, train_distribution=args.dist)

    net = dde.nn.FNN([2] + args.arch + [2], "tanh", "Glorot normal")

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period)

    # --- Valutazione su griglia (t, x) regolare -------------------------------
    nx, nt = 256, 101
    xs = np.linspace(X_MIN, X_MAX, nx)
    ts = np.linspace(0, T_FINAL, nt)
    XX, TT = np.meshgrid(xs, ts, indexing="ij")
    X = np.stack([XX.ravel(), TT.ravel()], axis=1)
    y_pred, t_eval = timed_predict(model, X)
    u, v = y_pred[:, 0:1], y_pred[:, 1:2]
    mod = np.sqrt(u ** 2 + v ** 2)

    name = "schrodinger_1d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"nota": "errore da calcolare vs FEM su mesh fine "
                      "(nessuna soluzione analitica)"},
             X, np.hstack([u, v, mod]),
             ["x", "t", "u_re", "u_im", "abs_h"],
             losshistory=losshistory, train_state=train_state)

    # --- Grafico: heatmap |h| + profili --------------------------------------
    H = mod.reshape(nx, nt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].pcolormesh(TT, XX, H, shading="auto")
    axes[0].set(xlabel="t", ylabel="x", title="Schroedinger 1D - PINN |h(t,x)|")
    fig.colorbar(im, ax=axes[0])
    for j, lbl in [(0, "t=0"), (nt // 2, "t=pi/4"), (nt - 1, "t=pi/2")]:
        axes[1].plot(xs, H[:, j], label=lbl)
    axes[1].set(xlabel="x", ylabel="|h|")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
