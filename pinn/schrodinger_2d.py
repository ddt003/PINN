"""
Schroedinger semilineare 2D (Sezione 6.2 del paper)

    i h_t = -0.5 Delta h - |h|^2 h,   (x,y) in [-5,5]^2, t in [0, pi/2]
    h(0, x, y) = sech(x) + 0.5 sech(y-2) + 0.5 sech(y+2)
    bordo periodico in x e in y (valori e derivate prime)

h e' complessa: rete con 2 output, h = u + i v. Residui:

    f_u = u_t + 0.5 (v_xx + v_yy) + (u^2 + v^2) v
    f_v = v_t - 0.5 (u_xx + u_yy) - (u^2 + v^2) u

Nessuna soluzione analitica: ground truth FEM su mesh fine (vedi paper).
Lo script esporta u, v, |h| sulla sezione t = pi/4 (quella mostrata in
Figura 12 del paper) e su una griglia spazio-tempo piu' rada.

Setup PINN del paper:
  - Nf = 5000 collocation, Ng = 100 sul bordo, Nh = 100 sull'istante iniziale
  - rete fully-connected tanh, output di dimensione 2
  - Adam: 50000 iterazioni, lr = 1e-3, poi L-BFGS
  - architetture del paper: [20,20,20], [100,100,100], [20]*4, [100]*4,
      [20]*5, [100]*5

Esempio:  python schrodinger_2d.py --arch 100 100 100 100
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

from pinn.common import (get_parser, apply_fast, train_pinn, timed_predict,
                    save_run)

L = 5.0
T_FINAL = np.pi / 2


def sech(z):
    return 1.0 / np.cosh(z)


def init_re(X):
    x, y = X[:, 0:1], X[:, 1:2]
    return sech(x) + 0.5 * sech(y - 2) + 0.5 * sech(y + 2)


def pde(X, h):
    """Residui separati in parte reale (u) e immaginaria (v)."""
    u, v = h[:, 0:1], h[:, 1:2]
    u_t = dde.grad.jacobian(h, X, i=0, j=2)
    v_t = dde.grad.jacobian(h, X, i=1, j=2)
    lap_u = (dde.grad.hessian(h, X, component=0, i=0, j=0)
             + dde.grad.hessian(h, X, component=0, i=1, j=1))
    lap_v = (dde.grad.hessian(h, X, component=1, i=0, j=0)
             + dde.grad.hessian(h, X, component=1, i=1, j=1))
    mod2 = u ** 2 + v ** 2
    f_u = u_t + 0.5 * lap_v + mod2 * v
    f_v = v_t - 0.5 * lap_u - mod2 * u
    return [f_u, f_v]


def on_bnd_x(X, on_boundary):   # facce x = -5 e x = 5
    return on_boundary and np.isclose(abs(X[0]), L)


def on_bnd_y(X, on_boundary):   # facce y = -5 e y = 5
    return on_boundary and np.isclose(abs(X[1]), L)


def main():
    args = get_parser(__doc__, default_arch=[100, 100, 100, 100],
                      default_adam=50000, default_lr=1e-3).parse_args()
    apply_fast(args, adam=2000, lbfgs=200)
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Rectangle([-L, -L], [L, L])
    timedomain = dde.geometry.TimeDomain(0, T_FINAL)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # periodicita' in x (component_x=0) e in y (component_x=1),
    # per entrambe le componenti della rete, in valore e derivata prima
    bcs = []
    for comp_x, on_b in [(0, on_bnd_x), (1, on_bnd_y)]:
        for der in (0, 1):
            for comp in (0, 1):
                bcs.append(dde.icbc.PeriodicBC(
                    geomtime, comp_x, on_b,
                    derivative_order=der, component=comp))

    on_init = lambda X, on_initial: on_initial
    ics = [
        dde.icbc.IC(geomtime, init_re, on_init, component=0),
        dde.icbc.IC(geomtime, lambda X: 0, on_init, component=1),
    ]

    # Nf=5000, Ng=100, Nh=100 come nel paper (ridotti in modalita' --fast)
    nf = 1000 if args.fast else 5000
    data = dde.data.TimePDE(geomtime, pde, bcs + ics,
                            num_domain=nf, num_boundary=100,
                            num_initial=100, train_distribution=args.dist)

    net = dde.nn.FNN([3] + args.arch + [2], "tanh", "Glorot normal")

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period)

    # --- Valutazione: sezione t = pi/4 su griglia nxn ------------------------
    n = 128
    xs = np.linspace(-L, L, n)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    t_slice = np.pi / 4
    X = np.stack([XX.ravel(), YY.ravel(),
                  np.full(n * n, t_slice)], axis=1)
    y_pred, t_eval = timed_predict(model, X)
    u, v = y_pred[:, 0:1], y_pred[:, 1:2]
    mod = np.sqrt(u ** 2 + v ** 2)

    name = "schrodinger_2d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"nota": "errore da calcolare vs FEM su mesh fine "
                      "(nessuna soluzione analitica); sezione a t=pi/4"},
             X, np.hstack([u, v, mod]),
             ["x", "y", "t", "u_re", "u_im", "abs_h"],
             losshistory=losshistory, train_state=train_state)

    # --- Grafico: |h| e condizione iniziale ----------------------------------
    H = mod.reshape(n, n)
    H0 = init_re(np.stack([XX.ravel(), YY.ravel()], axis=1)).reshape(n, n)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im0 = axes[0].pcolormesh(XX, YY, np.abs(H0), shading="auto")
    axes[0].set(xlabel="x", ylabel="y", title="|h(0,x,y)| (cond. iniziale)")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(XX, YY, H, shading="auto")
    axes[1].set(xlabel="x", ylabel="y", title="PINN |h(pi/4,x,y)|")
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
