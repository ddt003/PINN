"""
Allen-Cahn 1D (Sezione 5 del paper)

    u_t = eps * u_xx - (2/eps) * u (1-u) (1-2u),   x in [0,1], t in [0, 0.05]
    u(t,0) = u(t,1)                                (bordo periodico)
    u(0,x) = 0.25 sin(2 pi x) + 0.25 sin(16 pi x) + 0.5

con eps = 0.01. Non esiste soluzione analitica: il paper usa come ground
truth un FEM su mesh finissima (7993 nodi, dt = 1e-4/3). Questo script
esporta la predizione PINN su una griglia (t, x) regolare, da confrontare
con la soluzione FEM calcolata a parte.

Setup PINN del paper:
  - Nf = 20000 collocation nel dominio, Ng = 250 sul bordo,
    Nh = 500 sulla condizione iniziale (LHS, ricampionati)
  - loss pesata: termine della condizione iniziale moltiplicato per 1000
  - pre-training: Adam lr = 1e-4 per 7000 iterazioni SOLO sulla loss
    della condizione iniziale, poi Adam lr = 1e-4 per 50000 iterazioni
    sulla loss completa, infine L-BFGS
  - architetture del paper: da [20,20,20] a [100]*7 e [500]*6
    (con 20 nodi per layer il paper NON riesce ad approssimare la soluzione;
     servono almeno 100 nodi per layer)

Esempio:  python allen_cahn_1d.py --arch 100 100 100 100
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde

from pinn.common import (get_parser, apply_fast, train_pinn, timed_predict,
                    save_run)

EPS = 0.01
T_FINAL = 0.05


def init_cond(X):
    x = X[:, 0:1]
    return 0.25 * np.sin(2 * np.pi * x) + 0.25 * np.sin(16 * np.pi * x) + 0.5


def pde(X, u):
    """Residuo: u_t - eps u_xx + (2/eps) u (1-u) (1-2u)."""
    u_t = dde.grad.jacobian(u, X, i=0, j=1)
    u_xx = dde.grad.hessian(u, X, i=0, j=0)
    return u_t - EPS * u_xx + (2.0 / EPS) * u * (1 - u) * (1 - 2 * u)


def main():
    parser = get_parser(__doc__, default_arch=[100, 100, 100, 100],
                        default_adam=50000, default_lr=1e-4)
    parser.add_argument("--pretrain-iters", type=int, default=7000,
                        help="Iterazioni Adam sulla sola condizione iniziale")
    args = parser.parse_args()
    apply_fast(args, adam=2000, lbfgs=200)
    if args.fast:
        args.pretrain_iters = min(args.pretrain_iters, 500)
    dde.config.set_random_seed(args.seed)

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, T_FINAL)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    # bordo periodico: u(t,0) = u(t,1)
    bc = dde.icbc.PeriodicBC(geomtime, 0,
                             lambda X, on_boundary: on_boundary,
                             derivative_order=0, component=0)
    ic = dde.icbc.IC(geomtime, init_cond,
                     lambda X, on_initial: on_initial)

    # Nf=20000, Ng=250, Nh=500 come nel paper (ridotti in modalita' --fast)
    nf, ng, nh = (2000, 50, 100) if args.fast else (20000, 250, 500)
    data = dde.data.TimePDE(geomtime, pde, [bc, ic],
                            num_domain=nf, num_boundary=ng,
                            num_initial=nh, train_distribution=args.dist)

    net = dde.nn.FNN([2] + args.arch + [1], "tanh", "Glorot normal")

    # Pesi della loss nell'ordine [residuo PDE, BC periodica, IC]:
    # il paper pesa il termine della condizione iniziale con 1000.
    weights_full = [1, 1, 1000]
    weights_ic_only = [0, 0, 1]

    model, losshistory, train_state, t_train = train_pinn(
        data, net, args.lr, args.adam_iters,
        loss_weights=weights_full,
        pretrain=(args.pretrain_iters, args.lr, weights_ic_only),
        use_lbfgs=not args.no_lbfgs, lbfgs_maxiter=args.lbfgs_iters,
        resample_period=args.resample_period)

    # --- Valutazione su griglia (t, x) regolare -------------------------------
    # (stessa griglia da usare per esportare la soluzione FEM di riferimento)
    nx, nt = 512, 51
    xs = np.linspace(0, 1, nx)
    ts = np.linspace(0, T_FINAL, nt)
    XX, TT = np.meshgrid(xs, ts, indexing="ij")
    X = np.stack([XX.ravel(), TT.ravel()], axis=1)
    y_pred, t_eval = timed_predict(model, X)

    name = "allen_cahn_1d_" + "-".join(map(str, args.arch))
    save_run(args.outdir, name, args.arch,
             {"train": t_train, "eval": t_eval},
             {"nota": "errore da calcolare vs FEM su mesh fine "
                      "(nessuna soluzione analitica)"},
             X, y_pred, ["x", "t", "u_pinn"],
             losshistory=losshistory, train_state=train_state)

    # --- Grafico: heatmap + profili a t = 0, T/2, T ---------------------------
    U = y_pred.reshape(nx, nt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].pcolormesh(TT, XX, U, shading="auto")
    axes[0].set(xlabel="t", ylabel="x", title="Allen-Cahn 1D - PINN u(t,x)")
    fig.colorbar(im, ax=axes[0])
    for j, lbl in [(0, "t=0"), (nt // 2, "t=T/2"), (nt - 1, "t=T")]:
        axes[1].plot(xs, U[:, j], label=lbl)
    axes[1].plot(xs, init_cond(xs[:, None]), "k:", label="cond. iniziale")
    axes[1].set(xlabel="x", ylabel="u")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/{name}.png", dpi=150)
    print(f"Grafico salvato in: {args.outdir}/{name}.png")


if __name__ == "__main__":
    main()
# fine
