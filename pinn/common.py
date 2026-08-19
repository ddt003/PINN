"""
Utility comuni per gli esperimenti PINN con DeepXDE.

Progetto NAML - riproduzione del confronto PINN vs FEM di:
    T. G. Grossmann, U. J. Komorowska, J. Latz, C.-B. Schoenlieb,
    "Can Physics-Informed Neural Networks beat the Finite Element Method?"
    arXiv:2302.04107 (2023)

Ogni script esporta:
  - <caso>_pred.csv   : soluzione PINN valutata su una griglia regolare
                        (stessa griglia da usare per il FEM nel confronto)
  - <caso>_info.json  : architettura, tempo di training, tempo di
                        valutazione, errori
  - <caso>.png        : grafico della soluzione
  - loss.dat / train.dat : storia della loss (salvata da DeepXDE)
"""
import argparse
import json
import os
import time

import numpy as np
import deepxde as dde


def get_parser(description, default_arch, default_adam, default_lr):
    """Argomenti da riga di comando comuni a tutti i casi."""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--arch", type=int, nargs="+", default=default_arch,
        help="Nodi dei layer nascosti, es. --arch 60 60 60 (default: %(default)s)",
    )
    p.add_argument("--adam-iters", type=int, default=default_adam,
                   help="Iterazioni Adam (default: %(default)s, come nel paper)")
    p.add_argument("--lr", type=float, default=default_lr,
                   help="Learning rate Adam (default: %(default)s, come nel paper)")
    p.add_argument("--lbfgs-iters", type=int, default=15000,
                   help="Iterazioni massime L-BFGS di rifinitura")
    p.add_argument("--no-lbfgs", action="store_true",
                   help="Salta la rifinitura L-BFGS")
    p.add_argument("--fast", action="store_true",
                   help="Training molto ridotto: solo per verificare che il "
                        "codice giri, i risultati NON sono validi")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--outdir", default="results")
    p.add_argument(
        "--dist", default="Hammersley", choices=["Hammersley", "LHS", "pseudo"],
        help="Distribuzione dei collocation points. Il paper usa LHS, ma "
             "l'implementazione LHS di scikit-optimize e' molto lenta con "
             "molti punti; Hammersley (quasi-random) e' equivalente in "
             "pratica e assai piu' veloce (default: %(default)s)",
    )
    p.add_argument(
        "--resample-period", type=int, default=100,
        help="Ogni quante iterazioni ricampionare i collocation points. "
             "Il paper ricampiona ad ogni epoca (=1), qui il default e' 100 "
             "per contenere i tempi; 0 = mai (default: %(default)s)",
    )
    p.add_argument(
        "--bc", default="soft", choices=["soft", "hard"],
        help="Come imporre le condizioni al bordo (slide 22 di PINNs.pdf, "
             "'Hard vs Soft Boundary Conditions'): "
             "'soft' (default, = paper Grossmann et al.) le impone come "
             "termine di loss extra (dde.icbc.DirichletBC); "
             "'hard' le incorpora nell'architettura della rete tramite un "
             "cambio di variabile che le soddisfa esattamente per costruzione "
             "(vedi common.hard_bc_transform_interval). Con --bc hard non c'e' "
             "nessun peso di loss da bilanciare per il bordo.",
    )
    return p


def hard_bc_transform_interval(u0, u1):
    """Transform di output per DeepXDE che impone in modo HARD le condizioni
    di Dirichlet u(0)=u0, u(1)=u1 su un intervallo (0,1) (slide 22 di
    PINNs.pdf):

        u_hat(x) = (1-x) u0 + x u1 + x(1-x) N(x;theta)

    Per costruzione u_hat(0)=u0 e u_hat(1)=u1 qualunque siano i pesi theta
    della rete N: le condizioni al bordo sono soddisfatte ESATTAMENTE, non
    approssimativamente, e non richiedono alcun termine di loss (ne' quindi
    alcun peso da bilanciare rispetto al residuo della PDE).

    Uso:
        net.apply_output_transform(hard_bc_transform_interval(u0, u1))
    """
    def transform(x, y):
        x0 = x[:, 0:1]
        return (1 - x0) * u0 + x0 * u1 + x0 * (1 - x0) * y
    return transform


def apply_fast(args, adam=1000, lbfgs=200):
    """Riduce le iterazioni in modalita' --fast (senza aumentare eventuali
    valori piu' bassi passati esplicitamente da riga di comando)."""
    if args.fast:
        args.adam_iters = min(args.adam_iters, adam)
        args.lbfgs_iters = min(args.lbfgs_iters, lbfgs)
    return args


def train_pinn(data, net, lr, adam_iters, loss_weights=None, pretrain=None,
               use_lbfgs=True, lbfgs_maxiter=15000, resample_period=100,
               metrics=None):
    """Training in stile paper: [pre-training] -> Adam -> L-BFGS, con timing.

    pretrain: tupla (iterazioni, lr, loss_weights) eseguita PRIMA del training
        completo. Usata per Allen-Cahn, dove il paper allena prima la sola
        condizione iniziale (7000 iterazioni) per stabilizzare l'ottimizzazione.

    Ritorna (model, losshistory, train_state, tempo_training_sec).
    """
    model = dde.Model(data, net)
    callbacks = []
    if resample_period and resample_period > 0:
        # Il paper campiona nuovi collocation points (Latin Hypercube) ad ogni
        # epoca; PDEPointResampler riproduce questo comportamento.
        callbacks.append(dde.callbacks.PDEPointResampler(period=resample_period))

    t0 = time.perf_counter()

    if pretrain is not None:
        it_pre, lr_pre, w_pre = pretrain
        model.compile("adam", lr=lr_pre, loss_weights=w_pre, metrics=metrics)
        model.train(iterations=it_pre, display_every=1000)

    model.compile("adam", lr=lr, loss_weights=loss_weights, metrics=metrics)
    losshistory, train_state = model.train(
        iterations=adam_iters, callbacks=callbacks, display_every=1000
    )

    if use_lbfgs and lbfgs_maxiter > 0:
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_maxiter)
        model.compile("L-BFGS", loss_weights=loss_weights, metrics=metrics)
        losshistory, train_state = model.train(display_every=1000)

    train_time = time.perf_counter() - t0
    return model, losshistory, train_state, train_time


def timed_predict(model, X, batch_size=100_000):
    """Valuta la rete su X misurando il tempo (a blocchi per griglie grandi)."""
    t0 = time.perf_counter()
    parts = [model.predict(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    y = np.vstack(parts)
    return y, time.perf_counter() - t0


def rel_l2(y_pred, y_true):
    """Errore L2 relativo, la metrica usata nel paper."""
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def save_run(outdir, name, arch, times, errors, X, y_pred, columns, y_true=None,
             losshistory=None, train_state=None, extra_info=None):
    """Salva CSV con la predizione su griglia + JSON con tempi ed errori.

    extra_info: dict opzionale (es. {"bc_mode": "hard", "method": "..."})
    unito al JSON di output, senza toccare lo schema base.
    """
    os.makedirs(outdir, exist_ok=True)

    blocks = [X, y_pred]
    if y_true is not None:
        blocks.append(y_true)
    table = np.hstack(blocks)
    csv_path = os.path.join(outdir, f"{name}_pred.csv")
    np.savetxt(csv_path, table, delimiter=",", header=",".join(columns), comments="")

    info = {"case": name, "architecture": list(arch),
            "times_sec": times, "errors": errors}
    if extra_info:
        info.update(extra_info)
    with open(os.path.join(outdir, f"{name}_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    if losshistory is not None and train_state is not None:
        dde.utils.saveplot(losshistory, train_state, issave=True, isplot=False,
                           output_dir=outdir)

    print(f"\n=== {name} ===")
    print(json.dumps(info, indent=2))
    print(f"Predizione salvata in: {csv_path}")
    return csv_path
