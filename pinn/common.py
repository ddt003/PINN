"""
utility for PINNs

exported files:
  - <case>_pred.csv   : pinn solution evaluated on a grid of 512 points (as in the paper)
                        which is the same grid used for the fem solution
  - <case>_info.json  : architecture, training time, evaluation time,
                        l2 relative error, ...
  - <case>.png        : solution graph (pinn vs exact) + error graph
  - loss.dat / train.dat : loss history and collocation points on which the training was performed
"""
import argparse
import json
import os
import time

import numpy as np
import deepxde as dde


def get_parser(description, default_arch, default_adam, default_lr):
    """command line parser for pinn examples"""
    p = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--arch", type=int, nargs="+", default=default_arch,
        help="hidden layer sizes, es. --arch 60 60 60 (default: %(default)s)",
    )
    p.add_argument("--adam-iters", type=int, default=default_adam,
                   help="Adam iterations (default: %(default)s)")
    p.add_argument("--lr", type=float, default=default_lr,
                   help="Learning rate (default: %(default)s)")
    p.add_argument("--lbfgs-iters", type=int, default=15000,
                   help="max iterations for L-BFGS refinement (default: %(default)s)")
    p.add_argument("--no-lbfgs", action="store_true",
                   help="no L-BFGS refinement after Adam (default: %(default)s)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--outdir", default="results")
    p.add_argument(
        "--dist", default="Hammersley", choices=["Hammersley", "LHS", "pseudo"],
        help="Collocation points distribution. The paper uses LHS, but the LHS implementation in scikit-optimize is very slow with many points;" \
        "Hammersley (quasi-random) is equivalent in practice and much faster (default: %(default)s)",
    )
    p.add_argument(
        "--resample-period", type=int, default=100,
        help="how often to resample collocation points (in epochs). "
             "The paper resamples at every epoch (=1), here the default is 100 "
             "to contain the times; 0 = never (default: %(default)s)",
    )
    p.add_argument(
        "--bc", default="soft", choices=["soft", "hard"],
        help="how boundary conditions are imposed: "
             "'soft' (default, = paper) impose them as an extra loss term (dde.icbc.DirichletBC); "
             "'hard' directly incorporates them in the network architecture via a "
             "variable change that satisfies them exactly by construction, so no loss term is needed."
    )
    return p


def train_pinn(data, net, lr, adam_iters, loss_weights=None, pretrain=None,
               use_lbfgs=True, lbfgs_maxiter=15000, resample_period=100,
               metrics=None):
    """Training: [pre-training] -> Adam -> L-BFGS, con timing.

    pretrain: tuple (iterations, lr, loss_weights) executed BEFORE the full training.
            Used for Allen-Cahn, where the paper first trains only the initial condition (7000 iterations) to stabilize optimization.
    """
    model = dde.Model(data, net)
    callbacks = []
    if resample_period and resample_period > 0:
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
    """Evaluate net on X, measuring the time (in blocks for large grids)."""
    t0 = time.perf_counter()
    parts = [model.predict(X[i:i + batch_size]) for i in range(0, len(X), batch_size)]
    y = np.vstack(parts)
    return y, time.perf_counter() - t0


def rel_l2(y_pred, y_true):
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def save_run(outdir, name, arch, times, errors, X, y_pred, columns, y_true=None,
             losshistory=None, train_state=None, extra_info=None):
    """Save results to CSV and JSON."""
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
    print(f"Prediction saved in: {csv_path}")
    return csv_path
