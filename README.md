# PINN con DeepXDE — tutti i casi del paper

Implementazione con [DeepXDE](https://github.com/lululxvi/deepxde) della parte
PINN del confronto in:

> T. G. Grossmann, U. J. Komorowska, J. Latz, C.-B. Schönlieb,
> *"Can Physics-Informed Neural Networks beat the Finite Element Method?"*,
> arXiv:2302.04107 (2023).

## I 6 casi del paper

| Script | PDE | Dominio | Ground truth | Setup paper (Adam) |
|---|---|---|---|---|
| `poisson_1d.py` | Δu = (4x³−6x)e^(−x²), Dirichlet | (0,1) | esatta: x·e^(−x²) | 15000 it, lr 1e−4, N=256 |
| `poisson_2d.py` | Δu = f, BC miste (Neumann + Dirichlet) | (0,1)² | esatta: x²(x−1)²y(y−1)² | 20000 it, lr 1e−3, Nf=2000, Ng=250 |
| `poisson_3d.py` | Δu = −3π² sin(πx)sin(πy)sin(πz), Dirichlet 0 | (0,1)³ | esatta: sin(πx)sin(πy)sin(πz) | 20000 it, lr 1e−3, Nf=1000 |
| `allen_cahn_1d.py` | u_t = εu_xx − (2/ε)u(1−u)(1−2u), ε=0.01, periodica | (0,1)×(0,0.05] | FEM mesh fine | pre-train IC 7000 it + 50000 it, lr 1e−4, peso IC ×1000 |
| `schrodinger_1d.py` | i h_t = −0.5h_xx − \|h\|²h, periodica | (−5,5)×(0,π/2] | FEM mesh fine | 50000 it, lr 1e−4, Nf=20000 |
| `schrodinger_2d.py` | i h_t = −0.5Δh − \|h\|²h, periodica in x e y | (−5,5)²×(0,π/2] | FEM mesh fine | 50000 it, lr 1e−3, Nf=5000 |

In tutti i casi: rete fully-connected con attivazione **tanh**, collocation
points campionati con **Latin Hypercube**, training **Adam + rifinitura L-BFGS**
(struttura identica al paper). Per Schrödinger la rete ha 2 output
(parte reale e immaginaria).

## Installazione

```bash
pip install -r requirements.txt
```

Serve un backend per DeepXDE (uno tra pytorch / tensorflow / paddle):

```bash
# PyTorch (consigliato) — CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu
# selezione del backend:
export DDE_BACKEND=pytorch        # oppure: tensorflow
```

Su Windows (PowerShell): `$env:DDE_BACKEND="pytorch"`.

## Uso

```bash
# smoke test: verifica che tutto giri (pochi secondi/minuti, risultati NON validi)
python run_all.py --fast

# singolo caso col setup del paper (default: una delle architetture del paper)
python poisson_1d.py
python allen_cahn_1d.py --arch 100 100 100 100

# architettura diversa (nodi dei layer nascosti)
python poisson_2d.py --arch 120 120 120 120 120
```

Le architetture provate nel paper per ciascun caso sono elencate nel
docstring in testa a ogni script: per riprodurre le curve tempo-vs-errore
basta lanciare lo stesso script in un ciclo su più architetture.

Ogni run salva in `results/`:

- `<caso>_pred.csv` — soluzione PINN sulla griglia di valutazione
  (**stessa griglia** su cui esportare la soluzione FEM per il confronto);
- `<caso>_info.json` — architettura, tempo di training, tempo di
  valutazione, errore L² relativo (dove esiste la soluzione esatta);
- `<caso>.png` — grafico della soluzione;
- `loss.dat`, `train.dat` — storia della loss.

## Note e differenze rispetto al paper

- **Campionamento**: il paper usa Latin Hypercube; l'implementazione LHS
  usata da DeepXDE (scikit-optimize, criterio maximin) è però lentissima con
  molti punti (minuti per Nf=20000). Il default qui è **Hammersley**
  (quasi-random, copertura equivalente o migliore); `--dist LHS` per fedeltà
  totale al paper.
- **Ricampionamento**: il paper ricampiona i collocation points ad ogni
  epoca; qui il default è ogni 100 iterazioni (`--resample-period 1` per
  fedeltà totale, più lento; con Hammersley il ricampionamento è ininfluente
  perché la sequenza è deterministica).
- **Griglie di valutazione**: ridotte rispetto al paper dove servirebbero
  milioni di punti (es. 3D: default 64³ invece di 150³, `--eval-n 150` per
  replicare).
- **Allen–Cahn e Schrödinger** non hanno soluzione analitica: il paper usa
  come riferimento un FEM su mesh finissima. I CSV esportati servono proprio
  al confronto con la parte FEM del progetto.
- **Tempi**: con i setup completi del paper il training richiede ore
  (nel paper: GPU). Le architetture grandi di Allen–Cahn ([500]*6) hanno
  senso solo su GPU.
- I casi Allen–Cahn con ε=0.001 falliscono anche nel paper (Sez. 5):
  è un risultato, non un bug.
