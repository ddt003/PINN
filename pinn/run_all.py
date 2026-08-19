"""
Esegue in sequenza tutti i 6 casi del paper, ciascuno con la sua
architettura di default (una di quelle usate nel paper).

Uso:
    python run_all.py            # training completo (lungo! meglio su GPU)
    python run_all.py --fast     # smoke test: verifica solo che tutto giri

Per lo studio completo del paper (piu' architetture per caso) lanciare i
singoli script in un ciclo, ad esempio:

    for A in "20" "40" "20 20" "40 40" "20 20 20" "40 40 40"; do
        python poisson_1d.py --arch $A
    done
"""
import subprocess
import sys

CASES = [
    "poisson_1d.py",
    "poisson_2d.py",
    "poisson_3d.py",
    "allen_cahn_1d.py",
    "schrodinger_1d.py",
    "schrodinger_2d.py",
]


def main():
    extra = sys.argv[1:]  # es. --fast
    failed = []
    for case in CASES:
        print(f"\n{'=' * 60}\n>>> {case} {' '.join(extra)}\n{'=' * 60}")
        ret = subprocess.run([sys.executable, case, *extra])
        if ret.returncode != 0:
            failed.append(case)
    print("\n" + "=" * 60)
    if failed:
        print("FALLITI:", ", ".join(failed))
        sys.exit(1)
    print("Tutti i casi completati. Risultati nella cartella 'results/'.")


if __name__ == "__main__":
    main()
