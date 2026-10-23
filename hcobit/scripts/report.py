"""Read the paper's numbers off the evaluation CSVs.

    python scripts/report.py                       # the paper's own results (results/)
    python scripts/report.py --results my_results  # your reproduction
    python scripts/report.py --plot frontiers.png  # also draw the GenPPL/entropy frontiers

Table 1   GenPPL at dataset entropy: the nearest measured point of each arm's Pareto
          frontier (over all NFE and churn values) within --tol nats of the data entropy.
Table 2   GSM8K accuracy: best churn at each NFE.
Table 11  MAUVE at fixed entropy levels: at each level, the best MAUVE over NFE among the
          points within 0.015 nats of it.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ENTROPY = {"lm1b": 4.343, "owt": 5.46}
#: The OWT H-CoBit sweep has no point within 0.01 nats of 5.46; its nearest is read instead.
TOL_OVERRIDE = {("owt", "hcobit"): 0.02}
ARMS = ["cobit", "hcobit", "cobit_snoise1.05", "hcobit_pms"]
MAUVE_LEVELS = (0.02, 0.0, -0.05)


def load(path):
    if not path.is_file():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("nfe",):
            r[k] = int(r[k])
        for k in ("gamma", "entropy", "gen_ppl", "mauve", "accuracy"):
            if r.get(k, "") != "":
                r[k] = float(r[k])
    return rows


def pareto(points):
    """Points not dominated by a more diverse AND lower-GenPPL point, by increasing entropy."""
    keep, best = [], float("inf")
    for p in sorted(points, key=lambda p: (-p["entropy"], p["gen_ppl"])):
        if p["gen_ppl"] < best:
            best = p["gen_ppl"]
            keep.append(p)
    return keep[::-1]


def at_entropy(points, target, tol):
    front = pareto(points)
    if not front:
        return None
    p = min(front, key=lambda p: abs(p["entropy"] - target))
    return p if abs(p["entropy"] - target) <= tol else None


def table1(results, tol):
    print("Table 1: GenPPL at dataset entropy")
    print(f"  {'arm':<18}" + "".join(f"{ds:>32}" for ds in DATA_ENTROPY))
    by = {ds: defaultdict(list) for ds in DATA_ENTROPY}
    for ds in DATA_ENTROPY:
        for r in load(results / f"frontier_{ds}.csv"):
            by[ds][r["arm"]].append(r)
    for arm in ARMS:
        cells = []
        for ds, h in DATA_ENTROPY.items():
            p = at_entropy(by[ds][arm], h, TOL_OVERRIDE.get((ds, arm), tol)) if by[ds][arm] else None
            cells.append("--" if p is None else
                         f"{p['gen_ppl']:.1f} (H={p['entropy']:.3f}, NFE {p['nfe']}, g={p['gamma']:g})")
        print(f"  {arm:<18}" + "".join(f"{c:>32}" for c in cells))


def table2(results):
    rows = [r for r in load(results / "gsm8k.csv") if r["gamma"] > 0]
    if not rows:
        return
    print("\nTable 2: GSM8K accuracy (best churn per NFE)")
    for nfe in sorted({r["nfe"] for r in rows}):
        b = max((r for r in rows if r["nfe"] == nfe), key=lambda r: r["accuracy"])
        print(f"  NFE {nfe:<5} {100 * b['accuracy']:.1f}%  (g={b['gamma']:g}, "
              f"{b['num_correct']}/{b['num_examples']})")


def table11(results):
    real = {(r["dataset"], r["arm"]): float(r["entropy"]) for r in csv.DictReader(open(results / "real_text.csv"))} \
        if (results / "real_text.csv").is_file() else {}
    print("\nTable 11: MAUVE at fixed entropy (best NFE)")
    for ds in DATA_ENTROPY:
        rows = load(results / f"mauve_{ds}.csv")
        if not rows:
            continue
        h_ref = real.get((ds, "cobit"), DATA_ENTROPY[ds])
        levels = [h_ref + d for d in MAUVE_LEVELS]
        print(f"  {ds:<15}" + "".join(f"{'H_data' + (f'{d:+.2f}' if d else ''):>18}" for d in MAUVE_LEVELS))
        for arm in ("cobit", "hcobit"):
            pts = [r for r in rows if r["arm"] == arm]
            cells = []
            for h in levels:
                best = None
                for nfe in {p["nfe"] for p in pts}:
                    c = min((p for p in pts if p["nfe"] == nfe), key=lambda p: abs(p["entropy"] - h))
                    if abs(c["entropy"] - h) <= 0.015 and (best is None or c["mauve"] > best["mauve"]):
                        best = c
                cells.append("--" if best is None else f"{best['mauve']:.3f} ({best['nfe']})")
            print(f"  {arm:<15}" + "".join(f"{c:>18}" for c in cells))


def plot(results, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (ds, h) in zip(axes, DATA_ENTROPY.items()):
        by = defaultdict(list)
        for r in load(results / f"frontier_{ds}.csv"):
            by[r["arm"]].append(r)
        for arm in ARMS:
            if by[arm]:
                f = pareto(by[arm])
                ax.plot([p["entropy"] for p in f], [p["gen_ppl"] for p in f], marker="o", ms=3, label=arm)
        ax.axvline(h, ls="--", c="grey", lw=0.8)
        ax.set(title=ds, xlabel="token unigram entropy H (nats)", ylabel="GenPPL (GPT-2 Large)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    table1(args.results, args.tol)
    table2(args.results)
    table11(args.results)
    if args.plot:
        plot(args.results, args.plot)


if __name__ == "__main__":
    main()
