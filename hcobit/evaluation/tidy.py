"""Collapse an evaluation's results.csv into one row per operating point.

The tidy files (frontier_<dataset>.csv, mauve_<dataset>.csv, real_text.csv) have the
format of the paper's own results in results/, so scripts/report.py reads both.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

_CLCH = re.compile(r"-clch([0-9.]+)")
_QC = re.compile(r"-qc([0-9.]+)_([0-9.]+)")
FRONTIER_COLS = ["arm", "checkpoint", "nfe", "gamma", "s_noise", "gamma_cluster", "cluster_window", "entropy", "gen_ppl"]
MAUVE_COLS = FRONTIER_COLS + ["mauve", "mauve_std"]


def _points(raw_csv: Path):
    by_tag = defaultdict(dict)
    with open(raw_csv) as f:
        for r in csv.DictReader(f):
            if r.get("value") in (None, ""):
                continue
            by_tag[r["tag"]].setdefault("_row", r)
            by_tag[r["tag"]][r["metric"]] = float(r["value"])
    return by_tag


def _sampler_fields(row):
    nfe = int(row["nfe"])
    n = max(1, nfe - 1)
    m, q = _CLCH.search(row["tag"]), _QC.search(row["tag"])
    return dict(
        nfe=nfe,
        gamma=round(float(row["s_churn"]) / n, 4),
        s_noise=float(row["s_noise"]),
        gamma_cluster=round(float(m.group(1)) / n, 4) if m else "",
        cluster_window=f"[{q.group(1)},{q.group(2)}]" if q else "",
    )


def _merge(path: Path, cols, new_rows, key):
    rows = []
    if path.exists():
        with open(path) as f:
            rows = list(csv.DictReader(f))
    keys = {key(r) for r in new_rows}
    rows = [r for r in rows if key(r) not in keys] + new_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def export(raw_csv: Path, *, dataset: str, arm: str, results_dir: Path, kind: str) -> None:
    """kind = 'frontier' or 'mauve'."""
    pts = _points(raw_csv)
    rows, real_h, real_p = [], [], []
    for tag, m in pts.items():
        if "real_full_token_unigram_entropy" in m:
            real_h.append(m["real_full_token_unigram_entropy"])
        if "real_full_external_ppl" in m:
            real_p.append(m["real_full_external_ppl"])
        if tag == "REAL" or "gen_full_external_ppl" not in m or "gen_full_token_unigram_entropy" not in m:
            continue
        if kind == "mauve" and "full_mauve" not in m:
            continue
        row = dict(arm=arm, checkpoint=m["_row"]["checkpoint"], **_sampler_fields(m["_row"]),
                   entropy=round(m["gen_full_token_unigram_entropy"], 5), gen_ppl=round(m["gen_full_external_ppl"], 3))
        if kind == "mauve":
            row.update(mauve=round(m["full_mauve"], 5), mauve_std=round(m.get("full_mauve_std", float("nan")), 5))
        rows.append(row)

    def key(r):
        return (r["arm"], str(r["nfe"]), f"{float(r['gamma']):g}")

    cols = MAUVE_COLS if kind == "mauve" else FRONTIER_COLS
    _merge(Path(results_dir) / f"{kind}_{dataset}.csv", cols, rows, key)
    if real_h and real_p:
        real = [dict(dataset=dataset, arm=arm, entropy=round(sum(real_h) / len(real_h), 5),
                     gen_ppl=round(sum(real_p) / len(real_p), 3))]
        _merge(Path(results_dir) / "real_text.csv", ["dataset", "arm", "entropy", "gen_ppl"], real,
               lambda r: (r["dataset"], r["arm"]))
    print(f"{len(rows)} operating point(s) -> {Path(results_dir) / f'{kind}_{dataset}.csv'}")
