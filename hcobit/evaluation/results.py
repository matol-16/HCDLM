"""Output paths, results.csv / results.jsonl appends, and resume bookkeeping."""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _resolve_eval_dirs(cfg):
    out_dir = Path(getattr(cfg.evaluation, "out_dir", f"runs/{cfg.experiment}/evaluation")).expanduser()
    samples_dir = Path(getattr(cfg.evaluation, "samples_dir", str(out_dir / "samples"))).expanduser()
    results_csv = Path(getattr(cfg.evaluation, "results_csv", str(out_dir / "results.csv"))).expanduser()
    return out_dir, samples_dir, results_csv


def _resolve_results_paths(cfg) -> Tuple[Path, Path]:
    _, _, results_csv = _resolve_eval_dirs(cfg)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    return results_csv, results_csv.with_suffix(".jsonl")


def _shared_cache_root(cfg, metric_cfg) -> Path:
    root = getattr(metric_cfg, "shared_cache_dir", None) or getattr(cfg.evaluation, "shared_text_cache_dir", None)
    return Path(root) if root else Path(cfg.evaluation.out_dir) / "shared_text_cache"


def _resolve_effective_stochastic_fields(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The churn settings of a spec, as recorded in the cache key and the results rows."""
    if not bool(spec.get("stochastic_enabled", False)):
        return {"stochastic_enabled": False}
    out = dict(
        stochastic_enabled=True,
        s_churn=float(spec.get("s_churn", 0.0)),
        s_noise=float(spec.get("s_noise", 1.0)),
        window_mode=str(spec.get("window_mode", "entropy_cdf")),
        entropy_quantile_lo=spec.get("entropy_quantile_lo"),
        entropy_quantile_hi=spec.get("entropy_quantile_hi"),
        s_tmin=spec.get("s_tmin"),
        s_tmax=spec.get("s_tmax"),
        entropy_fallback=str(spec.get("entropy_fallback", "deterministic")),
    )
    for k in ("s_churn_cluster", "entropy_quantile_lo_cluster", "entropy_quantile_hi_cluster"):
        if spec.get(k) is not None:
            out[k] = spec[k]
    return out


def spec_cache_fields(spec: Dict[str, Any], *, sigma_max, use_ema: bool) -> Dict[str, Any]:
    """Fields of a spec that enter the text-cache key and its results rows."""
    return dict(
        sampler_name=str(spec["sampler_name"]),
        terminal_sigma=float(spec["terminal_sigma"]),
        guidance_scale=0.0,
        sc_refresh_mode="carry",
        sigma_max=sigma_max,
        use_ema=bool(use_ema),
        ati_eta=0.0,
        **_resolve_effective_stochastic_fields(spec),
    )


def _generation_meta_fields(meta: Dict[str, object]) -> Dict[str, object]:
    """Speed statistics of a cached generation, as results-row fields."""
    keys = (
        "generation_wall_time_sec", "world_size", "sequence_len_tokens", "sequence_len_model",
        "full_lm_tokens", "generated_lm_tokens", "full_model_positions", "generated_model_positions",
        "samples_per_sec", "full_lm_tokens_per_sec", "generated_lm_tokens_per_sec", "lm_tokens_per_sec",
        "full_model_positions_per_sec", "generated_model_positions_per_sec", "model_positions_per_sec",
    )
    out = {k: meta[k] for k in keys if isinstance(meta.get(k), (int, float))}
    if isinstance(meta.get("num_samples"), (int, float)):
        out["generated_num_samples"] = meta["num_samples"]
    return out


def _append_results_online(cfg, rows: List[Dict[str, Any]]) -> None:
    """Merge rows into results.csv / results.jsonl right away, so a partial run keeps them.

    A row replaces nothing: rows whose (checkpoint, tag, metric) is already present are
    skipped. The CSV is rewritten with the union of all columns.
    """
    if not rows:
        return
    results_csv, results_jsonl = _resolve_results_paths(cfg)
    existing: List[Dict[str, Any]] = []
    if results_csv.exists() and results_csv.stat().st_size > 0:
        with results_csv.open("r", newline="") as f:
            existing = list(csv.DictReader(f))

    def key(r):
        return (str(r.get("checkpoint", "")), str(r.get("tag", "")), str(r.get("metric", "")))

    seen = {key(r) for r in existing}
    new = []
    for r in rows:
        if key(r) not in seen:
            seen.add(key(r))
            new.append(r)
    merged = existing + new
    base = ["timestamp_utc", "experiment", "framework", "dataset", "checkpoint", "metric", "split", "value", "details"]
    fieldnames = base + sorted({k for r in merged for k in r if k not in base})
    tmp = results_csv.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(merged)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(results_csv)
    with results_jsonl.open("a") as f:
        for r in new:
            f.write(json.dumps(r) + "\n")


def _load_scored_keys(cfg, metric_prefix: str) -> set:
    """(checkpoint, tag) of the specs already scored in results.csv, to resume a run."""
    _, _, results_csv = _resolve_eval_dirs(cfg)
    if not results_csv.exists() or results_csv.stat().st_size == 0:
        return set()
    with results_csv.open("r", newline="") as f:
        return {
            (row.get("checkpoint", ""), row.get("tag", ""))
            for row in csv.DictReader(f)
            if str(row.get("metric", "")).startswith(metric_prefix)
        }
