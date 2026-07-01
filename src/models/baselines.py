"""Classical imputation baselines + evaluation runner.

These are the floor every deep model must beat, scored through the Stage-A
hide-and-recover harness (src/evaluation/masking.py). Targets scored: WVHT and
APD, in physical units.

Each baseline takes the masked model-input tensor (NaN at hidden+missing cells,
normalised space) and returns a FULL imputed tensor (normalised space):

  mean_fill        per-station-per-feature training-set mean.
  forward_fill     last-observation-carried-forward per station/feature, then
                   back-fill leading gaps.
  linear_interp    temporal linear interpolation per station/feature, hold edges.
  spatial_idw      inverse-distance-weighted mean of the SAME feature at the k
                   nearest OBSERVED neighbour stations at that timestamp, using
                   haversine distances from the spatial graph. THE key baseline:
                   it tests whether naive spatial info recovers a blacked-out
                   station.
  spatial_temporal mean of linear_interp and spatial_idw (simple hybrid).

The runner evaluates every baseline against the saved mask suite, aggregates
across the 3 seeds (mean +/- std) per mechanism/config, writes
reports/baseline_imputation_results.csv, prints the headline station-outage
table, and explicitly flags the spatial (IDW) vs temporal comparison.

Run:  ``python -m src.models.baselines``
Importable; nothing runs on import.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.download import load_config
from ..evaluation.masking import (
    TARGET_FEATURES, load_masks, make_model_input, score,
)
from ..features.preprocess import load_meta

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_NPZ = PROCESSED_DIR / "wave_tensor_model.npz"
GRAPH_NPZ = PROCESSED_DIR / "wave_graph.npz"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
RESULTS_CSV = PROJECT_ROOT / "reports" / "baseline_imputation_results.csv"


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_truth() -> np.ndarray:
    """The original normalised tensor (ground truth for scoring)."""
    return np.load(MODEL_NPZ)["tensor"]


def load_distance() -> np.ndarray:
    """The [S, S] haversine distance matrix, in the tensor's station order."""
    import json
    z = np.load(GRAPH_NPZ)
    graph_order = [str(s) for s in z["station_order"]]
    axes_order = [str(s) for s in json.load(open(AXES_JSON))["station_ids"]]
    if graph_order != axes_order:
        raise RuntimeError("graph station order does not match tensor station order")
    return z["distance_km"].astype(np.float64)


# ---------------------------------------------------------------------------
# temporal helpers
# ---------------------------------------------------------------------------
def _ffill_2d(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs down axis 0 of a [T, S] array (leading NaNs untouched)."""
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def _bfill_2d(a: np.ndarray) -> np.ndarray:
    """Back-fill NaNs down axis 0 of a [T, S] array."""
    return _ffill_2d(a[::-1])[::-1]


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def mean_fill(X: np.ndarray, train_end: int) -> np.ndarray:
    """Fill every NaN with its station+feature mean over observed TRAINING cells."""
    out = X.copy()
    _, _, F = X.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        per_sf = np.nanmean(X[:train_end], axis=0)                 # [S, F]
        global_f = np.nanmean(X[:train_end].reshape(-1, F), axis=0)  # [F]
    per_sf = np.where(np.isnan(per_sf), global_f[None, :], per_sf)
    fill = np.broadcast_to(per_sf[None, :, :], X.shape)
    nan = np.isnan(out)
    out[nan] = fill[nan]
    return out


def forward_fill(X: np.ndarray) -> np.ndarray:
    """LOCF per station/feature along time, then back-fill leading gaps."""
    out = X.copy()
    for f in range(X.shape[2]):
        out[:, :, f] = _bfill_2d(_ffill_2d(out[:, :, f]))
    out[np.isnan(out)] = 0.0  # all-NaN column safety (0 == normalised mean)
    return out


def linear_interp(X: np.ndarray) -> np.ndarray:
    """Temporal linear interpolation per station/feature; hold-last at edges."""
    out = X.copy()
    for f in range(X.shape[2]):
        df = pd.DataFrame(out[:, :, f]).interpolate(
            method="linear", axis=0, limit_direction="both")
        out[:, :, f] = df.ffill().bfill().to_numpy()
    out[np.isnan(out)] = 0.0
    return out


def spatial_idw(X: np.ndarray, distance: np.ndarray, k: int = 8, power: float = 2.0) -> np.ndarray:
    """Inverse-distance-weighted fill from the k nearest OBSERVED neighbours.

    For each missing cell (t, s, f), average feature f over the k stations
    nearest to s (by haversine distance) that are observed at time t, weighted
    by 1/distance**power. Vectorised across time per station.
    """
    T, S, F = X.shape
    out = X.copy()
    order = np.argsort(distance, axis=1)  # nearest-first (self at column 0)

    for f in range(F):
        V = X[:, :, f]                     # [T, S]
        observed = ~np.isnan(V)            # [T, S]
        missing_stations = np.where(np.isnan(V).any(axis=0))[0]
        for s in missing_stations:
            miss_t = np.where(np.isnan(V[:, s]))[0]   # only fill missing timesteps
            nbr = order[s][order[s] != s]             # neighbours, nearest first [S-1]
            d = np.maximum(distance[s, nbr], 1e-6)
            w = 1.0 / d ** power                       # [S-1]
            obs_n = observed[miss_t][:, nbr]           # [M, S-1]
            val_n = np.where(obs_n, V[miss_t][:, nbr], 0.0)
            # Keep only the first k observed neighbours per timestamp.
            keep = obs_n & (np.cumsum(obs_n, axis=1) <= k)
            wk = keep * w[None, :]
            num = (wk * val_n).sum(axis=1)
            den = wk.sum(axis=1)
            out[miss_t, s, f] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    out[np.isnan(out)] = 0.0  # cells with no observed neighbour ever
    return out


def spatial_temporal(X: np.ndarray, distance: np.ndarray, k: int = 8) -> np.ndarray:
    """Simple hybrid: mean of linear interpolation and spatial IDW."""
    return 0.5 * (linear_interp(X) + spatial_idw(X, distance, k))


# ---------------------------------------------------------------------------
# evaluation runner
# ---------------------------------------------------------------------------
def _config_key(mask) -> str:
    if mask.mechanism == "station_outage":
        dur = mask.params.get("duration_hours")
        return f"station_outage_{'full' if dur is None else int(dur)}"
    if mask.mechanism in ("mcar", "block"):
        return f"{mask.mechanism}_{int(mask.params['ratio'] * 100)}"
    return mask.mechanism


def run_evaluation(k: int | None = None) -> pd.DataFrame:
    """Evaluate every baseline against the saved mask suite; write + return results."""
    meta = load_meta()
    truth = load_truth()
    distance = load_distance()
    masks = load_masks()
    train_end = int(meta["split"]["train"]["end"])
    if k is None:
        k = int(load_config()["graph"]["k_neighbors"])

    records = []
    for i, mask in enumerate(masks, start=1):
        logger.info("[%d/%d] %s", i, len(masks), mask.id)
        X = make_model_input(truth, mask)

        lin = linear_interp(X)
        idw = spatial_idw(X, distance, k)
        imputes = {
            "mean_fill": mean_fill(X, train_end),
            "forward_fill": forward_fill(X),
            "linear_interp": lin,
            "spatial_idw": idw,
            "spatial_temporal": 0.5 * (lin + idw),
        }
        for name, imp in imputes.items():
            res = score(imp, mask, meta, truth)
            for tgt in TARGET_FEATURES:
                if tgt not in res["overall"]:
                    continue
                for metric, value in res["overall"][tgt].items():
                    if metric == "n":
                        continue
                    records.append({
                        "baseline": name, "config": _config_key(mask),
                        "mechanism": mask.mechanism, "seed": mask.seed,
                        "target": tgt, "metric": metric, "value": value,
                    })
        del X, lin, idw, imputes

    long = pd.DataFrame(records)
    agg = (long.groupby(["baseline", "config", "mechanism", "target", "metric"])["value"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"count": "n_seeds"}))
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(RESULTS_CSV, index=False)
    logger.info("Wrote %s (%d rows).", RESULTS_CSV, len(agg))
    return agg


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
_BASELINE_ORDER = ["mean_fill", "forward_fill", "linear_interp", "spatial_idw", "spatial_temporal"]


def _config_table(agg: pd.DataFrame, config: str) -> pd.DataFrame:
    """Baseline x (target, metric) table of 'mean±std' for one config."""
    sub = agg[agg["config"] == config]
    cells = {}
    for tgt in TARGET_FEATURES:
        for metric in ("MAE", "RMSE"):
            col = f"{tgt}_{metric}"
            cells[col] = {}
            for _, r in sub[(sub["target"] == tgt) & (sub["metric"] == metric)].iterrows():
                std = 0.0 if pd.isna(r["std"]) else r["std"]
                cells[col][r["baseline"]] = f"{r['mean']:.3f}±{std:.3f}"
    table = pd.DataFrame(cells)
    return table.reindex([b for b in _BASELINE_ORDER if b in table.index])


def _mean_for(agg, config, baseline, target, metric="MAE"):
    row = agg[(agg.config == config) & (agg.baseline == baseline)
              & (agg.target == target) & (agg.metric == metric)]
    return float(row["mean"].iloc[0]) if len(row) else float("nan")


def print_report(agg: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n" + "=" * 74)
    print("IMPUTATION BASELINES — scored via hide-and-recover harness")
    print("targets: WVHT (m), APD (s); values are mean±std over 3 seeds")
    print("=" * 74)

    print("\n### HEADLINE — station_outage (full test period) ###")
    print(_config_table(agg, "station_outage_full").to_string())

    for cfg in ("station_outage_720", "mcar_50", "block_50"):
        if (agg["config"] == cfg).any():
            print(f"\n### {cfg} ###")
            print(_config_table(agg, cfg).to_string())

    # ---- the spatial-vs-temporal moment of truth ----
    print("\n" + "=" * 74)
    print("SPATIAL (IDW) vs TEMPORAL (forward_fill / linear_interp)")
    print("on blacked-out stations  [station_outage_full]")
    print("=" * 74)
    for tgt in TARGET_FEATURES:
        idw = _mean_for(agg, "station_outage_full", "spatial_idw", tgt)
        ff = _mean_for(agg, "station_outage_full", "forward_fill", tgt)
        li = _mean_for(agg, "station_outage_full", "linear_interp", tgt)
        best_temporal = min(ff, li)
        winner = "IDW (spatial)" if idw < best_temporal else "TEMPORAL"
        delta = 100.0 * (best_temporal - idw) / best_temporal
        print(f"  {tgt} MAE (m/s): IDW={idw:.3f}  ffill={ff:.3f}  linear={li:.3f}  "
              f"-> {winner} wins by {abs(delta):.1f}%")
    print("\n(If IDW clearly beats the temporal methods here, that is the first"
          "\n evidence spatial/graph structure helps recover blacked-out stations.)")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    agg = run_evaluation()
    print_report(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
