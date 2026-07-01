"""Shared infrastructure for deep imputation models (Stage-C scaffold).

Every deep model (SAITS/BRITS/GRIN/CSDI, later) trains through this plumbing and
is scored through the same Stage-A hide-and-recover harness, so deep and
classical results are directly comparable. No real model lives here yet — just
the machinery plus a trivial MeanImputer to smoke-test the pipeline end to end.

Windowing
---------
The time axis is cut into NON-OVERLAPPING windows of length W (config
``deep.window_hours``, default 48). We chose non-overlapping windows because
(a) the full 43,824-hour axis tiles exactly into 913 windows at W=48, giving
complete, unambiguous coverage for reassembly, and (b) each observed cell is
used exactly once during training, so simple statistics (like MeanImputer's
per-station mean) are unbiased. When a split length is not divisible by W, the
final window is NaN-padded (and marked unobserved) so no data is dropped.
``reassemble`` averages any overlap, so an overlapping scheme could be swapped
in later without changing callers.

Leakage-safe splits
-------------------
Training windows come from 2021-2023, validation from 2024. The 2025 test period
is only seen at impute time, and only as the MASKED input — a model never sees
hidden-cell truth. Given a MaskSpec, :func:`make_model_input` blanks the hidden
cells; the model imputes the full tensor; scoring reads only the hidden target
cells.

Interface
---------
DeepImputer: ``.fit(train_windows, val_windows)`` and
``.impute(full_masked_tensor) -> full_imputed_tensor``. :func:`evaluate_deep_model`
runs a fitted model over the saved mask suite and aggregates results in the SAME
format as reports/baseline_imputation_results.csv.

Run:  ``python -m src.models.deep_common``  (device + window counts + smoke test)
Importable; nothing runs on import.
"""

from __future__ import annotations

import abc
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.download import load_config
from ..evaluation.masking import TARGET_FEATURES, load_masks, make_model_input, score
from ..features.preprocess import load_meta
from .baselines import _config_key, load_truth, mean_fill

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# device
# ---------------------------------------------------------------------------
def select_device(deep_cfg: dict) -> torch.device:
    """Resolve the compute device (auto: MPS -> CUDA -> CPU)."""
    pref = deep_cfg.get("device", "auto")
    if pref == "auto":
        if torch.backends.mps.is_available():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
    else:
        name = pref
    return torch.device(name)


# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------
@dataclass
class Windows:
    """A batch of temporal windows over the [time, station, feature] tensor."""
    values: np.ndarray      # [N, W, S, F] float64; NaN at missing/padding
    observed: np.ndarray    # [N, W, S, F] bool
    starts: np.ndarray      # [N] int; global time index of each window start

    def __len__(self) -> int:
        return len(self.starts)


def window_series(tensor: np.ndarray, observed: np.ndarray,
                  start: int, end: int, W: int) -> Windows:
    """Cut ``tensor[start:end]`` into non-overlapping length-W windows.

    The last window is NaN-padded (and marked unobserved) if ``end - start`` is
    not a multiple of W, so every cell in [start, end) is covered exactly once.
    """
    S, F = tensor.shape[1], tensor.shape[2]
    starts = np.arange(start, end, W)
    n = len(starts)
    # float64 so simple statistics match the classical (float64) path exactly;
    # a real model casts each batch to float32 at the torch boundary.
    values = np.full((n, W, S, F), np.nan, dtype=np.float64)
    obs = np.zeros((n, W, S, F), dtype=bool)
    for i, s in enumerate(starts):
        e = min(s + W, end)
        length = e - s
        values[i, :length] = tensor[s:e]
        obs[i, :length] = observed[s:e]
    return Windows(values=values, observed=obs, starts=starts)


def reassemble(window_values: np.ndarray, starts: np.ndarray, W: int, T: int) -> np.ndarray:
    """Stitch per-window outputs [N, W, S, F] back into a full [T, S, F] tensor.

    Overlapping windows are averaged; padding rows (start+j >= T) are ignored.
    """
    S, F = window_values.shape[2], window_values.shape[3]
    acc = np.zeros((T, S, F), dtype=np.float64)
    cnt = np.zeros((T, S, F), dtype=np.float64)
    for i, s in enumerate(starts):
        length = min(W, T - s)
        acc[s:s + length] += window_values[i, :length]
        cnt[s:s + length] += 1.0
    return acc / np.maximum(cnt, 1.0)


def build_training_windows(truth: np.ndarray, observed: np.ndarray, meta: dict, W: int):
    """Leakage-safe train (2021-2023) and val (2024) windows from the clean tensor."""
    tr, va = meta["split"]["train"], meta["split"]["val"]
    train = window_series(truth, observed, int(tr["start"]), int(tr["end"]), W)
    val = window_series(truth, observed, int(va["start"]), int(va["end"]), W)
    return train, val


# ---------------------------------------------------------------------------
# model interface
# ---------------------------------------------------------------------------
class DeepImputer(abc.ABC):
    """Common interface every deep imputation model conforms to."""
    name = "deep"

    @abc.abstractmethod
    def fit(self, train_windows: Windows, val_windows: Windows) -> "DeepImputer":
        """Train on windowed 2021-2023 data, validating on 2024."""

    @abc.abstractmethod
    def impute(self, full_masked_tensor: np.ndarray) -> np.ndarray:
        """Impute a full [T, S, F] masked tensor (NaN at hidden+missing cells)."""


class MeanImputer(DeepImputer):
    """Trivial reference model: fill every gap with the per-station training mean.

    Exists only to smoke-test the window -> train -> impute -> score pipeline; by
    construction its output equals the classical ``mean_fill`` baseline.
    """
    name = "mean_imputer"

    def __init__(self, window_hours: int):
        self.W = int(window_hours)
        self.means = None  # [S, F]

    def fit(self, train_windows: Windows, val_windows: Windows) -> "MeanImputer":
        S, F = train_windows.values.shape[2], train_windows.values.shape[3]
        flat = train_windows.values.reshape(-1, S, F)  # NaN at missing+padding
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            per_sf = np.nanmean(flat, axis=0)                    # [S, F]
            global_f = np.nanmean(flat.reshape(-1, F), axis=0)   # [F]
        self.means = np.where(np.isnan(per_sf), global_f[None, :], per_sf)
        return self

    def impute(self, full_masked_tensor: np.ndarray) -> np.ndarray:
        if self.means is None:
            raise RuntimeError("MeanImputer.impute called before fit")
        T = full_masked_tensor.shape[0]
        observed = ~np.isnan(full_masked_tensor)
        win = window_series(full_masked_tensor, observed, 0, T, self.W)
        filled = win.values.copy()
        nan = np.isnan(filled)
        broadcast = np.broadcast_to(self.means[None, None, :, :], filled.shape)
        filled[nan] = broadcast[nan]
        return reassemble(filled, win.starts, self.W, T)


# ---------------------------------------------------------------------------
# evaluation glue (same output format as the baseline runner)
# ---------------------------------------------------------------------------
def evaluate_deep_model(model: DeepImputer, masks=None, save_path: Path | None = None) -> pd.DataFrame:
    """Score a fitted model over the saved mask suite; aggregate mean+/-std over seeds.

    Output columns match reports/baseline_imputation_results.csv exactly
    (baseline, config, mechanism, target, metric, mean, std, n_seeds) so deep and
    classical results concatenate directly. The ``baseline`` column holds the
    model name.
    """
    meta = load_meta()
    truth = load_truth()
    masks = masks or load_masks()

    records = []
    for i, mask in enumerate(masks, start=1):
        logger.info("[%d/%d] %s on %s", i, len(masks), model.name, mask.id)
        imputed = model.impute(make_model_input(truth, mask))
        res = score(imputed, mask, meta, truth)
        for tgt in TARGET_FEATURES:
            if tgt not in res["overall"]:
                continue
            for metric, value in res["overall"][tgt].items():
                if metric == "n":
                    continue
                records.append({
                    "baseline": model.name, "config": _config_key(mask),
                    "mechanism": mask.mechanism, "seed": mask.seed,
                    "target": tgt, "metric": metric, "value": value,
                })

    agg = (pd.DataFrame(records)
           .groupby(["baseline", "config", "mechanism", "target", "metric"])["value"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"count": "n_seeds"}))
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(save_path, index=False)
        logger.info("Wrote %s (%d rows).", save_path, len(agg))
    return agg


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    config = load_config()
    W = int(config["deep"]["window_hours"])

    device = select_device(config["deep"])
    print(f"Selected device: {device}")

    truth = load_truth()
    observed = ~np.isnan(truth)
    meta = load_meta()
    T = truth.shape[0]

    train, val = build_training_windows(truth, observed, meta, W)
    ts, te = int(meta["split"]["test"]["start"]), int(meta["split"]["test"]["end"])
    test = window_series(truth, observed, ts, te, W)
    print(f"Window length W = {W} h")
    print(f"Window counts -> train: {len(train)}  val: {len(val)}  test: {len(test)}  "
          f"(full-tensor impute windows: {len(np.arange(0, T, W))})")

    # Exact plumbing check: window -> reassemble must reconstruct any tensor.
    finite = np.nan_to_num(truth, nan=0.0).astype(np.float32)
    win = window_series(finite, observed, 0, T, W)
    rebuilt = reassemble(win.values, win.starts, W, T)
    assert np.allclose(rebuilt, finite, atol=1e-5), "reassemble round-trip failed"
    print("window -> reassemble round-trip: exact  [ok]")

    # Fit trivial model and compare to the classical mean_fill baseline.
    model = MeanImputer(W).fit(train, val)
    masks = {m.id: m for m in load_masks()}
    mask = masks["station_outage_v14_durfull_seed0"]
    X = make_model_input(truth, mask)

    imp_deep = model.impute(X)
    imp_base = mean_fill(X, int(meta["split"]["train"]["end"]))
    max_abs_diff = float(np.nanmax(np.abs(imp_deep - imp_base)))

    res_deep = score(imp_deep, mask, meta, truth)
    res_base = score(imp_base, mask, meta, truth)

    print(f"\nSmoke test on {mask.id}")
    print(f"  imputed-tensor max|deep - mean_fill| = {max_abs_diff:.2e}")
    print(f"  {'target':6s} {'MeanImputer':>14s} {'mean_fill':>12s} {'Δ':>10s}")
    for tgt in TARGET_FEATURES:
        d = res_deep["overall"][tgt]["MAE"]
        b = res_base["overall"][tgt]["MAE"]
        print(f"  {tgt:6s} {d:14.6f} {b:12.6f} {abs(d - b):10.2e}   (MAE, physical units)")

    matches = all(abs(res_deep["overall"][t]["MAE"] - res_base["overall"][t]["MAE"]) < 1e-9
                  for t in TARGET_FEATURES)
    print(f"\n  deep pipeline reproduces mean_fill exactly: {matches}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
