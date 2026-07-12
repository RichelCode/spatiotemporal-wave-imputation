"""Leakage-safe 1-hour-ahead forecasting evaluation (EVALUATION.md, section 1b).

Forecasting is scored under a strict leakage boundary so that no prediction is
informed by future data:

  * TRAINING may use the GRIN-completed tensor (2021-2023). Bidirectional
    imputation in the training data is acceptable.
  * At TEST time (2025) we score an origin t (predicting t+1) ONLY if BOTH
    (a) the target y_{t+1} is GENUINELY OBSERVED (never score against an imputed
        "truth"), and
    (b) the entire input window [t-L+1, t] (L = INPUT_WINDOW hours) is genuinely
        observed for that target feature -- so the forecaster consumes no
        imputed, future-informed value. Origins with any imputed/missing cell in
        the window are excluded.
  This window rule is the strictest reading of the leakage constraint and is the
  SAME set every baseline (and, later, the deep forecaster) is scored on. L=24
  covers all baseline inputs (persistence uses t; seasonal-naive uses t-23;
  AR(p<=24) uses the window), so the comparison is apples-to-apples.

The completed tensor is produced by running the saved GRIN checkpoint on the
REAL naturally-missing data (no artificial masks) -- the actual gap-fill for the
pipeline -- and inverse-transformed to physical units.

Metrics: MAE and RMSE per target in physical units (m for WVHT, s for APD), plus
a persistence-normalised skill score 1 - MAE_model / MAE_persistence.

Importable; nothing runs on import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..features.preprocess import inverse_transform, load_meta

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_TENSOR_NPZ = PROCESSED_DIR / "wave_tensor.npz"          # physical, NaN at missing
MODEL_NPZ = PROCESSED_DIR / "wave_tensor_model.npz"         # normalised, NaN at missing
COMPLETED_NPZ = PROCESSED_DIR / "grin_completed_physical.npz"  # gitignored artifact

FORECAST_TARGETS = ["WVHT", "APD"]
HORIZON = 1          # hours ahead
INPUT_WINDOW = 24    # L: input history window that must be fully observed


def complete_tensor_with_grin(device=None, force: bool = False) -> np.ndarray:
    """Run the saved GRIN checkpoint on the real data; return the physical completed tensor.

    Cached to ``grin_completed_physical.npz``. This is the actual gap-fill: GRIN
    imputes the naturally-missing cells (no artificial masks), and the result is
    inverse-transformed to physical units (m, s).
    """
    if COMPLETED_NPZ.exists() and not force:
        return np.load(COMPLETED_NPZ)["completed"]

    # Heavy imports (torch/tsl) are deferred to here.
    from ..data.download import load_config
    from ..models.deep_common import select_device
    from ..models.grin_imputer import GRINImputer, GRIN_CHECKPOINT

    config = load_config()
    meta = load_meta()
    dev = device if device is not None else select_device(config["deep"])
    norm = np.load(MODEL_NPZ)["tensor"]  # normalised, NaN at missing

    logger.info("completing real tensor with GRIN checkpoint on %s ...", dev)
    model = GRINImputer(config, device=dev, adjacency="adjacency_knn_basin").load_checkpoint(GRIN_CHECKPOINT)
    imputed = model.impute(norm)  # normalised, gaps filled

    completed = np.empty_like(imputed, dtype=np.float64)
    for f in range(imputed.shape[2]):
        completed[:, :, f] = inverse_transform(imputed[:, :, f], f, meta)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COMPLETED_NPZ, completed=completed.astype(np.float32))
    logger.info("wrote %s", COMPLETED_NPZ)
    return completed.astype(np.float32)


def valid_origin_mask(observed_feature: np.ndarray, test_start: int, test_end: int,
                      L: int = INPUT_WINDOW) -> np.ndarray:
    """Boolean [T, S] of leakage-safe test origins for one target feature.

    Origin t is valid iff t is in the test period, the window [t-L+1, t] is fully
    observed, and the target y_{t+1} is observed. (b) + (a) of the leakage rule.
    """
    T, S = observed_feature.shape
    obs_int = observed_feature.astype(np.int64)
    prefix = np.zeros((T + 1, S), dtype=np.int64)
    prefix[1:] = np.cumsum(obs_int, axis=0)  # prefix[k] = sum obs[0..k-1]

    t = np.arange(T)
    lower = prefix[np.maximum(t - L + 1, 0)]          # sum below window start
    window_count = prefix[t + 1] - lower              # observed count in [t-L+1, t]
    window_full = (window_count == L) & (t >= L - 1)[:, None]

    target_ok = np.zeros((T, S), dtype=bool)
    target_ok[:-1] = observed_feature[1:]             # y_{t+1} observed

    origin_range = np.zeros(T, dtype=bool)
    origin_range[test_start:test_end - 1] = True       # t in [test_start, test_end-2]

    return window_full & target_ok & origin_range[:, None]


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def skill_vs_persistence(mae_model: float, mae_persistence: float) -> float:
    """1 - MAE_model / MAE_persistence (>0 means it beats persistence)."""
    return float(1.0 - mae_model / mae_persistence) if mae_persistence > 0 else float("nan")
