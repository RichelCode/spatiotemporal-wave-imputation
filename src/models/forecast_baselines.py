"""Classical 1-hour-ahead forecasting baselines (the floor before any deep forecaster).

All baselines predict y_{t+1} for WVHT and APD from history up to t, and are
scored through the leakage-safe harness in src/evaluation/forecast_eval.py
(target observed, input window fully observed, physical units).

  persistence     y_hat_{t+1} = y_t                       (the critical baseline)
  seasonal_naive  y_hat_{t+1} = y_{t-23} (same hour, prev day)
  ar24            AR(24) per station, fit on the GRIN-completed train series
                  (2021-2023), 1-step forecast from the observed test window.
                  AR(24) is exactly ARIMA(24,0,0); a full ARIMA(p,d,q) with MA
                  terms would require statsmodels (not installed) -- flagged, not
                  added, so the environment stays pinned.

Training on the completed (bidirectionally-imputed) train series is permitted;
at test the AR model consumes only genuinely-observed lags (the window is
required observed), so there is no future-informed leakage.

Run:  ``python -m src.models.forecast_baselines``
Importable; nothing runs on import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation.forecast_eval import (
    FORECAST_TARGETS, INPUT_WINDOW, RAW_TENSOR_NPZ, complete_tensor_with_grin,
    mae, rmse, skill_vs_persistence, valid_origin_mask,
)
from ..features.preprocess import load_meta

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "reports" / "forecast_baseline_results.csv"
AR_ORDER = 24  # AR(p); equals ARIMA(p,0,0). Requires p <= INPUT_WINDOW.


def fit_ar(series: np.ndarray, p: int) -> np.ndarray | None:
    """Least-squares AR(p) with intercept on a continuous series. Returns [c, phi_1..phi_p]."""
    n = len(series)
    if n <= p + 2 or not np.all(np.isfinite(series)):
        return None
    y = series[p:]
    cols = [np.ones(n - p)] + [series[p - k - 1: n - k - 1] for k in range(p)]  # intercept + lag1..lagp
    x = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coef


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = load_meta()
    feats = meta["feature_names"]
    test_start = int(meta["split"]["test"]["start"])
    test_end = int(meta["split"]["test"]["end"])
    train_end = int(meta["split"]["train"]["end"])

    raw = np.load(RAW_TENSOR_NPZ)["tensor"].astype(np.float64)   # physical, NaN at missing
    observed = ~np.isnan(raw)
    completed = complete_tensor_with_grin().astype(np.float64)   # physical, gap-filled

    records, origin_counts = [], {}
    for tgt in FORECAST_TARGETS:
        f = feats.index(tgt)
        valid = valid_origin_mask(observed[:, :, f], test_start, test_end, INPUT_WINDOW)
        ts, ss = np.where(valid)
        n = len(ts)
        origin_counts[tgt] = n
        logger.info("%s: %d valid test origins", tgt, n)

        y_true = raw[ts + 1, ss, f]
        pred_pers = raw[ts, ss, f]                 # y_t (observed)
        pred_seas = raw[ts - 23, ss, f]            # same hour previous day (observed)

        # AR(24): one fit per station on the completed train series.
        coefs = {int(s): fit_ar(completed[:train_end, s, f], AR_ORDER) for s in np.unique(ss)}
        lagmat = np.column_stack([raw[ts - k, ss, f] for k in range(AR_ORDER)])  # col k = y_{t-k}
        C = np.array([coefs[int(s)] if coefs[int(s)] is not None else np.zeros(AR_ORDER + 1) for s in ss])
        pred_ar = C[:, 0] + np.einsum("nk,nk->n", lagmat, C[:, 1:])
        none = np.array([coefs[int(s)] is None for s in ss])
        pred_ar[none] = pred_pers[none]            # fall back to persistence if unfittable

        methods = {"persistence": pred_pers, "seasonal_naive": pred_seas, "ar24": pred_ar}
        mae_pers = mae(pred_pers, y_true)
        for name, pred in methods.items():
            m = mae(pred, y_true)
            records.append({
                "method": name, "target": tgt, "n_origins": n,
                "MAE": m, "RMSE": rmse(pred, y_true),
                "skill_vs_persistence": skill_vs_persistence(m, mae_pers),
            })

    df = pd.DataFrame(records)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 66)
    print("1-HOUR-AHEAD FORECAST BASELINES (physical units; leakage-safe)")
    print(f"input window L={INPUT_WINDOW} h fully observed; target observed; test period 2025")
    for tgt in FORECAST_TARGETS:
        print(f"  valid test origins ({tgt}): {origin_counts[tgt]:,}")
    print("=" * 66)
    unit = {"WVHT": "m", "APD": "s"}
    print(f"\n{'target':6s}{'method':16s}{'MAE':>9s}{'RMSE':>9s}{'skill vs pers':>15s}")
    for tgt in FORECAST_TARGETS:
        for name in ("persistence", "seasonal_naive", "ar24"):
            r = df[(df.target == tgt) & (df.method == name)].iloc[0]
            print(f"{tgt:6s}{name:16s}{r['MAE']:9.4f}{r['RMSE']:9.4f}{r['skill_vs_persistence']:+15.3f}"
                  f"   ({unit[tgt]})")
    print(f"\nWrote {RESULTS_CSV}")
    print("Note: AR(24) = ARIMA(24,0,0); ARIMA with MA terms needs statsmodels (not installed).")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
