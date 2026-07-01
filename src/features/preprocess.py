"""Preprocess the raw tensor into a model-ready, leakage-safe tensor.

Pipeline (in order):
  1. Load the raw tensor + axes from ``data/processed/``.
  2. Transform: ``log1p`` on the WVHT channel only (DPD, APD left raw). The
     EDA transform diagnostic showed WVHT is strongly right-skewed (skew +1.71
     -> +0.46 under log1p) while DPD/APD are near-symmetric and would be
     over-corrected.
  3. Temporal split (EVALUATION.md): train = 2021-2023, val = 2024, test = 2025,
     split along the time axis only.
  4. Normalize: per-feature z-score using TRAIN observed cells only. The same
     mean/std are applied to val and test — they never inform the statistics.
     This is the leakage boundary: validation/test data must not influence the
     normalization.

Outputs (to gitignored ``data/processed/`` — reproducible build artifacts):
  wave_tensor_model.npz   normalized values (NaN at missing) + boolean mask
  preprocess_meta.json    per-channel transform, per-feature mean/std, split
                          boundaries, feature order

A companion :func:`inverse_transform` undoes normalization and (for WVHT) the
log1p so model predictions can be scored in physical units (m, s), as required
by EVALUATION.md.

Run:  ``python -m src.features.preprocess``
Importable; nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TENSOR_NPZ = PROCESSED_DIR / "wave_tensor.npz"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
MODEL_NPZ = PROCESSED_DIR / "wave_tensor_model.npz"
META_JSON = PROCESSED_DIR / "preprocess_meta.json"

# Per-channel transform decision (from the EDA transform diagnostic).
TRANSFORMS = {"WVHT": "log1p", "DPD": "none", "APD": "none"}

# Temporal split by calendar year (EVALUATION.md).
TRAIN_YEARS = (2021, 2022, 2023)
VAL_YEAR = 2024
TEST_YEAR = 2025


def _apply_transforms(tensor: np.ndarray, features: list[str]) -> np.ndarray:
    """Apply the per-channel transform. WVHT -> log1p; others unchanged.

    Works on a float64 copy; NaNs (missing cells) propagate untouched.
    """
    out = tensor.astype(np.float64).copy()
    for f_idx, name in enumerate(features):
        if TRANSFORMS.get(name, "none") == "log1p":
            # WVHT is non-negative, so log1p is always defined.
            out[:, :, f_idx] = np.log1p(out[:, :, f_idx])
    return out


def _split_boundaries(times: pd.DatetimeIndex) -> dict:
    """Index boundaries for the train/val/test temporal split.

    The grid is contiguous hourly, so years are non-decreasing and the split is
    two cut points: first hour of VAL_YEAR and first hour of TEST_YEAR.
    """
    years = times.year.to_numpy()
    train_end = int(np.searchsorted(years, VAL_YEAR))   # first index in 2024
    val_end = int(np.searchsorted(years, TEST_YEAR))    # first index in 2025
    total = len(times)

    def span(start, end, label_years):
        return {
            "start": start,
            "end": end,
            "hours": end - start,
            "years": label_years,
            "date_start": times[start].isoformat(),
            "date_end": times[end - 1].isoformat(),
        }

    return {
        "train": span(0, train_end, list(TRAIN_YEARS)),
        "val": span(train_end, val_end, [VAL_YEAR]),
        "test": span(val_end, total, [TEST_YEAR]),
    }


def _train_stats(transformed: np.ndarray, train_end: int, features: list[str]):
    """Per-feature mean/std over TRAIN observed cells only (post-transform)."""
    train = transformed[:train_end]  # [Ttr, S, F]
    means, stds = [], []
    for f_idx in range(len(features)):
        # nan-aware: NaN == missing cells are ignored, so stats use observed only.
        mean = float(np.nanmean(train[:, :, f_idx]))
        std = float(np.nanstd(train[:, :, f_idx]))  # ddof=0 (population)
        if std == 0.0:
            std = 1.0  # guard; does not occur for real wave data
        means.append(mean)
        stds.append(std)
    return means, stds


def preprocess():
    """Run the full preprocessing pipeline and write the model tensor + meta.

    Returns (normalized_tensor, mask, meta).
    """
    z = np.load(TENSOR_NPZ)
    tensor, mask = z["tensor"], z["mask"]
    axes = json.load(open(AXES_JSON))
    times = pd.DatetimeIndex(pd.to_datetime(axes["timestamps"], utc=True))
    features = list(axes["feature_names"])

    transformed = _apply_transforms(tensor, features)
    split = _split_boundaries(times)
    means, stds = _train_stats(transformed, split["train"]["end"], features)

    # Standardize every split with TRAIN statistics. NaNs stay NaN (missing).
    mean_arr = np.array(means).reshape(1, 1, -1)
    std_arr = np.array(stds).reshape(1, 1, -1)
    normalized = ((transformed - mean_arr) / std_arr).astype(np.float32)

    meta = {
        "feature_names": features,
        "transforms": {name: TRANSFORMS.get(name, "none") for name in features},
        "normalization": {
            "mean": means,
            "std": stds,
            "ddof": 0,
            "basis": "train observed cells, post-transform",
        },
        "split": split,
        "shape": list(normalized.shape),
        "source_tensor": TENSOR_NPZ.name,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_NPZ, tensor=normalized, mask=mask)
    with open(META_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Wrote %s and %s", MODEL_NPZ, META_JSON)

    return normalized, mask, meta


def load_meta(meta_json: Path = META_JSON) -> dict:
    """Load the preprocessing metadata sidecar."""
    with open(meta_json) as fh:
        return json.load(fh)


def inverse_transform(predictions, feature_idx: int, meta: dict | None = None):
    """Map normalized model outputs for one feature back to physical units.

    Undoes the z-score (``x * std + mean``) and then, for any channel that was
    log1p-transformed (WVHT), the ``expm1`` inverse. Accepts an array of any
    shape holding values for the single feature ``feature_idx``.
    """
    meta = meta or load_meta()
    name = meta["feature_names"][feature_idx]
    mean = meta["normalization"]["mean"][feature_idx]
    std = meta["normalization"]["std"][feature_idx]

    x = np.asarray(predictions, dtype=np.float64) * std + mean
    if meta["transforms"][name] == "log1p":
        x = np.expm1(x)
    return x


def print_summary(normalized, meta) -> None:
    """Print split sizes, train stats, and verify post-norm train mean≈0/std≈1."""
    features = meta["feature_names"]
    split = meta["split"]
    means = meta["normalization"]["mean"]
    stds = meta["normalization"]["std"]
    train_end = split["train"]["end"]

    print("\n=== Preprocessing summary ===")
    print(f"feature order : {features}")
    print(f"transforms    : {meta['transforms']}")

    print("\nTemporal split (hours):")
    for name in ("train", "val", "test"):
        s = split[name]
        print(f"  {name:5s}: {s['hours']:6d} h  [{s['date_start']} .. {s['date_end']}]  "
              f"idx [{s['start']}:{s['end']}]  years={s['years']}")

    print("\nPer-feature TRAIN statistics (post-transform scale):")
    for f_idx, name in enumerate(features):
        scale = "log1p" if meta["transforms"][name] == "log1p" else "raw"
        print(f"  {name:5s} ({scale:5s}): mean={means[f_idx]:+.4f}  std={stds[f_idx]:.4f}")

    print("\nPost-normalization TRAIN check (should be ~0 mean, ~1 std):")
    train = normalized[:train_end]
    for f_idx, name in enumerate(features):
        m = float(np.nanmean(train[:, :, f_idx]))
        s = float(np.nanstd(train[:, :, f_idx]))
        print(f"  {name:5s}: mean={m:+.2e}  std={s:.4f}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    normalized, _mask, meta = preprocess()
    print_summary(normalized, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
