"""Assemble the frozen 140-station set into one spatiotemporal tensor.

This is the Phase-2 feature-build step. It takes the frozen station universe
(``reports/selected_stations.csv``), loads each station's 2021-2025 hourly
record via the existing download->parse->resample pipeline, aligns everything
onto one shared hourly grid, and stacks it into a dense 3-D array:

    tensor  shape [time=43824, station=140, feature=3]   float32, NaN = missing
    mask    shape [time=43824, station=140, feature=3]   bool, True = observed

Features are the three wave TARGETS only: WVHT, DPD, APD. Wind fields are
deliberately excluded: 65 of the 140 selected stations are wave-only buoys with
no anemometer, so wind would be structurally missing-not-at-random at the
station level. Keeping the tensor targets-only ensures every missing cell is a
genuine, potentially-recoverable sensor gap, which keeps the imputation problem
and the spatial contribution clean. Wind is reserved for a possible later
ablation on the 75 wind-equipped stations.

Station order matches the (basin, station_id) ordering of selected_stations.csv.

Outputs (to ``data/processed/``, which is gitignored — this is a reproducible
build artifact, not source):
    wave_tensor.npz        arrays ``tensor`` and ``mask``
    wave_tensor_axes.json  self-describing axis labels (timestamps, station_ids,
                           feature_names) so the .npz can be interpreted alone

Run:  ``python -m src.features.build_tensor``
Nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.audit_coverage import load_station_hourly
from ..data.download import _session, load_config, years_from_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SELECTED_CSV = PROJECT_ROOT / "reports" / "selected_stations.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TENSOR_NPZ = PROCESSED_DIR / "wave_tensor.npz"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"

# Modeled features, in fixed order: the three wave TARGETS only. Wind fields
# (WDIR/WSPD/GST) are excluded because they are structurally absent on the 65
# wave-only stations; everything else from the stdmet record is dropped too.
FEATURES = ["WVHT", "DPD", "APD"]
# All three features are response variables in the targets-only tensor.
TARGET_FEATURES = ["WVHT", "DPD", "APD"]


def build_grid(config: dict) -> pd.DatetimeIndex:
    """The shared hourly UTC grid spanning the configured window (inclusive)."""
    start = pd.Timestamp(config["data"]["window_start"], tz="UTC")
    end = pd.Timestamp(config["data"]["window_end"], tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="1h")


def load_selected_station_ids(selected_csv: Path = SELECTED_CSV) -> list[str]:
    """Read the frozen station universe, preserving the file's row order."""
    selected = pd.read_csv(selected_csv, dtype={"station_id": str})
    return selected["station_id"].tolist()


def build_tensor(
    station_ids: list[str],
    grid: pd.DatetimeIndex,
    years: list[int],
    session=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the [time, station, feature] tensor and its observed-mask.

    Each station is loaded with the existing hourly pipeline, reindexed onto the
    shared grid and onto the fixed feature order; missing cells stay NaN.
    """
    n_t, n_s, n_f = len(grid), len(station_ids), len(FEATURES)
    tensor = np.full((n_t, n_s, n_f), np.nan, dtype=np.float32)

    session = session or _session()
    for s_idx, station in enumerate(station_ids, start=0):
        logger.info("[%d/%d] loading %s", s_idx + 1, n_s, station)
        hourly = load_station_hourly(station, years, session=session)
        if hourly is None:
            # Should not happen for the frozen set (all had data in the audit),
            # but if a station yields nothing we leave its slab as all-NaN.
            logger.warning("No data for frozen station %s — leaving all-NaN.", station)
            continue
        # Align onto the shared time grid and the fixed feature order in one step.
        aligned = hourly.reindex(index=grid, columns=FEATURES)
        tensor[:, s_idx, :] = aligned.to_numpy(dtype=np.float32)

    mask = ~np.isnan(tensor)
    return tensor, mask


def save_tensor(
    tensor: np.ndarray,
    mask: np.ndarray,
    grid: pd.DatetimeIndex,
    station_ids: list[str],
) -> None:
    """Persist the tensor/mask (.npz) plus a self-describing axes sidecar (.json)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TENSOR_NPZ, tensor=tensor, mask=mask)
    axes = {
        "timestamps": [t.isoformat() for t in grid],
        "station_ids": list(station_ids),
        "feature_names": list(FEATURES),
        "shape": list(tensor.shape),
        "dims": ["time", "station", "feature"],
    }
    with open(AXES_JSON, "w") as fh:
        json.dump(axes, fh)
    logger.info("Wrote %s and %s", TENSOR_NPZ, AXES_JSON)


def print_summary(tensor: np.ndarray, mask: np.ndarray) -> None:
    """Print shape and observed-fraction breakdowns."""
    overall = float(mask.mean())
    per_feature = mask.mean(axis=(0, 1))  # fraction observed per feature

    print("\n=== Spatiotemporal tensor summary ===")
    print(f"shape [time, station, feature] : {tensor.shape}")
    print(f"overall observed fraction      : {overall:.4f}  ({overall:.1%})")

    print("\nper-feature observed fraction:")
    for f_idx, name in enumerate(FEATURES):
        tag = "  <- target" if name in TARGET_FEATURES else ""
        print(f"  {name:5s}: {per_feature[f_idx]:.4f}  ({per_feature[f_idx]:.1%}){tag}")

    print("\ntargets specifically:")
    for name in TARGET_FEATURES:
        frac = float(per_feature[FEATURES.index(name)])
        print(f"  {name:5s}: {frac:.4f}  ({frac:.1%})")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = load_config()
    years = years_from_config(config)
    grid = build_grid(config)
    station_ids = load_selected_station_ids()
    logger.info(
        "Building tensor for %d stations x %d hours x %d features.",
        len(station_ids), len(grid), len(FEATURES),
    )

    tensor, mask = build_tensor(station_ids, grid, years)
    save_tensor(tensor, mask, grid, station_ids)
    print_summary(tensor, mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
