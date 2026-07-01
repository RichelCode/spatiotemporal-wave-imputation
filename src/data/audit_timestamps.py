"""Audit timestamp consistency across the 140 frozen stations (diagnostic only).

The forecasting pipeline assumes a clean, gap-free hourly grid. This module
verifies that assumption before any modelling:

  1. Native cadence: for each station, the median inter-observation interval of
     the raw (pre-resample) record, classified as ~10-min, ~hourly, mixed, or
     other. Reports the spread and flags anything non-standard.
  2. Hour-boundary alignment: after ``to_hourly``, every timestamp lands exactly
     on an hour (minute=0, second=0) and is a member of the global grid.
  3. Tensor time axis: the saved axes span 2021-01-01 00:00 .. 2025-12-31 23:00
     UTC as a monotonic, duplicate-free, gap-free hourly grid (43,824 steps).

It changes nothing and writes nothing — run it and read the summary.

Run:  ``python -m src.data.audit_timestamps``
Importable; nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .download import _session, download_station_year, load_config, years_from_config
from .parse_stdmet import parse_stdmet_file, to_hourly

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"

# Cadence classification tolerances (minutes).
TEN_MIN = (8.0, 15.0)     # ~10-minute native cadence
HOURLY = (45.0, 75.0)     # ~hourly native cadence
MIXED_FRACTION = 0.20     # both cadences must exceed this share to be "mixed"


def build_grid(config: dict) -> pd.DatetimeIndex:
    """The expected hourly UTC grid spanning the configured window (inclusive)."""
    start = pd.Timestamp(config["data"]["window_start"], tz="UTC")
    end = pd.Timestamp(config["data"]["window_end"], tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="1h")


def load_station_native(station: str, years: list[int], session=None) -> pd.DataFrame | None:
    """Concatenate a station's raw (pre-resample) hourly-stdmet records."""
    frames = []
    for year in years:
        path = download_station_year(station, year, session=session)
        if path is None:
            continue
        try:
            frames.append(parse_stdmet_file(path))
        except Exception as exc:
            logger.warning("parse failed %s %d: %s", station, year, exc)
    if not frames:
        return None
    native = pd.concat(frames).sort_index()
    return native[~native.index.duplicated(keep="first")]


def native_cadence_stats(index: pd.DatetimeIndex) -> dict:
    """Median inter-observation interval (min) and cadence class for a series."""
    diffs_min = index.to_series().diff().dropna().dt.total_seconds().to_numpy() / 60.0
    if diffs_min.size == 0:
        return {"n": len(index), "median_min": np.nan, "frac_10": 0.0,
                "frac_60": 0.0, "cadence": "empty"}
    median = float(np.median(diffs_min))
    frac_10 = float(np.mean((diffs_min >= TEN_MIN[0]) & (diffs_min <= TEN_MIN[1])))
    frac_60 = float(np.mean((diffs_min >= HOURLY[0]) & (diffs_min <= HOURLY[1])))

    if frac_10 > MIXED_FRACTION and frac_60 > MIXED_FRACTION:
        cadence = "mixed"          # substantial mass at both 10-min and hourly
    elif 5.0 <= median <= 7.0:
        cadence = "6-min"          # NOS 6-minute reporting
    elif TEN_MIN[0] <= median <= TEN_MIN[1]:
        cadence = "10-min"
    elif 25.0 <= median <= 35.0:
        cadence = "30-min"         # wave-only CDIP/SCRIPPS buoys
    elif HOURLY[0] <= median <= HOURLY[1]:
        cadence = "hourly"
    else:
        cadence = "irregular"      # genuinely nonstandard
    return {"n": len(index), "median_min": median, "frac_10": frac_10,
            "frac_60": frac_60, "cadence": cadence}


def audit_station(station: str, years: list[int], grid: pd.DatetimeIndex, session=None) -> dict:
    """Per-station cadence + hour-boundary + grid-membership checks."""
    native = load_station_native(station, years, session=session)
    if native is None:
        return {"station_id": station, "cadence": "no-data", "n_native": 0,
                "median_min": np.nan, "frac_10": np.nan, "frac_60": np.nan,
                "on_hour": False, "in_grid": False, "n_hourly": 0}

    stats = native_cadence_stats(native.index)
    hourly = to_hourly(native)
    idx = hourly.index
    on_hour = bool((idx.minute == 0).all() and (idx.second == 0).all())
    # Separate benign out-of-window boundary hours (dropped when a station is
    # reindexed onto the grid in build_tensor) from genuine in-window misalign.
    n_pre = int((idx < grid[0]).sum())
    n_post = int((idx > grid[-1]).sum())
    in_window = idx[(idx >= grid[0]) & (idx <= grid[-1])]
    n_inwin_offgrid = int((~in_window.isin(grid)).sum())

    return {
        "station_id": station, "cadence": stats["cadence"], "n_native": stats["n"],
        "median_min": stats["median_min"], "frac_10": stats["frac_10"],
        "frac_60": stats["frac_60"], "on_hour": on_hour,
        "n_pre": n_pre, "n_post": n_post, "n_inwin_offgrid": n_inwin_offgrid,
        "n_hourly": len(hourly),
    }


def check_tensor_grid(grid: pd.DatetimeIndex) -> dict:
    """Verify the saved tensor time axis is a clean hourly grid."""
    axes = json.load(open(AXES_JSON))
    times = pd.DatetimeIndex(pd.to_datetime(axes["timestamps"], utc=True))
    step_ok = bool((times.to_series().diff().dropna() == pd.Timedelta("1h")).all())
    return {
        "n_steps": len(times),
        "expected_steps": len(grid),
        "monotonic": bool(times.is_monotonic_increasing),
        "no_duplicates": bool(not times.has_duplicates),
        "gap_free_hourly": step_ok,
        "start": times[0].isoformat(),
        "end": times[-1].isoformat(),
        "matches_expected_grid": bool(times.equals(grid)),
    }


def run_audit() -> pd.DataFrame:
    """Run the full timestamp audit and print the summary. Returns the per-station table."""
    config = load_config()
    years = years_from_config(config)
    grid = build_grid(config)
    station_ids = list(json.load(open(AXES_JSON))["station_ids"])

    session = _session()
    rows = []
    for i, station in enumerate(station_ids, start=1):
        logger.info("[%d/%d] %s", i, len(station_ids), station)
        rows.append(audit_station(station, years, grid, session=session))
    table = pd.DataFrame(rows)

    print("\n" + "=" * 68)
    print("TIMESTAMP AUDIT — 140 frozen stations")
    print("=" * 68)

    print("\n--- Native cadence classification ---")
    counts = table["cadence"].value_counts()
    for cadence, n in counts.items():
        print(f"  {cadence:10s}: {n}")

    print("\n--- Median native interval (minutes) across stations ---")
    med = table["median_min"].dropna()
    print(f"  min={med.min():.1f}  p25={med.quantile(.25):.1f}  median={med.median():.1f}  "
          f"p75={med.quantile(.75):.1f}  max={med.max():.1f}")

    irregular = table[table["cadence"] == "irregular"]
    print("\n--- Irregular / nonstandard cadence ---")
    if irregular.empty:
        print("  None — every station has a regular cadence "
              "(6-min, 10-min, 30-min, hourly, or mixed).")
    else:
        print(irregular[["station_id", "cadence", "n_native", "median_min",
                         "frac_10", "frac_60"]].to_string(index=False))

    print("\n--- Hour-boundary + grid alignment (post to_hourly) ---")
    on_hour_ok = int(table["on_hour"].sum())
    inwin_offgrid = int(table["n_inwin_offgrid"].sum())
    with_pre = int((table["n_pre"] > 0).sum())
    with_post = int((table["n_post"] > 0).sum())
    print(f"  on exact hour (min=0,sec=0)        : {on_hour_ok}/{len(table)}")
    print(f"  in-window off-grid timestamps      : {inwin_offgrid}  (must be 0)")
    print(f"  stations w/ pre-window boundary hr : {with_pre}  "
          f"(2020-12-31 23:00; benign, trimmed by build_tensor's grid reindex)")
    print(f"  stations w/ post-window boundary hr: {with_post}")

    print("\n--- Tensor time axis ---")
    g = check_tensor_grid(grid)
    for key, value in g.items():
        print(f"  {key:22s}: {value}")

    grid_clean = (g["n_steps"] == g["expected_steps"] and g["monotonic"]
                  and g["no_duplicates"] and g["gap_free_hourly"]
                  and g["matches_expected_grid"])
    # In-window alignment is what the model relies on; out-of-window boundary
    # hours never reach the tensor.
    all_aligned = on_hour_ok == len(table) and inwin_offgrid == 0
    print("\n--- Verdict ---")
    print(f"  tensor grid clean & gap-free       : {grid_clean}")
    print(f"  all 140 on-hour, 0 in-window offgrid: {all_aligned}")
    print(f"  hourly assumption SAFE             : {grid_clean and all_aligned}")

    return table


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    run_audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
