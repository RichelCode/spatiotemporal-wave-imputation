"""National wave-coverage audit — the table that decides project scope.

This module crawls every wave-plausible NDBC station, builds a multi-year hourly
record per station from its stdmet files, and measures how much WVHT / DPD data
each station actually has over the configured window (2021-2025). The output is
a per-station coverage table plus a threshold-distribution table that tells us
how many stations survive at various (coverage, min-years) bars — the basis for
choosing the modelling region.

It is a long crawl (hundreds of stations x 5 years, politely rate-limited), so
everything is cached on disk by :mod:`src.data.download` and progress is logged.

Outputs:
  - reports/coverage_audit.csv       one row per station with coverage stats
  - reports/coverage_thresholds.csv  #stations clearing each (coverage x years) bar

Run:  ``python -m src.data.audit_coverage [--limit N] [--force]``
Nothing runs on import.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .download import (
    _session,
    download_station_year,
    fetch_station_list,
    load_config,
    years_from_config,
)
from .parse_stdmet import parse_stdmet_file, to_hourly

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
AUDIT_CSV = REPORTS_DIR / "coverage_audit.csv"
THRESHOLDS_CSV = REPORTS_DIR / "coverage_thresholds.csv"

# Station types that plausibly carry wave measurements. Everything else
# (notably dart tsunami buoys, TAO tropical-array moorings, and USV uncrewed
# surface vehicles) is dropped — they don't provide a usable stdmet wave record.
STATION_TYPES_KEEP = {"buoy", "fixed", "other"}

# Variables we audit. WVHT (significant wave height) is the primary scope driver;
# DPD (dominant period) is reported for context (it is genuinely sparser).
AUDIT_VARS = ("WVHT", "DPD")

# Reference bar used for the per-basin summary printout.
REF_COVERAGE = 0.70
REF_MIN_YEARS = 3


# ---------------------------------------------------------------------------
# Station filtering and basin tagging
# ---------------------------------------------------------------------------
def filter_wave_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """Keep only wave-plausible stations (the crawl set)."""
    types = stations["type"].astype(str).str.lower().str.strip()
    keep = types.isin(STATION_TYPES_KEEP)
    # Belt-and-suspenders: drop anything flagged as a DART tsunami station even
    # if its type slipped through.
    keep &= ~stations["dart"].fillna(False).astype(bool)
    survivors = stations[keep].copy()
    logger.info(
        "Wave-station filter: %d of %d stations kept (types %s, dart excluded).",
        len(survivors), len(stations), sorted(STATION_TYPES_KEEP),
    )
    return survivors


def assign_basin(lat: Optional[float], lon: Optional[float]) -> str:
    """Tag a station with a coarse basin from a lat/lon bounding box.

    Boxes are deliberately approximate (NDBC coastlines are irregular) and are
    checked in order, first match wins. Longitudes are negative west of the
    prime meridian. Documented boxes:

      Great Lakes   : 40.5..49.5 N, -93..-76      (inland, distinct from coasts)
      Gulf          : 17..31 N,    -98..-80        (Gulf of Mexico)
      S. Atlantic   : 24..35 N,    -82..-73        (FL/GA/SC to Cape Hatteras)
      Mid-Atlantic  : 35..40.5 N,  -78..-69        (Hatteras to NY Bight)
      NE            : 40.5..45 N,  -72..-65        (New England / Gulf of Maine)
      W. Coast      : 30..49.5 N,  -130..-116      (CA/OR/WA shelf)
      Pacific Is.   : -15..30 N,   lon<=-150 or lon>=140  (Hawaii + tropical Pac.)
      Other/Intl    : anything else (incl. missing coords)
    """
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return "Other/Intl"

    if 40.5 <= lat <= 49.5 and -93 <= lon <= -76:
        return "Great Lakes"
    if 17 <= lat <= 31 and -98 <= lon <= -80:
        return "Gulf"
    if 24 <= lat <= 35 and -82 <= lon <= -73:
        return "S. Atlantic"
    if 35 < lat <= 40.5 and -78 <= lon <= -69:
        return "Mid-Atlantic"
    if 40.5 < lat <= 45 and -72 <= lon <= -65:
        return "NE"
    if 30 <= lat <= 49.5 and -130 <= lon <= -116:
        return "W. Coast"
    if -15 <= lat <= 30 and (lon <= -150 or lon >= 140):
        return "Pacific Is."
    return "Other/Intl"


def add_basin(stations: pd.DataFrame) -> pd.DataFrame:
    """Add a ``basin`` column from each station's lat/lon."""
    stations = stations.copy()
    stations["basin"] = [
        assign_basin(lat, lon) for lat, lon in zip(stations["lat"], stations["lon"])
    ]
    return stations


# ---------------------------------------------------------------------------
# Per-station hourly record + coverage stats
# ---------------------------------------------------------------------------
def build_grid(config: dict) -> pd.DatetimeIndex:
    """The full hourly UTC grid spanning the configured window (inclusive)."""
    start = pd.Timestamp(config["data"]["window_start"], tz="UTC")
    # window_end is a date (midnight); extend to that day's final hour (23:00).
    end = pd.Timestamp(config["data"]["window_end"], tz="UTC") + pd.Timedelta(hours=23)
    return pd.date_range(start, end, freq="1h")


def load_station_hourly(
    station: str,
    years: list[int],
    session=None,
    force: bool = False,
) -> Optional[pd.DataFrame]:
    """Download+parse+resample all available years for one station.

    Returns one concatenated hourly frame, or ``None`` if the station has no
    available files at all.
    """
    frames = []
    for year in years:
        path = download_station_year(station, year, session=session, force=force)
        if path is None:
            continue
        try:
            frames.append(to_hourly(parse_stdmet_file(path)))
        except Exception as exc:  # a single bad file shouldn't sink the station
            logger.warning("Parse failed for %s %d: %s", station, year, exc)

    if not frames:
        return None

    hourly = pd.concat(frames).sort_index()
    # Guard against any cross-file duplicate timestamps.
    hourly = hourly[~hourly.index.duplicated(keep="first")]
    return hourly


def _longest_nan_run(series: pd.Series) -> int:
    """Longest run of consecutive NaNs in a (grid-aligned) series, in steps."""
    is_nan = series.isna().to_numpy()
    longest = current = 0
    for missing in is_nan:
        if missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def audit_station(
    meta_row: pd.Series,
    grid: pd.DatetimeIndex,
    years: list[int],
    session=None,
    force: bool = False,
) -> Optional[dict]:
    """Compute coverage stats for one station over the full grid.

    Returns a record dict, or ``None`` if the station has no data to audit.
    """
    station = meta_row["station_id"]
    hourly = load_station_hourly(station, years, session=session, force=force)
    if hourly is None:
        return None

    aligned = hourly.reindex(grid)
    record = {
        "station_id": station,
        "name": meta_row.get("name"),
        "owner": meta_row.get("owner"),
        "lat": meta_row.get("lat"),
        "lon": meta_row.get("lon"),
        "basin": meta_row.get("basin"),
        "expected_hours": len(grid),
    }

    for var in AUDIT_VARS:
        series = aligned[var] if var in aligned.columns else pd.Series(index=grid, dtype=float)
        non_null = int(series.notna().sum())
        record[f"{var}_nonnull"] = non_null
        record[f"{var}_coverage"] = non_null / len(grid)
        record[f"{var}_longest_gap_h"] = _longest_nan_run(series)
        record[f"{var}_years_with_data"] = int(series.dropna().index.year.nunique())

    return record


# ---------------------------------------------------------------------------
# Aggregate tables
# ---------------------------------------------------------------------------
def build_threshold_table(
    audit_df: pd.DataFrame,
    coverage_grid: list[float],
    min_years_grid: list[int],
) -> pd.DataFrame:
    """#stations clearing BOTH a WVHT-coverage bar and a min-years bar."""
    rows = []
    for cov in coverage_grid:
        row = {"wvht_coverage_threshold": cov}
        for min_years in min_years_grid:
            if audit_df.empty:
                count = 0
            else:
                clears = (audit_df["WVHT_coverage"] >= cov) & (
                    audit_df["WVHT_years_with_data"] >= min_years
                )
                count = int(clears.sum())
            row[f"min_years>={min_years}"] = count
        rows.append(row)
    return pd.DataFrame(rows).set_index("wvht_coverage_threshold")


def basin_summary(
    audit_df: pd.DataFrame,
    coverage: float = REF_COVERAGE,
    min_years: int = REF_MIN_YEARS,
) -> pd.DataFrame:
    """Per-basin counts: total audited vs surviving the reference bar."""
    if audit_df.empty:
        return pd.DataFrame(columns=["audited", "surviving"])
    clears = (audit_df["WVHT_coverage"] >= coverage) & (
        audit_df["WVHT_years_with_data"] >= min_years
    )
    summary = pd.DataFrame(
        {
            "audited": audit_df.groupby("basin").size(),
            "surviving": audit_df[clears].groupby("basin").size(),
        }
    )
    return summary.fillna(0).astype(int).sort_values("surviving", ascending=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_audit(limit: Optional[int] = None, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full audit and write both report CSVs. Returns (audit, thresholds)."""
    config = load_config()
    years = years_from_config(config)
    grid = build_grid(config)
    coverage_grid = config["audit"]["coverage_grid"]
    min_years_grid = config["audit"]["min_years_grid"]

    logger.info(
        "Audit window: %s..%s  (%d years, %d expected hours)",
        config["data"]["window_start"], config["data"]["window_end"],
        len(years), len(grid),
    )

    stations = add_basin(filter_wave_stations(fetch_station_list()))
    if limit is not None:
        stations = stations.head(limit)
        logger.info("LIMIT active: auditing only the first %d stations.", limit)

    session = _session()
    total = len(stations)
    records, skipped = [], 0

    for i, (_, row) in enumerate(stations.iterrows(), start=1):
        logger.info("[%d/%d] %s  basin=%s", i, total, row["station_id"], row["basin"])
        try:
            record = audit_station(row, grid, years, session=session, force=force)
        except Exception as exc:  # never let one station kill the whole crawl
            logger.warning("Station %s failed: %s", row["station_id"], exc)
            record = None
        if record is None:
            skipped += 1
            continue
        records.append(record)

    logger.info("Audited %d stations; skipped %d with no data.", len(records), skipped)

    audit_df = pd.DataFrame(records)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(AUDIT_CSV, index=False)
    logger.info("Wrote %s (%d rows).", AUDIT_CSV, len(audit_df))

    thresholds = build_threshold_table(audit_df, coverage_grid, min_years_grid)
    thresholds.to_csv(THRESHOLDS_CSV)
    logger.info("Wrote %s.", THRESHOLDS_CSV)

    # ----- console report -----
    print("\n=== WVHT threshold distribution (stations clearing BOTH bars) ===")
    print(thresholds.to_string())

    print(
        f"\n=== Per-basin summary at reference bar "
        f"(WVHT coverage >= {REF_COVERAGE:.0%}, >= {REF_MIN_YEARS} years) ==="
    )
    print(basin_summary(audit_df).to_string())

    return audit_df, thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description="NDBC national wave-coverage audit.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Audit only the first N stations (for a quick test run).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files even if cached.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    run_audit(limit=args.limit, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
