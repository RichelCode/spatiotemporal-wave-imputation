"""Freeze the project's station set from the national coverage audit.

This applies the agreed SCOPE DECISION to ``reports/coverage_audit.csv`` and
writes the surviving stations to ``reports/selected_stations.csv``. Those
stations are the fixed universe for all downstream imputation/forecasting work.

It also (a) runs a lightweight coordinate audit that flags stations whose
coordinate is internally inconsistent with their NDBC id (e.g. a Hawaii 51xxx
buoy placed in the eastern hemisphere), and (b) applies the documented
coordinate corrections from ``configs/station_corrections.yaml`` to the frozen
set, re-deriving basin from the corrected coordinate. The raw crawl record
(coverage_audit.csv) is left uncorrected; corrections live only downstream.

Run:  ``python -m src.data.select_stations``
Nothing runs on import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from .audit_coverage import assign_basin

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_CSV = PROJECT_ROOT / "reports" / "coverage_audit.csv"
SELECTED_CSV = PROJECT_ROOT / "reports" / "selected_stations.csv"
CORRECTIONS_YAML = PROJECT_ROOT / "configs" / "station_corrections.yaml"

# --- SCOPE DECISION --------------------------------------------------------
# The project scope: keep stations whose significant-wave-height (WVHT) record
# is at least 70% complete over the 2021-2025 window AND spans at least 3 of
# the 5 years. This is the agreed national selection bar; changing these two
# numbers re-defines the project's station universe.
MIN_WVHT_COVERAGE = 0.70
MIN_WVHT_YEARS = 3

# NDBC numeric id prefixes are region-coded. Every buoy region uses western-
# hemisphere longitudes (negative) EXCEPT the 52xxx western-Pacific series
# (Guam / Marianas), which is east of the antimeridian (positive longitude).
EAST_HEMISPHERE_PREFIXES = {"52"}


def select_stations(audit_csv: Path = AUDIT_CSV) -> pd.DataFrame:
    """Read the audit table and return the stations clearing the scope bar."""
    audit = pd.read_csv(audit_csv, dtype={"station_id": str})
    keep = (audit["WVHT_coverage"] >= MIN_WVHT_COVERAGE) & (
        audit["WVHT_years_with_data"] >= MIN_WVHT_YEARS
    )
    selected = audit[keep].sort_values(["basin", "station_id"]).reset_index(drop=True)
    return selected


def audit_coordinates(stations: pd.DataFrame) -> pd.DataFrame:
    """Flag stations with internally inconsistent coordinates.

    Checks (1) coordinates in physical range, and (2) longitude hemisphere vs
    NDBC id-prefix region. Returns a DataFrame of flagged rows (empty if clean).
    """
    flags = []
    for _, row in stations.iterrows():
        sid = str(row["station_id"])
        lat, lon = float(row["lat"]), float(row["lon"])
        issues = []
        if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
            issues.append("coordinate out of range")
        if sid[:2].isdigit():
            prefix = sid[:2]
            if prefix in EAST_HEMISPHERE_PREFIXES and lon < 0:
                issues.append(f"{prefix}xxx expects eastern-hemisphere lon > 0")
            elif prefix not in EAST_HEMISPHERE_PREFIXES and lon > 0:
                issues.append(f"{prefix}xxx expects western-hemisphere lon < 0")
        if issues:
            flags.append({
                "station_id": sid, "name": row["name"], "lat": lat, "lon": lon,
                "basin": row["basin"], "issues": "; ".join(issues),
            })
    return pd.DataFrame(flags)


def _load_corrections_file(path: Path = CORRECTIONS_YAML) -> dict:
    """Load the full corrections config (empty dict if the file is absent)."""
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def apply_corrections(stations: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply documented corrections to the frozen set.

    First, coordinate corrections (fix lat/lon, then re-derive basin from the
    corrected coordinate). Then explicit basin overrides (coordinate is correct
    but the bounding-box basin tag is not appropriate). Returns the corrected
    table and a log of what was applied.
    """
    cfg = _load_corrections_file()
    coord_corrections = cfg.get("coordinate_corrections", {})
    basin_overrides = cfg.get("basin_overrides", {})
    stations = stations.copy()
    applied = []

    for sid, corr in coord_corrections.items():
        mask = stations["station_id"] == str(sid)
        if not mask.any():
            continue  # correction targets a station outside the frozen set
        idx = stations.index[mask][0]
        old = (float(stations.at[idx, "lat"]), float(stations.at[idx, "lon"]),
               stations.at[idx, "basin"])
        if "lat" in corr:
            stations.at[idx, "lat"] = float(corr["lat"])
        if "lon" in corr:
            stations.at[idx, "lon"] = float(corr["lon"])
        new_basin = assign_basin(stations.at[idx, "lat"], stations.at[idx, "lon"])
        stations.at[idx, "basin"] = new_basin
        applied.append({
            "station_id": str(sid), "kind": "coordinate",
            "old": old,
            "new": (float(stations.at[idx, "lat"]), float(stations.at[idx, "lon"]), new_basin),
            "reason": " ".join(str(corr.get("reason", "")).split()),
        })

    for sid, override in basin_overrides.items():
        mask = stations["station_id"] == str(sid)
        if not mask.any():
            continue
        idx = stations.index[mask][0]
        old_basin = stations.at[idx, "basin"]
        stations.at[idx, "basin"] = override["basin"]
        applied.append({
            "station_id": str(sid), "kind": "basin",
            "old": old_basin, "new": override["basin"],
            "reason": " ".join(str(override.get("reason", "")).split()),
        })

    return stations, applied


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Coordinate audit over the full national audit table (all crawled stations).
    full = pd.read_csv(AUDIT_CSV, dtype={"station_id": str})
    flags = audit_coordinates(full)
    print("=== Coordinate audit (all audited stations) ===")
    if flags.empty:
        print("  No coordinate anomalies flagged.")
    else:
        print(f"  {len(flags)} station(s) flagged:")
        print(flags.to_string(index=False))

    selected = select_stations()
    selected, applied = apply_corrections(selected)
    # Basin may change under correction, so re-sort for a stable ordering.
    selected = selected.sort_values(["basin", "station_id"]).reset_index(drop=True)

    SELECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_CSV, index=False)
    logger.info("Wrote %s (%d stations).", SELECTED_CSV, len(selected))

    print("\n=== Applied corrections ===")
    if not applied:
        print("  None.")
    for a in applied:
        print(f"  [{a['kind']}] {a['station_id']}: {a['old']} -> {a['new']}")
        print(f"      reason: {a['reason']}")

    print("\n=== Selected station set ===")
    print(
        f"Scope: WVHT coverage >= {MIN_WVHT_COVERAGE:.0%} "
        f"AND >= {MIN_WVHT_YEARS} years with data"
    )
    print(f"Final count: {len(selected)} stations")

    print("\nPer-basin breakdown of selected set:")
    by_basin = selected.groupby("basin").size().sort_values(ascending=False)
    print(by_basin.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
