"""Freeze the project's station set from the national coverage audit.

This applies the agreed SCOPE DECISION to ``reports/coverage_audit.csv`` and
writes the surviving stations to ``reports/selected_stations.csv``. Those
stations are the fixed universe for all downstream imputation/forecasting work.

Run:  ``python -m src.data.select_stations``
Nothing runs on import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_CSV = PROJECT_ROOT / "reports" / "coverage_audit.csv"
SELECTED_CSV = PROJECT_ROOT / "reports" / "selected_stations.csv"

# --- SCOPE DECISION --------------------------------------------------------
# The project scope: keep stations whose significant-wave-height (WVHT) record
# is at least 70% complete over the 2021-2025 window AND spans at least 3 of
# the 5 years. This is the agreed national selection bar; changing these two
# numbers re-defines the project's station universe.
MIN_WVHT_COVERAGE = 0.70
MIN_WVHT_YEARS = 3


def select_stations(audit_csv: Path = AUDIT_CSV) -> pd.DataFrame:
    """Read the audit table and return the stations clearing the scope bar."""
    audit = pd.read_csv(audit_csv)
    keep = (audit["WVHT_coverage"] >= MIN_WVHT_COVERAGE) & (
        audit["WVHT_years_with_data"] >= MIN_WVHT_YEARS
    )
    selected = audit[keep].sort_values(["basin", "station_id"]).reset_index(drop=True)
    return selected


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    selected = select_stations()
    SELECTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(SELECTED_CSV, index=False)
    logger.info("Wrote %s (%d stations).", SELECTED_CSV, len(selected))

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
