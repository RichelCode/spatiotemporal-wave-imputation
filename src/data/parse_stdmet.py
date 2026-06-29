"""Parse ONE NDBC standard-meteorological (stdmet) file into a clean frame.

Scope: this module turns a single raw stdmet ``.txt`` file into a tidy,
numeric, UTC-indexed DataFrame and (optionally) resamples it to an hourly grid.
It does NOT crawl, merge multiple stations, or run any coverage/audit logic.

NDBC stdmet file layout (post-2007 standard met):
    Line 1:  ``#YY  MM DD hh mm WDIR WSPD GST WVHT DPD APD ...``  (column names)
    Line 2:  ``#yr  mo dy hr mn degT m/s  m/s    m  sec sec ...``  (units)
    Line 3+: whitespace-delimited observations, ~10-minute cadence.

Two header rows both start with ``#``: we read the column names from the first
and skip the second (units). Missing values are coded with per-field sentinels
(e.g. 99.0 for wave height, 999.0 for direction, 9999.0 for pressure); we strip
each field's own sentinel to NaN, never blanket-nulling across columns.

Nothing runs on import — call :func:`parse_stdmet_file` / :func:`to_hourly`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# The five leading columns encode the observation timestamp (UTC).
_TIME_COLS = ["YY", "MM", "DD", "hh", "mm"]

# NDBC's per-field missing-value codes. Each physical field has its OWN fill
# value, so we must strip per-column — a 999.0 in PRES is a real 999 hPa
# reading, while a 999.0 in WDIR is "no data". Values here are cross-checked
# against configs/config.yaml::audit.sentinels before being applied, so the
# config stays the single source of truth for what counts as a sentinel.
NDBC_FILL_VALUES = {
    "WDIR": 999.0,   # wind direction (degT)
    "WSPD": 99.0,    # wind speed (m/s)
    "GST": 99.0,     # gust speed (m/s)
    "WVHT": 99.0,    # significant wave height (m)
    "DPD": 99.0,     # dominant wave period (s)
    "APD": 99.0,     # average wave period (s)
    "MWD": 999.0,    # mean wave direction (degT)
    "PRES": 9999.0,  # sea-level pressure (hPa)
    "ATMP": 999.0,   # air temperature (degC)
    "WTMP": 999.0,   # water temperature (degC)
    "DEWP": 999.0,   # dewpoint (degC)
    "VIS": 99.0,     # visibility (nmi)
    "TIDE": 99.0,    # tide (ft)
}


def _load_sentinels(config_path: Path = CONFIG_PATH) -> set[float]:
    """Read the allowed sentinel set from the project config."""
    with open(config_path, "r") as fh:
        config = yaml.safe_load(fh)
    return {float(s) for s in config["audit"]["sentinels"]}


def _read_column_names(path: Path) -> list[str]:
    """Pull column names from the first ``#`` header line."""
    with open(path, "r") as fh:
        header = fh.readline()
    # Drop the leading '#' and split on whitespace: '#YY MM ...' -> ['YY','MM',...]
    return header.lstrip("#").split()


def parse_stdmet_file(path) -> pd.DataFrame:
    """Parse one NDBC stdmet ``.txt`` file into a clean, UTC-indexed frame.

    Steps:
      1. Read column names from header line 1; skip the units line (line 2).
      2. Coerce every column to numeric.
      3. Build a UTC ``DatetimeIndex`` from the YY/MM/DD/hh/mm columns.
      4. Strip each field's own sentinel value to NaN (per-column, config-gated).

    Returns the native (~10-minute) cadence data; use :func:`to_hourly` to grid
    it. Index name is ``datetime``; data columns keep their NDBC names.
    """
    path = Path(path)
    col_names = _read_column_names(path)

    # Both header rows start with '#', so skip the first two physical lines and
    # supply the parsed names ourselves.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=col_names,
        skiprows=2,
    )

    # Everything in a stdmet file is numeric; coerce defensively.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build the UTC timestamp. Modern files use 4-digit years; tolerate the rare
    # 2-digit legacy form by promoting anything < 100 into the 1900s.
    year = df["YY"].astype(int)
    year = year.where(year >= 100, year + 1900)
    index = pd.to_datetime(
        {
            "year": year,
            "month": df["MM"].astype(int),
            "day": df["DD"].astype(int),
            "hour": df["hh"].astype(int),
            "minute": df["mm"].astype(int),
        },
        utc=True,
    )

    df = df.drop(columns=_TIME_COLS)
    df.index = index
    df.index.name = "datetime"

    df = _strip_sentinels(df)
    return df


def _strip_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace each column's own NDBC fill value with NaN (config-gated)."""
    allowed = _load_sentinels()
    df = df.copy()
    for col, fill in NDBC_FILL_VALUES.items():
        if col in df.columns and fill in allowed:
            # Only this column's documented fill value is treated as missing.
            df.loc[df[col] == fill, col] = np.nan
    return df


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample native (~10-min) stdmet data onto a clean hourly UTC grid.

    Resampling choice: **first valid observation within each hour, per column**
    (equivalently, the reading nearest to and at/after the top of the hour).
    The data is binned into left-closed, hour-labelled bins ``[HH:00, HH+1:00)``
    and each column independently takes its first non-NaN value in the bin.

    Why not strict on-the-hour (minute == 00):
      - NDBC does NOT sample all fields on the hour. Meteorological fields
        (wind, pressure, temperature) report every 10 min *including* :00, but
        WAVE fields (WVHT/DPD/APD) report only off the hour — typically at :10
        and :40. A strict minute==00 slice therefore drops 100% of wave data.
      - First-valid-in-hour resolves each field to its own nearest reading:
        met fields to their :00 sample, waves to their :10 sample (10 min past
        the hour, the closest available). One column-wise rule handles both.

    Why not hourly mean:
      - Averaging several sub-hourly samples would blend/fabricate values and
        distort wave statistics (DPD/APD are not linear in time). We keep a
        single genuine instantaneous measurement per hour instead.

    Missingness stays honest: an hour is NaN for a field iff that field truly
    reported nothing in the hour, which matters for the hide-and-recover
    imputation evaluation (in-hour interpolation is the model's job, not the
    loader's — see EVALUATION.md).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("to_hourly expects a DatetimeIndex (run parse_stdmet_file first).")

    df = df.sort_index()
    # Resampler.first() skips NaN by default, so each column yields the first
    # *valid* observation within the hour. Bins span the file with no gaps.
    hourly = df.resample("1h").first()
    hourly.index.name = "datetime"
    return hourly
