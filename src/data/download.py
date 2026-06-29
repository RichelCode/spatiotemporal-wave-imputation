"""Fetch and cache raw NDBC data (station list + historical standard met files).

This module is a *downloader only* — it retrieves and caches data, it does not
clean, merge, or analyze anything. Two things it knows how to get:

  1. The national active-station list (``activestations.xml``), parsed into a
     tidy DataFrame of station metadata (id, lat/lon, owner, sensor flags, ...).
  2. Per-station historical "standard meteorological" (stdmet) text files, one
     file per station per year.

Everything is cached on disk: a second run re-reads local files instead of
hitting NDBC again. Network access is polite — a real User-Agent, a small
inter-request delay, and 404s treated as an expected "no such file" rather than
a hard error (many station/year combinations simply don't exist).

The year range to download is derived from ``window_start`` / ``window_end`` in
``configs/config.yaml``.

Importing this module does nothing on its own; run it as a script
(``python -m src.data.download`` or ``python src/data/download.py``) to fetch the
station list and print a summary. It deliberately does NOT kick off a full
multi-station historical download on import or on a bare run.
"""

from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import yaml

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STDMET_DIR = RAW_DIR / "stdmet"
ACTIVE_STATIONS_XML = RAW_DIR / "activestations.xml"

# --- Network ---------------------------------------------------------------
ACTIVE_STATIONS_URL = "https://www.ndbc.noaa.gov/activestations.xml"
# NDBC serves the historical stdmet files (gzipped on disk) through a small PHP
# viewer that returns the *decompressed* text, so we save the result as .txt.
STDMET_URL_TEMPLATE = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/"
)

# A real, identifiable User-Agent is the polite thing to send to a public
# government server (and avoids being treated as an anonymous bot).
USER_AGENT = (
    "spatiotemporal-wave-imputation/0.1 (research; "
    "contact: https://github.com/RichelCode/spatiotemporal-wave-imputation)"
)
REQUEST_DELAY_SEC = 0.5  # brief pause between network requests
REQUEST_TIMEOUT_SEC = 60

logger = logging.getLogger(__name__)

# Station-metadata attributes we pull out of each <station .../> element.
# The first group is descriptive; the flag group marks which sensor programs a
# station participates in (met / currents / water quality / DART tsunami).
_STATION_ATTRS = ["id", "lat", "lon", "name", "owner", "pgm", "type"]
_STATION_FLAGS = ["met", "currents", "waterquality", "dart"]


def _session() -> requests.Session:
    """A requests session that always sends our User-Agent."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load the project YAML config as a plain dict."""
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


def years_from_config(config: dict) -> list[int]:
    """Inclusive list of calendar years spanned by window_start..window_end."""
    start_year = int(str(config["data"]["window_start"])[:4])
    end_year = int(str(config["data"]["window_end"])[:4])
    return list(range(start_year, end_year + 1))


def parse_station_xml(xml_path: Path) -> pd.DataFrame:
    """Parse a saved ``activestations.xml`` into a station-metadata DataFrame."""
    root = ET.parse(xml_path).getroot()
    rows = []
    for station in root.findall("station"):
        row = {attr: station.get(attr) for attr in _STATION_ATTRS}
        # Sensor-program flags are "y"/"n" in the XML; normalize to booleans.
        for flag in _STATION_FLAGS:
            row[flag] = (station.get(flag, "n") or "n").lower() == "y"
        rows.append(row)

    df = pd.DataFrame(rows, columns=_STATION_ATTRS + _STATION_FLAGS)
    df = df.rename(columns={"id": "station_id"})
    # Coordinates are numeric; everything else stays as-is.
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


def fetch_station_list(
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """Fetch (or load from cache) the national active-station list.

    If ``data/raw/activestations.xml`` already exists and ``force`` is False,
    the cached file is parsed instead of re-fetching. Otherwise the XML is
    downloaded, saved raw to disk, and then parsed.
    """
    if ACTIVE_STATIONS_XML.exists() and not force:
        logger.info("Using cached station list: %s", ACTIVE_STATIONS_XML)
        return parse_station_xml(ACTIVE_STATIONS_XML)

    logger.info("Fetching station list from %s", ACTIVE_STATIONS_URL)
    sess = session or _session()
    resp = sess.get(ACTIVE_STATIONS_URL, timeout=REQUEST_TIMEOUT_SEC)
    resp.raise_for_status()

    ACTIVE_STATIONS_XML.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_STATIONS_XML.write_bytes(resp.content)
    logger.info("Saved raw station list to %s", ACTIVE_STATIONS_XML)

    return parse_station_xml(ACTIVE_STATIONS_XML)


def _looks_like_missing_file(text: str) -> bool:
    """Detect NDBC's soft 'file not found' responses.

    The view_text_file.php endpoint sometimes returns HTTP 200 with a tiny HTML
    error page instead of a 404 when a station/year file does not exist. Real
    stdmet files begin with a ``#``-prefixed header line, never with HTML.
    """
    head = text.lstrip()[:200].lower()
    return head.startswith("<") or "unable to" in head or "not found" in head


def download_station_year(
    station: str,
    year: int,
    force: bool = False,
    session: Optional[requests.Session] = None,
    delay: float = REQUEST_DELAY_SEC,
) -> Optional[Path]:
    """Download one station's historical stdmet file for one year.

    Saves to ``data/raw/stdmet/{station}/{station}h{year}.txt`` and returns that
    path. Returns the cached path without re-downloading if it already exists
    (and ``force`` is False). Returns ``None`` if NDBC has no such file (a 404
    or soft error) — this is expected for many station/year combinations and is
    logged, not raised.
    """
    out_path = STDMET_DIR / station / f"{station}h{year}.txt"
    if out_path.exists() and not force:
        logger.info("Cached: %s", out_path)
        return out_path

    url = STDMET_URL_TEMPLATE.format(station=station, year=year)
    sess = session or _session()

    time.sleep(delay)  # be polite between requests
    resp = sess.get(url, timeout=REQUEST_TIMEOUT_SEC)

    if resp.status_code == 404:
        logger.info("No file for %s %d (404) — skipping.", station, year)
        return None
    resp.raise_for_status()

    if _looks_like_missing_file(resp.text):
        logger.info("No file for %s %d (soft 404) — skipping.", station, year)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(resp.text)
    logger.info("Downloaded %s %d -> %s", station, year, out_path)
    return out_path


def main() -> int:
    """Fetch the station list and print a short summary.

    Intentionally does NOT download historical files for every station — that
    would be a huge crawl. Per-station/year downloads are driven explicitly by
    callers of :func:`download_station_year`.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    config = load_config()
    years = years_from_config(config)

    stations = fetch_station_list()
    n_met = int(stations["met"].sum())

    print("\n=== NDBC station list ===")
    print(f"Total active stations : {len(stations)}")
    print(f"  with met program    : {n_met}")
    print(f"Target years (config) : {years[0]}-{years[-1]} ({len(years)} years)")
    print("\nFirst few stations:")
    print(stations.head().to_string(index=False))
    print(
        "\nThis script only fetched the station list. Use "
        "download_station_year(station, year) to pull historical stdmet files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
