"""Exploratory data analysis — publication-ready figures for the paper.

Reads the spatiotemporal tensor (``data/processed/wave_tensor.npz`` + axes
sidecar) and the frozen station table (``reports/selected_stations.csv``), and
produces seven figures saved to ``reports/figures/`` as both 300-dpi PNG and
vector PDF. Each figure is its own function and prints the summary statistics it
is built on to the console.

Style is consistent and paper-oriented: readable fonts, axes labelled with
units, an Okabe-Ito colourblind-safe palette, and minimal chartjunk.

The modelled targets are WVHT (m) and APD (s); DPD (s) is retained as a predictor
feature, not a target. There are no wind features in this tensor (see
build_tensor.py for the rationale).

Run:  ``python -m src.features.eda``   (builds all figures)
Importable; nothing runs on import.
"""

from __future__ import annotations

import json
import logging
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, file-only rendering

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import stats

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TENSOR_NPZ = PROCESSED_DIR / "wave_tensor.npz"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
SELECTED_CSV = PROJECT_ROOT / "reports" / "selected_stations.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Okabe-Ito colourblind-safe qualitative palette.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
# Fixed basin -> colour map (stable across all figures).
BASIN_COLORS = {
    "W. Coast": OKABE_ITO["blue"],
    "Pacific Is.": OKABE_ITO["skyblue"],
    "Gulf": OKABE_ITO["vermillion"],
    "S. Atlantic": OKABE_ITO["orange"],
    "Mid-Atlantic": OKABE_ITO["green"],
    "NE": OKABE_ITO["purple"],
    "Other/Intl": OKABE_ITO["black"],
    "Great Lakes": OKABE_ITO["yellow"],
}
# Per-target display colours and unit labels.
TARGET_COLORS = {"WVHT": OKABE_ITO["blue"], "DPD": OKABE_ITO["vermillion"], "APD": OKABE_ITO["green"]}
TARGET_UNITS = {"WVHT": "m", "DPD": "s", "APD": "s"}

MIN_PAIR_OVERLAP = 1000  # minimum overlapping observed hours for a station-pair correlation
EARTH_RADIUS_KM = 6371.0


def set_style() -> None:
    """Apply a consistent, paper-ready matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return float(2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))


def load_data():
    """Load tensor, mask, time/station/feature axes and the station table."""
    z = np.load(TENSOR_NPZ)
    tensor, mask = z["tensor"], z["mask"]
    axes = json.load(open(AXES_JSON))
    times = pd.DatetimeIndex(pd.to_datetime(axes["timestamps"], utc=True))
    station_ids = list(axes["station_ids"])
    features = list(axes["feature_names"])
    # Align the station table to the tensor's station axis order.
    stations = (
        pd.read_csv(SELECTED_CSV, dtype={"station_id": str})
        .set_index("station_id")
        .loc[station_ids]
        .reset_index()
    )
    return tensor, mask, times, station_ids, features, stations


def save_fig(fig, name: str, saved: list[str]) -> None:
    """Save a figure as both PNG (300 dpi) and PDF; record the filenames."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path)
        saved.append(path.name)
    plt.close(fig)
    print(f"  saved {name}.png + {name}.pdf")


# ---------------------------------------------------------------------------
# Figure 1 — station map
# ---------------------------------------------------------------------------
def fig1_station_map(stations, saved):
    print("\n[fig1] station map")
    used_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f2f2f2")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        scatter_kw = dict(transform=ccrs.PlateCarree())
        used_cartopy = True
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter_kw = dict()

    for basin in sorted(stations["basin"].unique()):
        sub = stations[stations["basin"] == basin]
        ax.scatter(
            sub["lon"], sub["lat"], s=38, color=BASIN_COLORS.get(basin, "#777777"),
            edgecolor="white", linewidth=0.4, label=f"{basin} (n={len(sub)})",
            zorder=3, **scatter_kw,
        )

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Figure 1. NDBC wave-buoy network (140 frozen stations)")
    ax.legend(loc="lower left", title="Basin", ncol=2)
    if not used_cartopy:
        # Approximate an equal-area look by scaling for mean latitude.
        mean_lat = np.radians(stations["lat"].mean())
        ax.set_aspect(1.0 / max(np.cos(mean_lat), 0.2))
    save_fig(fig, "fig1_station_map", saved)

    print("  stations per basin:")
    for basin, n in stations["basin"].value_counts().items():
        print(f"    {basin:14s}: {n}")
    print(f"  cartopy used: {used_cartopy}")


# ---------------------------------------------------------------------------
# Figure 2 — WVHT missingness heatmap
# ---------------------------------------------------------------------------
def fig2_missingness(mask, times, stations, features, saved):
    print("\n[fig2] WVHT missingness heatmap")
    wvht = features.index("WVHT")
    order = stations.sort_values(["basin", "station_id"], kind="stable").index.to_numpy()
    M = mask[:, order, wvht].T.astype(float)  # [station, time], 1=observed

    times_naive = times.tz_convert(None)
    x0, x1 = mdates.date2num(times_naive[0]), mdates.date2num(times_naive[-1])

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = ListedColormap([OKABE_ITO["yellow"], OKABE_ITO["blue"]])  # missing, observed
    ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=[x0, x1, len(order), 0])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Basin separators + centred labels (stations are contiguous per basin here).
    ordered_basins = stations.iloc[order]["basin"].to_numpy()
    ticks, labels, start = [], [], 0
    for basin in pd.unique(ordered_basins):
        n = int((ordered_basins == basin).sum())
        if start > 0:
            ax.axhline(start, color="white", linewidth=0.8)
        ticks.append(start + n / 2)
        labels.append(basin)
        start += n
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.set_xlabel("Time")
    ax.set_ylabel("Station (grouped by basin)")
    ax.set_title("Figure 2. WVHT observation coverage over time")
    ax.legend(handles=[Patch(color=OKABE_ITO["blue"], label="observed"),
                       Patch(color=OKABE_ITO["yellow"], label="missing")],
              loc="upper right", ncol=2)
    save_fig(fig, "fig2_missingness", saved)
    print(f"  overall WVHT observed fraction: {M.mean():.3f}")


# ---------------------------------------------------------------------------
# Figure 3 — per-station coverage distribution
# ---------------------------------------------------------------------------
def fig3_coverage_dist(mask, features, saved):
    print("\n[fig3] per-station coverage")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name in ("WVHT", "DPD", "APD"):
        f = features.index(name)
        cov = np.sort(mask[:, :, f].mean(axis=0))[::-1]
        ax.plot(np.arange(1, len(cov) + 1), cov, label=name,
                color=TARGET_COLORS[name], linewidth=2)
        print(f"  {name}: min={cov.min():.3f}  median={np.median(cov):.3f}  mean={cov.mean():.3f}")
    ax.axhline(0.70, color="gray", linestyle="--", linewidth=1, label="0.70 selection bar")
    ax.set_xlabel("Station rank (sorted by coverage, per target)")
    ax.set_ylabel("Observed fraction over 2021–2025")
    ax.set_ylim(0, 1.02)
    ax.set_title("Figure 3. Per-station target coverage across the 140 stations")
    ax.legend(loc="lower left")
    save_fig(fig, "fig3_coverage_dist", saved)


# ---------------------------------------------------------------------------
# Figure 4 — pooled target distributions
# ---------------------------------------------------------------------------
def fig4_target_dists(tensor, mask, features, saved):
    print("\n[fig4] pooled target / feature distributions")
    # WVHT and APD are the modelled targets; DPD is a retained predictor feature.
    # Order targets first, feature last, and label each panel by its role so the
    # figure never implies DPD is a target.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    for ax, name in zip(axes, ("WVHT", "APD", "DPD")):
        f = features.index(name)
        vals = tensor[:, :, f][mask[:, :, f]]
        ax.hist(vals, bins=80, color=TARGET_COLORS[name], alpha=0.85)
        ax.set_xlabel(f"{name} ({TARGET_UNITS[name]})")
        ax.set_ylabel("count")
        role = "target" if name in ("WVHT", "APD") else "feature"
        ax.set_title(f"{name} ({role})")
        m, med = float(np.mean(vals)), float(np.median(vals))
        sd, sk = float(np.std(vals)), float(stats.skew(vals))
        ax.axvline(m, color="black", linestyle="-", linewidth=1)
        ax.axvline(med, color="black", linestyle="--", linewidth=1)
        print(f"  {name:5s}: mean={m:.3f} median={med:.3f} std={sd:.3f} skew={sk:+.3f} (n={vals.size:,})")
        if name == "WVHT":
            tag = "right-skewed (expected)" if sk > 0 else "NOT right-skewed (unexpected!)"
            print(f"         -> WVHT skew {sk:+.3f}: {tag}")
    fig.suptitle("Figure 4. Distributions of the targets (WVHT, APD) and retained feature (DPD)",
                 fontweight="bold")
    save_fig(fig, "fig4_target_dists", saved)


# ---------------------------------------------------------------------------
# Figure 5 — seasonal + diurnal WVHT cycles
# ---------------------------------------------------------------------------
def fig5_temporal_cycles(tensor, mask, times, features, saved):
    print("\n[fig5] seasonal + diurnal WVHT cycles")
    f = features.index("WVHT")
    field = np.where(mask[:, :, f], tensor[:, :, f], np.nan)
    with warnings.catch_warnings():
        # A handful of hours have zero stations reporting -> empty-slice NaN,
        # which is correct (those hours drop out of the cycle means below).
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spatial_mean = np.nanmean(field, axis=1)  # mean WVHT across stations, per hour
    s = pd.Series(spatial_mean, index=times)

    months = s.groupby(s.index.month)
    hours = s.groupby(s.index.hour)
    m_mean, m_std = months.mean(), months.std()
    h_mean, h_std = hours.mean(), hours.std()

    fig, (axm, axh) = plt.subplots(1, 2, figsize=(12, 4.5))
    axm.plot(m_mean.index, m_mean.values, color=OKABE_ITO["blue"], linewidth=2)
    axm.fill_between(m_mean.index, m_mean - m_std, m_mean + m_std,
                     color=OKABE_ITO["blue"], alpha=0.2)
    axm.set_xticks(range(1, 13))
    axm.set_xlabel("Month")
    axm.set_ylabel("Mean WVHT (m)")
    axm.set_title("Seasonal cycle")

    axh.plot(h_mean.index, h_mean.values, color=OKABE_ITO["vermillion"], linewidth=2)
    axh.fill_between(h_mean.index, h_mean - h_std, h_mean + h_std,
                     color=OKABE_ITO["vermillion"], alpha=0.2)
    axh.set_xticks(range(0, 24, 3))
    axh.set_xlabel("Hour of day (UTC)")
    axh.set_ylabel("Mean WVHT (m)")
    axh.set_title("Diurnal cycle")

    fig.suptitle("Figure 5. WVHT temporal cycles (spatial mean ± 1 SD)", fontweight="bold")
    save_fig(fig, "fig5_temporal_cycles", saved)
    print(f"  seasonal: peak month={int(m_mean.idxmax())} ({m_mean.max():.2f} m), "
          f"trough month={int(m_mean.idxmin())} ({m_mean.min():.2f} m)")
    print(f"  diurnal : range={h_mean.max() - h_mean.min():.3f} m "
          f"(peak hr={int(h_mean.idxmax())}, trough hr={int(h_mean.idxmin())})")


# ---------------------------------------------------------------------------
# Figure 6 — inter-target relationships
# ---------------------------------------------------------------------------
def fig6_target_relationships(tensor, mask, features, saved):
    print("\n[fig6] inter-target relationships")
    pairs = [("WVHT", "DPD"), ("WVHT", "APD"), ("DPD", "APD")]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    for ax, (a, b) in zip(axes, pairs):
        fa, fb = features.index(a), features.index(b)
        both = mask[:, :, fa] & mask[:, :, fb]
        x = tensor[:, :, fa][both]
        y = tensor[:, :, fb][both]
        pear = stats.pearsonr(x, y)[0]
        # Spearman on a deterministic subsample to bound runtime on millions of points.
        if x.size > 500_000:
            idx = rng.choice(x.size, 500_000, replace=False)
            spear = stats.spearmanr(x[idx], y[idx])[0]
        else:
            spear = stats.spearmanr(x, y)[0]
        # Hexbin on a deterministic subsample for a light vector PDF.
        if x.size > 300_000:
            idx = rng.choice(x.size, 300_000, replace=False)
            xp, yp = x[idx], y[idx]
        else:
            xp, yp = x, y
        hb = ax.hexbin(xp, yp, gridsize=45, cmap="viridis", mincnt=1, bins="log")
        ax.set_xlabel(f"{a} ({TARGET_UNITS[a]})")
        ax.set_ylabel(f"{b} ({TARGET_UNITS[b]})")
        # Single-line panel title; put the correlations in an in-axes box so a
        # two-line title can never collide with the suptitle.
        ax.set_title(f"{a} vs {b}", pad=8)
        ax.text(0.04, 0.96, f"Pearson = {pear:.2f}\nSpearman = {spear:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
        ax.grid(False)
        fig.colorbar(hb, ax=ax, label="log count")
        print(f"  {a:4s}-{b:4s}: Pearson={pear:+.3f}  Spearman={spear:+.3f}  (n={x.size:,})")
    # No suptitle: the LaTeX \caption already labels this "Figure 6", so a figure
    # title here is redundant and was colliding with the per-axes panel titles.
    # Keeping only the three single-line panel titles cannot overlap anything.
    fig.tight_layout()
    save_fig(fig, "fig6_target_relationships", saved)


# ---------------------------------------------------------------------------
# Figure 7 — correlation vs distance (the key figure)
# ---------------------------------------------------------------------------
def fig7_corr_distance(tensor, mask, features, stations, saved):
    print("\n[fig7] WVHT pairwise correlation vs distance (KEY FIGURE)")
    f = features.index("WVHT")
    field = tensor[:, :, f]
    obs = mask[:, :, f]
    lats = stations["lat"].to_numpy()
    lons = stations["lon"].to_numpy()
    basins = stations["basin"].to_numpy()
    n_s = field.shape[1]

    dists, corrs, within = [], [], []
    for i, j in combinations(range(n_s), 2):
        ov = obs[:, i] & obs[:, j]
        n = int(ov.sum())
        if n < MIN_PAIR_OVERLAP:
            continue
        xi = field[ov, i] - field[ov, i].mean()
        xj = field[ov, j] - field[ov, j].mean()
        denom = np.sqrt((xi * xi).sum() * (xj * xj).sum())
        if denom == 0:
            continue
        corrs.append(float((xi * xj).sum() / denom))
        dists.append(haversine(lats[i], lons[i], lats[j], lons[j]))
        within.append(basins[i] == basins[j])

    dists = np.array(dists)
    corrs = np.array(corrs)
    within = np.array(within)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(dists[~within], corrs[~within], s=8, alpha=0.25,
               color=OKABE_ITO["orange"], label="cross-basin", zorder=2)
    ax.scatter(dists[within], corrs[within], s=8, alpha=0.35,
               color=OKABE_ITO["blue"], label="within-basin", zorder=3)

    # Binned mean trend over all pairs.
    nbins = 25
    edges = np.linspace(0, dists.max(), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(dists, edges) - 1
    trend = [corrs[idx == b].mean() if np.any(idx == b) else np.nan for b in range(nbins)]
    ax.plot(centers, trend, color="black", linewidth=2.2, label="binned mean", zorder=4)

    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Great-circle distance (km)")
    ax.set_ylabel("WVHT correlation (Pearson, overlapping hours)")
    ax.set_title("Figure 7. WVHT spatial correlation vs distance")
    ax.legend(loc="upper right")
    save_fig(fig, "fig7_corr_distance", saved)

    near = corrs[dists < 100]
    far = corrs[dists > 2000]
    print(f"  pairs used (overlap >= {MIN_PAIR_OVERLAP}h): {len(corrs):,} of {n_s*(n_s-1)//2:,}")
    print(f"  distance range: {dists.min():.0f}–{dists.max():.0f} km")
    print(f"  mean corr <100 km : {near.mean():+.3f} (n={near.size})")
    if far.size:
        print(f"  mean corr >2000 km: {far.mean():+.3f} (n={far.size})")
    print(f"  within-basin mean corr: {corrs[within].mean():+.3f}; "
          f"cross-basin mean corr: {corrs[~within].mean():+.3f}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_style()
    tensor, mask, times, station_ids, features, stations = load_data()
    print(f"Loaded tensor {tensor.shape}; features={features}; {len(station_ids)} stations.")

    saved: list[str] = []
    fig1_station_map(stations, saved)
    fig2_missingness(mask, times, stations, features, saved)
    fig3_coverage_dist(mask, features, saved)
    fig4_target_dists(tensor, mask, features, saved)
    fig5_temporal_cycles(tensor, mask, times, features, saved)
    fig6_target_relationships(tensor, mask, features, saved)
    fig7_corr_distance(tensor, mask, features, stations, saved)

    print(f"\nSaved {len(saved)} files to {FIG_DIR}:")
    for name in saved:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
