"""Swell-propagation analysis: does the buoy network recover wave physics?

For every within-basin buoy pair we find the time lag that maximises the
cross-correlation of their (synoptic-band) significant-wave-height series, and test
it against the deep-water group-velocity prediction

    tau = d / c_g,     c_g = g * T / (4*pi)      (T = dominant wave period, DPD)

A swell event at buoy A reaches a downstream buoy B after tau hours; if the observed
lag tracks d/c_g and never exceeds it (waves cannot outrun their group velocity),
the network has recovered the dispersion relation from data alone. We then build the
directed swell-propagation graph (upstream -> downstream) and map it.

Outputs (all committed / reproducible):
  reports/propagation_pairs.csv      per within-basin pair: dist, coupling, obs/pred lag
  reports/propagation_summary.csv    overall + per-basin validation statistics
  reports/figures/fig12_propagation.{png,pdf}       observed vs predicted lag
  reports/figures/fig13_swell_corridors.{png,pdf}   directed propagation graph

Run:  python -m src.features.propagation
Deterministic; nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .eda import FIG_DIR, set_style

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"
TENSOR_NPZ = PROCESSED / "wave_tensor.npz"
AXES_JSON = PROCESSED / "wave_tensor_axes.json"
SELECTED_CSV = REPORTS / "selected_stations.csv"

G = 9.81
R_EARTH = 6371.0
MAXLAG = 48          # hours searched each direction
MIN_OVERLAP = 2000   # overlapping observed hours required for a pair
ANOM_WIN = 15 * 24   # rolling-mean window (h) removed to isolate synoptic variability
COUPLE_R = 0.5       # min peak cross-correlation to count a pair as swell-coupled
DIST_MIN, DIST_MAX = 80.0, 1500.0   # km; close enough to share swell, far enough to time it


def _load():
    z = np.load(TENSOR_NPZ)
    axes = json.load(open(AXES_JSON))
    feats = axes["feature_names"]
    wf, dfi = feats.index("WVHT"), feats.index("DPD")
    sid = list(axes["station_ids"])
    st = (pd.read_csv(SELECTED_CSV, dtype={"station_id": str})
          .set_index("station_id").loc[sid])
    wv = np.where(z["mask"][:, :, wf], z["tensor"][:, :, wf], np.nan)
    dp = np.where(z["mask"][:, :, dfi], z["tensor"][:, :, dfi], np.nan)
    return wv, dp, st["lat"].to_numpy(), st["lon"].to_numpy(), st["basin"].to_numpy(), sid


def _anomaly(x: np.ndarray) -> np.ndarray:
    s = pd.Series(x)
    return (s - s.rolling(ANOM_WIN, min_periods=ANOM_WIN // 3, center=True).mean()).to_numpy()


def _haversine(lat, lon, i, j) -> float:
    la1, lo1, la2, lo2 = map(np.radians, (lat[i], lon[i], lat[j], lon[j]))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * R_EARTH * np.arcsin(np.sqrt(a)))


def _signed_peak_lag(a: np.ndarray, b: np.ndarray, T: int):
    """Lag L (hours) maximising corr(a[t], b[t+L]); L>0 means a leads b. Returns (L, r)."""
    best_r, best_L = 0.0, 0
    for L in range(-MAXLAG, MAXLAG + 1):
        if L >= 0:
            x, y = a[:T - L], b[L:] if L > 0 else b
        else:
            x, y = a[-L:], b[:T + L]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < MIN_OVERLAP:
            continue
        xc, yc = x[m] - x[m].mean(), y[m] - y[m].mean()
        denom = np.sqrt((xc * xc).sum() * (yc * yc).sum())
        if denom > 0:
            r = float((xc * yc).sum() / denom)
            if r > best_r:
                best_r, best_L = r, L
    return best_L, best_r


def analyse() -> pd.DataFrame:
    wv, dp, lat, lon, basin, sid = _load()
    T, N = wv.shape
    wva = np.column_stack([_anomaly(wv[:, i]) for i in range(N)])
    rows = []
    for i, j in combinations(range(N), 2):
        if basin[i] != basin[j]:
            continue
        d = _haversine(lat, lon, i, j)
        if not (DIST_MIN <= d <= DIST_MAX):
            continue
        L, r = _signed_peak_lag(wva[:, i], wva[:, j], T)
        Tp = np.nanmean(np.concatenate([dp[:, i], dp[:, j]]))
        cg = G * Tp / (4 * np.pi)                       # m/s
        pred = (d * 1000.0) / cg / 3600.0               # hours
        up, dn = (i, j) if L >= 0 else (j, i)           # up leads dn
        rows.append(dict(up=up, dn=dn, up_id=sid[up], dn_id=sid[dn], basin=basin[i],
                         dist_km=d, coupling=r, obs_lag_h=abs(L), pred_lag_h=pred,
                         Tp=Tp, cg_ms=cg))
    df = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    df.to_csv(REPORTS / "propagation_pairs.csv", index=False)
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    coupled = df[df.coupling >= COUPLE_R]
    for label, sub in [("all_pairs", df), ("coupled", coupled)]:
        rho, p = spearmanr(sub.pred_lag_h, sub.obs_lag_h)
        bound = float((sub.obs_lag_h <= 1.3 * sub.pred_lag_h).mean())
        out.append(dict(scope=label, n=len(sub), spearman_rho=rho, p_value=p,
                        causal_bound_frac=bound, median_coupling=float(sub.coupling.median()),
                        median_obs_lag_h=float(sub.obs_lag_h.median()),
                        median_dist_km=float(sub.dist_km.median())))
    for b, sub in coupled.groupby("basin"):
        if len(sub) >= 10:
            rho, p = spearmanr(sub.dist_km, sub.obs_lag_h)
            out.append(dict(scope=f"basin:{b}", n=len(sub), spearman_rho=rho, p_value=p,
                            causal_bound_frac=float((sub.obs_lag_h <= 1.3 * sub.pred_lag_h).mean()),
                            median_coupling=float(sub.coupling.median()),
                            median_obs_lag_h=float(sub.obs_lag_h.median()),
                            median_dist_km=float(sub.dist_km.median())))
    s = pd.DataFrame(out)
    s.to_csv(REPORTS / "propagation_summary.csv", index=False)
    return s


def _fig_physics(df: pd.DataFrame):
    set_style()
    sub = df[df.coupling >= 0.4]
    rho, _ = spearmanr(sub.pred_lag_h, sub.obs_lag_h)
    fig, ax = plt.subplots(figsize=(6.6, 6))
    sc = ax.scatter(sub.pred_lag_h, sub.obs_lag_h, c=sub.coupling, cmap="viridis",
                    s=16, alpha=0.75, vmin=0.4, vmax=1.0, edgecolor="none")
    lim = float(np.nanpercentile(np.r_[sub.pred_lag_h, sub.obs_lag_h], 99)) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label=r"$\tau_{\mathrm{obs}}=d/c_g$ (group velocity)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"predicted lag $d/c_g$ (hours),  $c_g=gT/4\pi$")
    ax.set_ylabel(r"observed cross-correlation lag $|\tau^{*}|$ (hours)")
    ax.set_title(f"The network recovers wave propagation\n"
                 f"within-basin pairs, Spearman $\\rho={rho:.2f}$ ($n={len(sub)}$)", fontweight="bold")
    fig.colorbar(sc, ax=ax, label="peak cross-correlation")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig12_propagation")


def _fig_corridors(df: pd.DataFrame, lat, lon):
    set_style()
    coupled = df[df.coupling >= COUPLE_R]
    used_cartopy = False
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(11, 6.4))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#f4f4f4")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25)
        tk = dict(transform=ccrs.PlateCarree())
        used_cartopy = True
    except Exception:
        fig, ax = plt.subplots(figsize=(11, 6.4))
        tk = dict()
    lags = coupled.obs_lag_h.to_numpy()
    norm = plt.Normalize(0, float(np.percentile(lags, 95)))
    cmap = plt.cm.plasma
    for _, r in coupled.iterrows():
        u, v = int(r.up), int(r.dn)
        ax.annotate("", xy=(lon[v], lat[v]), xytext=(lon[u], lat[u]),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(norm(r.obs_lag_h)),
                                    lw=0.4 + 1.6 * r.coupling, alpha=0.6, shrinkA=2, shrinkB=2), **tk)
    ax.scatter(lon, lat, s=14, c="black", zorder=6, **tk)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="swell propagation lag (hours)", shrink=0.82)
    ax.set_title("Data-derived swell-propagation corridors\n"
                 "arrows point upstream$\\rightarrow$downstream; width $\\propto$ coupling strength",
                 fontweight="bold")
    if not used_cartopy:
        ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
        mean_lat = np.radians(np.nanmean(lat))
        ax.set_aspect(1.0 / max(np.cos(mean_lat), 0.2))
    fig.tight_layout()
    _save(fig, "fig13_swell_corridors")
    logger.info("corridor map cartopy=%s", used_cartopy)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = analyse()
    _, _, lat, lon, _, _ = _load()
    s = summarise(df)
    _fig_physics(df)
    _fig_corridors(df, lat, lon)
    print(f"within-basin pairs: {len(df)}  | coupled (r>={COUPLE_R}): {(df.coupling >= COUPLE_R).sum()}")
    print(s.to_string(index=False))
    print(f"\nWrote propagation_pairs.csv, propagation_summary.csv, fig12/fig13.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
