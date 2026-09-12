"""Regenerate the forecast/energy skill-vs-horizon figures from the committed CSVs.

Reads reports/forecast_horizon_sweep.csv + forecast_deep_results.csv (and any other
deep-forecaster result CSVs matching forecast_*_results.csv) and
reports/energy_forecast_results.csv, and writes:

  fig10_horizon_skill.png/pdf  -- forecast skill vs horizon, WVHT and APD panels
  fig11_energy_forecast.png/pdf -- wave-energy skill vs horizon

Every learned method present in the CSVs gets a line, so the plots grow
automatically as new models are added. Persistence is the zero-skill reference.
Run:  python reports/make_forecast_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent))
from src.features.eda import FIG_DIR, OKABE_ITO, set_style  # noqa: E402

# Stable colour + marker per method (paradigm-coded, colour-blind safe).
STYLE = {
    "ar24":         ("AR(24)",        OKABE_ITO["orange"],     "s"),
    "dlinear":      ("DLinear",       OKABE_ITO["vermillion"], "^"),
    "patchtst":     ("PatchTST",      OKABE_ITO["yellow"],     "v"),
    "itransformer": ("iTransformer",  OKABE_ITO["purple"],     "D"),
    "graphwavenet": ("GraphWaveNet",  OKABE_ITO["blue"],       "o"),
    "dcrnn":        ("DCRNN",         OKABE_ITO["skyblue"],    "P"),
    "agcrn":        ("AGCRN",         OKABE_ITO["green"],      "X"),
}
HORIZONS = [1, 3, 6, 12, 24]


def _load_forecast() -> pd.DataFrame:
    frames = [pd.read_csv(R / "forecast_horizon_sweep.csv")]
    for f in sorted(R.glob("forecast_*_results.csv")):
        if "baseline" in f.name:              # horizon-less h=1 baseline; skip
            continue
        d = pd.read_csv(f)
        if "horizon" in d.columns:
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["method", "target", "horizon"])


def _methods_in(df, col="method"):
    present = [m for m in STYLE if m in set(df[col].unique())]
    return present


def fig_forecast(df) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    unit = {"WVHT": "m", "APD": "s"}
    methods = _methods_in(df)
    for ax, tgt in zip(axes, ("WVHT", "APD")):
        ax.axhline(0, color="black", lw=0.8, ls=":", label="persistence")
        for m in methods:
            label, color, marker = STYLE[m]
            d = df[(df.method == m) & (df.target == tgt)].sort_values("horizon")
            if len(d):
                ax.plot(d.horizon, d.skill_vs_persistence, marker=marker, color=color,
                        label=label, linewidth=1.8, markersize=6)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("forecast horizon (hours)")
        ax.set_ylabel(f"{tgt} skill vs persistence")
        ax.set_title(f"{tgt} ({unit[tgt]})")
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Figure 10. Forecast skill vs horizon", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "fig10_horizon_skill")


def fig_energy() -> list[str]:
    e = pd.read_csv(R / "energy_forecast_results.csv")
    set_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for m in _methods_in(e):
        label, color, marker = STYLE[m]
        d = e[e.method == m].sort_values("horizon")
        if len(d):
            ax.plot(d.horizon, d.skill_vs_persistence, marker=marker, color=color,
                    label=label, linewidth=1.8, markersize=6)
    ax.axhline(0, color="black", lw=0.8, ls=":", label="persistence")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("forecast horizon (hours)")
    ax.set_ylabel("wave-energy skill vs persistence")
    ax.set_title("Figure 11. Wave-energy forecast skill vs horizon", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, "fig11_energy_forecast")


def _save(fig, name) -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(p.name)
    plt.close(fig)
    return out


def main() -> int:
    df = _load_forecast()
    saved = fig_forecast(df) + fig_energy()
    print("wrote:", ", ".join(saved))
    print("forecast methods plotted:", _methods_in(df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
