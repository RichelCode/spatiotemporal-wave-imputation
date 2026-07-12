"""Does graph connectivity predict the GRAPH'S BENEFIT for outage recovery?

The raw connectivity-vs-error correlation is confounded: dense graph regions are
high-energy oceans with intrinsically bigger errors. This module controls for
that two INDEPENDENT ways and checks whether they agree.

Analysis 1 - normalized skill (confound removed by per-station normalization):
    Divide each victim's GRIN recovery MAE by that station's intrinsic wave
    variability (per-station std from the raw tensor, train period), then
    correlate the normalized error against connectivity. A high-variance station
    naturally has a bigger absolute MAE; normalizing removes the "big waves =
    big error" effect.

Analysis 2 - GRIN-vs-SAITS per-victim advantage (isolates the graph directly):
    advantage = SAITS_MAE - GRIN_MAE per victim (positive = GRIN better). SAITS
    is temporal-only, so the advantage is the graph's marginal contribution on
    the SAME station under the SAME outage. Correlate advantage vs connectivity.
    Hypothesis: the graph's advantage is LARGER at better-connected stations.

Both use the full-test station-outage masks (single scenario/scale). Connectivity
measures (from adjacency_knn_basin): node degree, mean neighbour distance (km),
edge-weight sum. Targets WVHT and APD are kept equally. Spearman rho throughout.

Outputs: reports/figures/fig9_connectivity_recovery.{png,pdf} and the merged
per-station table reports/connectivity_analysis.csv.

Run:  ``python -m src.evaluation.connectivity_analysis``
Importable; nothing runs on import.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..features.eda import FIG_DIR, TARGET_COLORS, set_style
from ..features.preprocess import load_meta

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_NPZ = PROCESSED_DIR / "wave_graph.npz"
RAW_TENSOR_NPZ = PROCESSED_DIR / "wave_tensor.npz"     # physical units (pre-normalisation)
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
GRIN_PER_VICTIM = PROJECT_ROOT / "reports" / "grin_per_victim_outage.csv"
SAITS_PER_VICTIM = PROJECT_ROOT / "reports" / "saits_per_victim_outage.csv"
OUT_CSV = PROJECT_ROOT / "reports" / "connectivity_analysis.csv"

TARGETS = ["WVHT", "APD"]
MEASURES = [
    ("degree", "node degree (# neighbours)"),
    ("mean_nbr_dist_km", "mean neighbour distance (km)"),
    ("edge_weight_sum", "edge-weight sum (connectivity)"),
]
CONFIG = "station_outage_full"


def compute_connectivity(adjacency_key: str = "adjacency_knn_basin") -> pd.DataFrame:
    """Per-station degree, mean neighbour distance (km) and edge-weight sum."""
    z = np.load(GRAPH_NPZ)
    A = z[adjacency_key].astype(np.float64)
    D = z["distance_km"].astype(np.float64)
    station_ids = list(json.load(open(AXES_JSON))["station_ids"])
    rows = []
    for s in range(A.shape[0]):
        nbrs = np.nonzero(A[s])[0]
        rows.append({
            "station_idx": s, "station_id": station_ids[s], "degree": int(len(nbrs)),
            "mean_nbr_dist_km": float(D[s, nbrs].mean()) if len(nbrs) else np.nan,
            "edge_weight_sum": float(A[s].sum()),
        })
    return pd.DataFrame(rows)


def per_station_std(meta: dict) -> pd.DataFrame:
    """Per-station intrinsic std of WVHT/APD from raw (physical) train-period data."""
    raw = np.load(RAW_TENSOR_NPZ)["tensor"]
    train_end = int(meta["split"]["train"]["end"])
    feats = meta["feature_names"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(raw[:train_end], axis=0)  # [S, F]
    return pd.DataFrame({
        "station_idx": np.arange(std.shape[0]),
        "WVHT_std": std[:, feats.index("WVHT")],
        "APD_std": std[:, feats.index("APD")],
    })


def load_recovery(csv_path: Path, config: str = CONFIG) -> pd.DataFrame:
    """Mean per-victim MAE (over seeds) per station, wide over targets."""
    pv = pd.read_csv(csv_path)
    sub = pv[(pv["config"] == config) & (pv["metric"] == "MAE")]
    agg = sub.groupby(["station_idx", "target"])["value"].mean().reset_index()
    wide = agg.pivot_table(index="station_idx", columns="target", values="value")
    return wide.reset_index()


def _spearman(df: pd.DataFrame, ycol: str) -> dict:
    out = {}
    for measure, _ in MEASURES:
        v = df[[measure, ycol]].dropna()
        rho, p = stats.spearmanr(v[measure], v[ycol])
        out[measure] = (rho, p, len(v))
    return out


def _print_block(title: str, df: pd.DataFrame, ycols: list[str]) -> dict:
    print(f"\n{title}")
    print(f"{'y':16s}{'measure':22s}{'rho':>8s}{'p':>9s}{'n':>5s}")
    lookup = {}
    for ycol in ycols:
        res = _spearman(df, ycol)
        lookup[ycol] = res
        for measure, _ in MEASURES:
            rho, p, n = res[measure]
            print(f"{ycol:16s}{measure:22s}{rho:+8.2f}{p:9.3f}{n:5d}")
    return lookup


def make_figure(df: pd.DataFrame, panels: list[tuple], saved: list[str]) -> None:
    """panels: list of (row_label, ycol, ylabel, color)."""
    set_style()
    fig, axes = plt.subplots(len(panels), len(MEASURES), figsize=(14, 4.2 * len(panels)))
    if len(panels) == 1:
        axes = axes[None, :]
    for r, (label, ycol, ylabel, color) in enumerate(panels):
        for c, (measure, xlabel) in enumerate(MEASURES):
            ax = axes[r, c]
            v = df[[measure, ycol]].dropna()
            x, y = v[measure].to_numpy(), v[ycol].to_numpy()
            ax.scatter(x, y, s=28, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
            if len(x) > 2:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 50)
                ax.plot(xs, m * xs + b, color="black", linewidth=1.5)
            rho, p = stats.spearmanr(x, y)
            ax.set_title(rf"$\rho$={rho:+.2f} (p={p:.3f}, n={len(x)})", fontsize=10)
            if r == len(panels) - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"{label}\n{ylabel}")
    fig.suptitle("Figure 9. Confound-controlled: connectivity vs GRIN outage benefit "
                 "(full-test station-outage)", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"fig9_connectivity_recovery.{ext}"
        fig.savefig(path)
        saved.append(path.name)
    plt.close(fig)


def _verdict(norm_lu: dict, adv_lu: dict | None) -> None:
    """Does connectivity predict the graph's benefit? Reported honestly as
    DIRECTIONAL agreement across the two controls plus a significance count
    (simultaneous significance is not required — it is underpowered at n~35)."""
    print("\n" + "=" * 70)
    print("VERDICT — connectivity vs the graph's benefit (two independent controls)")
    print("=" * 70)
    # 'Graph helps more when better connected' -> expected signs:
    #   normalized error: degree(-), dist(+), edge(-)
    #   advantage:        degree(+), dist(-), edge(+)
    expect_norm = {"degree": -1, "mean_nbr_dist_km": +1, "edge_weight_sum": -1}
    expect_adv = {"degree": +1, "mean_nbr_dist_km": -1, "edge_weight_sum": +1}
    for tgt in TARGETS:
        norm = norm_lu[f"{tgt}_norm"]
        dir_agree = norm_sig = adv_sig = 0
        for measure, _ in MEASURES:
            n_rho, n_p, _ = norm[measure]
            n_dir = np.sign(n_rho) == expect_norm[measure]
            n_ok = n_dir and n_p < 0.05
            norm_sig += n_ok
            line = f"    {measure:18s} norm rho={n_rho:+.2f}(p={n_p:.3f})"
            if adv_lu is not None:
                a_rho, a_p, _ = adv_lu[f"{tgt}_adv"][measure]
                a_dir = np.sign(a_rho) == expect_adv[measure]
                a_ok = a_dir and a_p < 0.05
                adv_sig += a_ok
                both_dir = n_dir and a_dir
                dir_agree += both_dir
                line += (f" | adv rho={a_rho:+.2f}(p={a_p:.3f}) -> "
                         f"{'SAME direction' if both_dir else 'diverge'}"
                         f"{'  [both sig]' if (n_ok and a_ok) else ''}")
            else:
                dir_agree += n_dir
            print(line)
        note = ("directionally robust" if dir_agree >= 2 else
                "mixed direction" if dir_agree == 1 else "no consistent direction")
        print(f"  {tgt}: {dir_agree}/3 measures point the SAME way across both controls "
              f"({note}); significant: normalized {norm_sig}/3, advantage {adv_sig}/3\n")


def main() -> int:
    meta = load_meta()
    conn = compute_connectivity()
    grin = load_recovery(GRIN_PER_VICTIM).rename(columns={"WVHT": "WVHT_MAE", "APD": "APD_MAE"})
    std = per_station_std(meta)

    df = grin.merge(conn, on="station_idx").merge(std, on="station_idx")
    df["WVHT_norm"] = df["WVHT_MAE"] / df["WVHT_std"]
    df["APD_norm"] = df["APD_MAE"] / df["APD_std"]

    have_saits = SAITS_PER_VICTIM.exists()
    if have_saits:
        saits = load_recovery(SAITS_PER_VICTIM).rename(
            columns={"WVHT": "SAITS_WVHT_MAE", "APD": "SAITS_APD_MAE"})
        df = df.merge(saits, on="station_idx", how="left")
        df["WVHT_adv"] = df["SAITS_WVHT_MAE"] - df["WVHT_MAE"]
        df["APD_adv"] = df["SAITS_APD_MAE"] - df["APD_MAE"]

    df.to_csv(OUT_CSV, index=False)

    print("=" * 70)
    print(f"CONNECTIVITY vs GRIN OUTAGE RECOVERY (full-test) — n={len(df)} victim stations")
    print("=" * 70)
    _print_block("[reference] RAW error vs connectivity (confounded):", df, ["WVHT_MAE", "APD_MAE"])
    norm_lu = _print_block("[Analysis 1] NORMALIZED error (MAE/std) vs connectivity:",
                           df, ["WVHT_norm", "APD_norm"])
    adv_lu = None
    if have_saits:
        adv_lu = _print_block("[Analysis 2] GRAPH ADVANTAGE (SAITS-GRIN MAE) vs connectivity:",
                              df, ["WVHT_adv", "APD_adv"])
    else:
        print("\n[Analysis 2] SKIPPED — reports/saits_per_victim_outage.csv not found "
              "(run: python -m src.models.saits_imputer --outage-per-victim)")

    panels = [
        ("WVHT norm.", "WVHT_norm", "MAE / std", TARGET_COLORS["WVHT"]),
        ("APD norm.", "APD_norm", "MAE / std", TARGET_COLORS["APD"]),
    ]
    if have_saits:
        panels += [
            ("WVHT advantage", "WVHT_adv", "SAITS-GRIN (m)", TARGET_COLORS["WVHT"]),
            ("APD advantage", "APD_adv", "SAITS-GRIN (s)", TARGET_COLORS["APD"]),
        ]
    saved: list[str] = []
    make_figure(df, panels, saved)

    _verdict(norm_lu, adv_lu)
    print(f"Wrote {OUT_CSV} and {', '.join(saved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
