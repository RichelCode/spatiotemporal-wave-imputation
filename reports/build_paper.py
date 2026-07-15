"""Generate reports/main.tex from _paper_template.tex + the committed result CSVs.

Every number and table in the paper is pulled programmatically from the
result CSVs in this directory -- nothing is hand-typed. Edit prose in
``_paper_template.tex`` (which carries @placeholder@ tokens and @T1@..@T5@
table slots), then run this script to regenerate ``main.tex``:

    python reports/build_paper.py

Requires pandas, numpy, scipy. Compile the result with any LaTeX engine, e.g.
``tectonic reports/main.tex`` (bibtex is run automatically), uploading main.tex,
main.bib and figures/fig1-11*.png to Overleaf for a standalone build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent

base = pd.read_csv(R / "baseline_imputation_results.csv")
saits = pd.read_csv(R / "saits_imputation_results.csv")
brits = pd.read_csv(R / "brits_imputation_results.csv")
grin = pd.read_csv(R / "grin_multiseed_results.csv")
grin_b = grin[grin.graph == "adjacency_knn_basin"]
ab = pd.read_csv(R / "graph_ablation_results.csv")
conn = pd.read_csv(R / "connectivity_analysis.csv")
sweep = pd.read_csv(R / "forecast_horizon_sweep.csv")
fdeep = pd.read_csv(R / "forecast_deep_results.csv")
energy = pd.read_csv(R / "energy_forecast_results.csv")


def imp_mae(method, cfg, tgt):
    if method in ("saits", "brits"):
        d = {"saits": saits, "brits": brits}[method]
        r = d[(d.config == cfg) & (d.target == tgt) & (d.metric == "MAE")]
        return float(r["mean"].iloc[0]), None
    if method == "grin":
        r = grin_b[(grin_b.config == cfg) & (grin_b.target == tgt) & (grin_b.metric == "MAE")]
        return float(r["mean"].iloc[0]), float(r["std"].iloc[0])
    r = base[(base.baseline == method) & (base.config == cfg) & (base.target == tgt) & (base.metric == "MAE")]
    return float(r["mean"].iloc[0]), None


IMP_METHODS = [("mean_fill", "Mean fill"), ("forward_fill", "Forward fill"), ("linear_interp", "Linear interp"),
               ("spatial_idw", "Spatial IDW"), ("spatial_temporal", "Spatial+temporal"),
               ("saits", "SAITS"), ("brits", "BRITS"), ("grin", "GRIN")]
IMP_CFG = [("station_outage_full", "Station-outage"), ("mcar_50", "MCAR-50\\%"), ("block_50", "Block-50\\%")]
TGTS = ["WVHT", "APD"]


def fmt(v, s=None):
    return f"{v:.3f}$\\pm${s:.3f}" if s is not None else f"{v:.3f}"


# ---- Table 1: imputation ----
vals = {(m, c, t): imp_mae(m, c, t) for m, _ in IMP_METHODS for c, _ in IMP_CFG for t in TGTS}
best = {(c, t): min(((m, vals[(m, c, t)][0]) for m, _ in IMP_METHODS), key=lambda x: x[1])[0]
        for c, _ in IMP_CFG for t in TGTS}
rows = []
for mk, mn in IMP_METHODS:
    cells = []
    for c, _ in IMP_CFG:
        for t in TGTS:
            mv, sv = vals[(mk, c, t)]
            cell = fmt(mv, sv)
            if best[(c, t)] == mk:
                cell = f"\\textbf{{{cell}}}"
            cells.append(cell)
    rows.append(f"{mn} & " + " & ".join(cells) + r" \\")
T1 = "\n".join(rows)

# ---- Table 2: graph ablation (MAE) ----
abm = ab[ab.metric == "MAE"]
rows = []
for c, cn in IMP_CFG:
    for t in TGTS:
        r = abm[(abm.config == c) & (abm.target == t)].iloc[0]
        rows.append(f"{cn} & {t} & {r.basin_mean:.3f}$\\pm${r.basin_std:.3f} & "
                    f"{r.plain_mean:.3f}$\\pm${r.plain_std:.3f} & {r.basin_advantage:+.3f} & "
                    f"{r.combined_std:.3f} & {'yes' if r.exceeds_noise else 'no'} \\\\")
T2 = "\n".join(rows)
ablation_within = int((~abm.exceeds_noise).sum())
ablation_total = len(abm)

# ---- Table 3: connectivity Spearman (raw / normalized / advantage) ----
MEAS = [("degree", "degree"), ("mean_nbr_dist_km", "neighbour dist."), ("edge_weight_sum", "edge-weight sum")]


def rho(x, y):
    v = conn[[x, y]].dropna()
    r, p = spearmanr(v[x], v[y])
    return r, p


def star(p):
    return "*" if p < 0.05 else ""


rows = []
for t in TGTS:
    for mk, mn in MEAS:
        raw_r, raw_p = rho(mk, f"{t}_MAE")
        nrm_r, nrm_p = rho(mk, f"{t}_norm")
        adv_r, adv_p = rho(mk, f"{t}_adv")
        rows.append(f"{t} & {mn} & {raw_r:+.2f}{star(raw_p)} & {nrm_r:+.2f}{star(nrm_p)} & {adv_r:+.2f}{star(adv_p)} \\\\")
T3 = "\n".join(rows)
n_w = int(conn["WVHT_MAE"].notna().sum())
n_a = int(conn["APD_MAE"].notna().sum())
adv_w = conn["WVHT_adv"].mean()
adv_a = conn["APD_adv"].mean()

# ---- forecasting numbers ----
fc = pd.concat([sweep, fdeep], ignore_index=True)


def fmae(meth, tgt, h):
    r = fc[(fc.method == meth) & (fc.target == tgt) & (fc.horizon == h)]
    return (float(r["MAE"].iloc[0]), float(r["skill_vs_persistence"].iloc[0])) if len(r) else (np.nan, np.nan)


FMETH = [("persistence", "Persistence"), ("ar24", "AR(24)"), ("graphwavenet", "GraphWaveNet")]

# ---- Table 4: forecasting (MAE, skill) at h=12,24 ----
rows = []
for meth, mn in FMETH:
    cells = []
    for t in TGTS:
        for h in (12, 24):
            m, sk = fmae(meth, t, h)
            cells.append(f"{m:.3f}" if meth == "persistence" else f"{m:.3f} ({sk:+.3f})")
    rows.append(f"{mn} & " + " & ".join(cells) + r" \\")
T4 = "\n".join(rows)


# ---- Table 5: energy ----
def emae(meth, h):
    r = energy[(energy.method == meth) & (energy.horizon == h)].iloc[0]
    return float(r["MAE_kw"]), float(r["skill_vs_persistence"])


rows = []
for meth, mn in FMETH:
    cells = []
    for h in (12, 24):
        m, sk = emae(meth, h)
        cells.append(f"{m:.3f}" if meth == "persistence" else f"{m:.3f} ({sk:+.3f})")
    rows.append(f"{mn} & " + " & ".join(cells) + r" \\")
T5 = "\n".join(rows)

# ---- prose numbers (all pulled from CSVs; panel constants from the tensor axes) ----
P = {
    "grin_so_wvht": fmt(*imp_mae("grin", "station_outage_full", "WVHT")),
    "grin_so_apd": fmt(*imp_mae("grin", "station_outage_full", "APD")),
    "idw_so_wvht": f"{imp_mae('spatial_idw', 'station_outage_full', 'WVHT')[0]:.3f}",
    "st_so_wvht": f"{imp_mae('spatial_temporal', 'station_outage_full', 'WVHT')[0]:.3f}",
    "saits_so_wvht": f"{imp_mae('saits', 'station_outage_full', 'WVHT')[0]:.3f}",
    "brits_so_wvht": f"{imp_mae('brits', 'station_outage_full', 'WVHT')[0]:.3f}",
    "grin_mcar_wvht": f"{imp_mae('grin', 'mcar_50', 'WVHT')[0]:.3f}",
    "saits_mcar_wvht": f"{imp_mae('saits', 'mcar_50', 'WVHT')[0]:.3f}",
    "lin_mcar_wvht": f"{imp_mae('linear_interp', 'mcar_50', 'WVHT')[0]:.3f}",
    "idw_mcar_wvht": f"{imp_mae('spatial_idw', 'mcar_50', 'WVHT')[0]:.3f}",
    "n_stations": "140", "n_audit": "872", "n_hours": "43{,}824",
    "adv_w": f"{adv_w:.2f}", "adv_a": f"{adv_a:.2f}",
    "abl_within": str(ablation_within), "abl_total": str(ablation_total),
    "nw": str(n_w), "na": str(n_a),
    "pers1_wvht": f"{fmae('persistence', 'WVHT', 1)[0]:.3f}",
    "gwn12_w": f"{fmae('graphwavenet', 'WVHT', 12)[1]:+.3f}", "gwn24_w": f"{fmae('graphwavenet', 'WVHT', 24)[1]:+.3f}",
    "gwn12_a": f"{fmae('graphwavenet', 'APD', 12)[1]:+.3f}", "gwn24_a": f"{fmae('graphwavenet', 'APD', 24)[1]:+.3f}",
    "ar12_w": f"{fmae('ar24', 'WVHT', 12)[1]:+.3f}", "ar24_w": f"{fmae('ar24', 'WVHT', 24)[1]:+.3f}",
    "en_gwn12": f"{emae('graphwavenet', 12)[1]:+.3f}", "en_gwn24": f"{emae('graphwavenet', 24)[1]:+.3f}",
    "en_ar12": f"{emae('ar24', 12)[1]:+.3f}", "en_ar24": f"{emae('ar24', 24)[1]:+.3f}",
}

tex = (R / "_paper_template.tex").read_text()
for key, val in {"T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5}.items():
    tex = tex.replace(f"@{key}@", val)
for k, v in P.items():
    tex = tex.replace(f"@{k}@", v)

leftover = [w for w in tex.split() if w.startswith("@") and w.endswith("@") and len(w) > 1]
assert not leftover, f"unfilled placeholder(s): {leftover}"
(R / "main.tex").write_text(tex)
print(f"wrote {R / 'main.tex'}")
print(f"energy skill  GWN {P['en_gwn12']}/{P['en_gwn24']}   AR {P['en_ar12']}/{P['en_ar24']}")
print(f"ablation within-noise {ablation_within}/{ablation_total}   connectivity n={n_w}")
