"""Generate reports/slides.tex (committee Beamer deck) from the committed CSVs.

Mirrors build_paper.py: every number and table is pulled programmatically from
the committed result CSVs in reports/ -- nothing is hand-typed. Numbers that
cannot be read from a committed file are emitted as \\textbf{[VERIFY]} and listed
at the end of the run.

    python reports/_slides_template.py

Writes slides.tex (clean) and slides_notes.tex (\\setbeameroption{show notes}).
Compile either with tectonic. Requires pandas, numpy, scipy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = Path(__file__).resolve().parent
VERIFY: list[str] = []


def verify(label: str) -> str:
    """Register a number not backed by a committed file; returns a LaTeX marker."""
    VERIFY.append(label)
    return r"\textbf{[VERIFY]}"


# --------------------------------------------------------------------------- #
# Load every committed CSV we use
# --------------------------------------------------------------------------- #
base = pd.read_csv(R / "baseline_imputation_results.csv")
saits = pd.read_csv(R / "saits_imputation_results.csv")
brits = pd.read_csv(R / "brits_imputation_results.csv")
grin_ms = pd.read_csv(R / "grin_multiseed_results.csv")
grin_b = grin_ms[grin_ms.graph == "adjacency_knn_basin"]
grin_p = grin_ms[grin_ms.graph == "adjacency_knn"]
ab = pd.read_csv(R / "graph_ablation_results.csv")
conn = pd.read_csv(R / "connectivity_analysis.csv")
sweep = pd.read_csv(R / "forecast_horizon_sweep.csv")
fdeep = pd.read_csv(R / "forecast_deep_results.csv")
energy = pd.read_csv(R / "energy_forecast_results.csv")
audit = pd.read_csv(R / "coverage_audit.csv")
thresh = pd.read_csv(R / "coverage_thresholds.csv")
selected = pd.read_csv(R / "selected_stations.csv")
grin_pv = pd.read_csv(R / "grin_per_victim_outage.csv")
saits_pv = pd.read_csv(R / "saits_per_victim_outage.csv")

# --------------------------------------------------------------------------- #
# Method -> paradigm colour (temporal / spatial / spatiotemporal graph)
# --------------------------------------------------------------------------- #
PARADIGM = {
    "Mean fill": "t", "Forward fill": "t", "Linear interp": "t",
    "Spatial IDW": "s", "Spatial+temporal": "s",
    "SAITS": "t", "BRITS": "t", "GRIN": "g",
    "Persistence": "t", "Seasonal-naive": "t", "AR(24)": "t", "GraphWaveNet": "g",
}
COL = {"t": "coltemporal", "s": "colspatial", "g": "colgraph"}


def cname(mn: str) -> str:
    return f"\\textcolor{{{COL[PARADIGM[mn]]}}}{{{mn}}}"


def fmt(v, s=None):
    return f"{v:.3f}$\\pm${s:.3f}" if s is not None else f"{v:.3f}"


# --------------------------------------------------------------------------- #
# Imputation lookups
# --------------------------------------------------------------------------- #
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

# ---- TAB_IMP: full imputation table (8 methods x 3 cfg x 2 tgt) ----
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
    rows.append(f"{cname(mn)} & " + " & ".join(cells) + r" \\")
TAB_IMP = "\n".join(rows)

# ---- TAB_CONN: connectivity Spearman ----
MEAS = [("degree", "degree"), ("mean_nbr_dist_km", "neighbour dist."), ("edge_weight_sum", "edge-weight sum")]


def rho(x, y):
    v = conn[[x, y]].dropna()
    r, p = spearmanr(v[x], v[y])
    return r, p


def star(p):
    return "$^{*}$" if p < 0.05 else ""


rows = []
for t in TGTS:
    for mk, mn in MEAS:
        rr, rp = rho(mk, f"{t}_MAE")
        nr, npv = rho(mk, f"{t}_norm")
        ar, ap = rho(mk, f"{t}_adv")
        rows.append(f"{t} & {mn} & {rr:+.2f}{star(rp)} & {nr:+.2f}{star(npv)} & {ar:+.2f}{star(ap)} \\\\")
TAB_CONN = "\n".join(rows)

# ---- TAB_ABL: headline ablation (3 cfg x 2 tgt) + full 16-cell for backup ----
abm = ab[ab.metric == "MAE"]
HEAD_CFG = [("station_outage_full", "Station-outage"), ("mcar_50", "MCAR-50\\%"), ("block_50", "Block-50\\%")]
rows = []
for c, cn in HEAD_CFG:
    for t in TGTS:
        r = abm[(abm.config == c) & (abm.target == t)].iloc[0]
        rows.append(f"{cn} & {t} & {r.basin_mean:.3f}$\\pm${r.basin_std:.3f} & "
                    f"{r.plain_mean:.3f}$\\pm${r.plain_std:.3f} & {r.basin_advantage:+.3f} & "
                    f"{'yes' if r.exceeds_noise else 'no'} \\\\")
TAB_ABL = "\n".join(rows)

ALL_CFG = [("mcar_10", "MCAR-10\\%"), ("mcar_30", "MCAR-30\\%"), ("mcar_50", "MCAR-50\\%"),
           ("block_10", "Block-10\\%"), ("block_30", "Block-30\\%"), ("block_50", "Block-50\\%"),
           ("station_outage_720", "Outage-720h"), ("station_outage_full", "Outage-full")]
rows = []
for c, cn in ALL_CFG:
    for t in TGTS:
        r = abm[(abm.config == c) & (abm.target == t)].iloc[0]
        rows.append(f"{cn} & {t} & {r.basin_mean:.3f} & {r.plain_mean:.3f} & {r.basin_advantage:+.3f} & "
                    f"{r.combined_std:.3f} & {'yes' if r.exceeds_noise else 'no'} \\\\")
TAB_ABL_FULL = "\n".join(rows)
abl_within = int((~abm.exceeds_noise).sum())
abl_total = len(abm)

# ---- forecasting ----
fc = pd.concat([sweep, fdeep], ignore_index=True)
FMETH = [("persistence", "Persistence"), ("ar24", "AR(24)"), ("graphwavenet", "GraphWaveNet")]


def fmae(meth, tgt, h):
    r = fc[(fc.method == meth) & (fc.target == tgt) & (fc.horizon == h)]
    return (float(r["MAE"].iloc[0]), float(r["skill_vs_persistence"].iloc[0])) if len(r) else (np.nan, np.nan)


rows = []
for meth, mn in FMETH:
    cells = []
    for t in TGTS:
        for h in (12, 24):
            m, sk = fmae(meth, t, h)
            cells.append(f"{m:.3f}" if meth == "persistence" else f"{m:.3f} ({sk:+.3f})")
    rows.append(f"{cname(mn)} & " + " & ".join(cells) + r" \\")
TAB_FC = "\n".join(rows)

# ---- energy ----
def emae(meth, h):
    r = energy[(energy.method == meth) & (energy.horizon == h)].iloc[0]
    return float(r["MAE_kw"]), float(r["skill_vs_persistence"])


rows = []
for meth, mn in [("persistence", "Persistence"), ("ar24", "AR(24)"), ("graphwavenet", "GraphWaveNet")]:
    cells = []
    for h in (12, 24):
        m, sk = emae(meth, h)
        cells.append(f"{m:.3f}" if meth == "persistence" else f"{m:.3f} ({sk:+.3f})")
    rows.append(f"{cname(mn)} & " + " & ".join(cells) + r" \\")
TAB_ENERGY = "\n".join(rows)

# ---- TAB_PARALLEL: imputation & forecasting tell the same story ----
idw_so_w = imp_mae("spatial_idw", "station_outage_full", "WVHT")[0]
saits_so_w = imp_mae("saits", "station_outage_full", "WVHT")[0]
grin_so_w, grin_so_w_sd = imp_mae("grin", "station_outage_full", "WVHT")
ar24_w_sk = fmae("ar24", "WVHT", 24)[1]
gwn24_w_sk = fmae("graphwavenet", "WVHT", 24)[1]
TAB_PARALLEL = (
    f"Imputation & station-outage WVHT MAE & {cname('Spatial IDW')} {idw_so_w:.3f}, "
    f"{cname('SAITS')} {saits_so_w:.3f} & \\textbf{{{cname('GRIN')} {grin_so_w:.3f}}} \\\\\n"
    f"Forecasting & 24h WVHT skill & {cname('AR(24)')} {ar24_w_sk:+.3f} & "
    f"\\textbf{{{cname('GraphWaveNet')} {gwn24_w_sk:+.3f}}} \\\\"
)

# ---- backup: full baseline table (WVHT, all mcar/block rates + outage) ----
BK_CFG = [("mcar_10", "MCAR-10\\%"), ("mcar_30", "MCAR-30\\%"), ("mcar_50", "MCAR-50\\%"),
          ("block_10", "Block-10\\%"), ("block_30", "Block-30\\%"), ("block_50", "Block-50\\%"),
          ("station_outage_full", "Outage")]
BK_METHODS = [("mean_fill", "Mean fill"), ("forward_fill", "Forward fill"), ("linear_interp", "Linear interp"),
              ("spatial_idw", "Spatial IDW"), ("spatial_temporal", "Spatial+temporal")]
rows = []
for mk, mn in BK_METHODS:
    cells = [f"{imp_mae(mk, c, 'WVHT')[0]:.3f}" for c, _ in BK_CFG]
    rows.append(f"{cname(mn)} & " + " & ".join(cells) + r" \\")
TAB_BASE_FULL = "\n".join(rows)

# ---- backup: BRITS full (both targets, all configs) ----
rows = []
for c, cn in ALL_CFG:
    cells = [f"{imp_mae('brits', c, t)[0]:.3f}" for t in TGTS]
    rows.append(f"{cn} & " + " & ".join(cells) + r" \\")
TAB_BRITS_FULL = "\n".join(rows)

# ---- backup: per-victim GRIN vs SAITS advantage distribution ----
def pv_advantage(tgt):
    gg = grin_pv[(grin_pv.config == "station_outage_full") & (grin_pv.target == tgt) &
                 (grin_pv.metric == "MAE")].groupby("station_id").value.mean()
    ss = saits_pv[(saits_pv.config == "station_outage_full") & (saits_pv.target == tgt) &
                  (saits_pv.metric == "MAE")].groupby("station_id").value.mean()
    both = pd.concat([gg.rename("g"), ss.rename("s")], axis=1).dropna()
    return both.s - both.g, both


adv_w, pw = pv_advantage("WVHT")
adv_a, pa = pv_advantage("APD")
rows = [
    f"WVHT & {len(adv_w)} & {adv_w.min():+.3f} & {adv_w.median():+.3f} & {adv_w.mean():+.3f} & "
    f"{adv_w.max():+.3f} & {int((adv_w > 0).sum())}/{len(adv_w)} \\\\",
    f"APD & {len(adv_a)} & {adv_a.min():+.3f} & {adv_a.median():+.3f} & {adv_a.mean():+.3f} & "
    f"{adv_a.max():+.3f} & {int((adv_a > 0).sum())}/{len(adv_a)} \\\\",
]
TAB_PV = "\n".join(rows)

# --------------------------------------------------------------------------- #
# Scalar prose numbers
# --------------------------------------------------------------------------- #
n_audit = len(audit)
n_wave = int((audit.WVHT_nonnull > 0).sum())
n_sel = len(selected)
n_hours = int(audit.expected_hours.iloc[0])
thr_row = thresh[thresh.wvht_coverage_threshold == 0.7].iloc[0]
thr140 = int(thr_row["min_years>=3"])
ar_lo = min(abs(fmae("ar24", t, h)[1]) for t in TGTS for h in (12, 24))
ar_hi = max(abs(fmae("ar24", t, h)[1]) for t in TGTS for h in (12, 24))

P = {
    "n_audit": str(n_audit), "n_wave": str(n_wave), "n_sel": str(n_sel),
    "n_hours": f"{n_hours:,}".replace(",", "{,}"), "thr140": str(thr140),
    "idw_so_w": f"{idw_so_w:.3f}",
    "idw_mc_w": f"{imp_mae('spatial_idw', 'mcar_50', 'WVHT')[0]:.3f}",
    "saits_so_w": f"{saits_so_w:.3f}",
    "saits_mc_w": f"{imp_mae('saits', 'mcar_50', 'WVHT')[0]:.3f}",
    "grin_so_w": fmt(grin_so_w, grin_so_w_sd),
    "grin_so_a": fmt(*imp_mae("grin", "station_outage_full", "APD")),
    "grin_mc_w": f"{imp_mae('grin', 'mcar_50', 'WVHT')[0]:.3f}",
    "advw": f"{adv_w.mean():.3f}", "adva": f"{adv_a.mean():.3f}",
    "grin_pv_w": f"{pw.g.mean():.3f}", "saits_pv_w": f"{pw.s.mean():.3f}",
    "grin_pv_a": f"{pa.g.mean():.3f}", "saits_pv_a": f"{pa.s.mean():.3f}",
    "nvic_w": str(len(adv_w)), "nvic_a": str(len(adv_a)),
    "abl_within": str(abl_within), "abl_total": str(abl_total),
    "nconn_w": str(int(conn.WVHT_MAE.notna().sum())), "nconn_a": str(int(conn.APD_MAE.notna().sum())),
    "gwn12_w": f"{fmae('graphwavenet', 'WVHT', 12)[1]:+.3f}", "gwn24_w": f"{fmae('graphwavenet', 'WVHT', 24)[1]:+.3f}",
    "gwn12_a": f"{fmae('graphwavenet', 'APD', 12)[1]:+.3f}", "gwn24_a": f"{fmae('graphwavenet', 'APD', 24)[1]:+.3f}",
    "ar_lo": f"{ar_lo:.3f}", "ar_hi": f"{ar_hi:.3f}",
    "en_gwn12": f"{emae('graphwavenet', 12)[1]:+.3f}", "en_gwn24": f"{emae('graphwavenet', 24)[1]:+.3f}",
    "en_ar12": f"{emae('ar24', 12)[1]:+.3f}", "en_ar24": f"{emae('ar24', 24)[1]:+.3f}",
    # numbers NOT in any committed CSV -> VERIFY markers
    "v_plausible": verify("Slide 5: '1,176 wave-plausible stations' -- not in a committed CSV "
                          "(coverage_audit.csv has 872 audited / 283 with WVHT data; use those, or supply source)"),
    "v_skew": verify("Slide 9: WVHT log1p skew (~1.71) -- not in a committed CSV (tensor .npz is uncommitted)"),
    "v_anem": verify("Slide 9: '65 of 140 stations lack an anemometer' -- no wind columns in any committed CSV"),
    "v_discard": verify("Slide 33: discarded 100-epoch run scored ~0.331 -- not in a committed CSV "
                        "(canonical GRIN outage WVHT is the committed 0.346)"),
}

TABLES = {
    "TAB_IMP": TAB_IMP, "TAB_CONN": TAB_CONN, "TAB_ABL": TAB_ABL, "TAB_ABL_FULL": TAB_ABL_FULL,
    "TAB_FC": TAB_FC, "TAB_ENERGY": TAB_ENERGY, "TAB_PARALLEL": TAB_PARALLEL,
    "TAB_BASE_FULL": TAB_BASE_FULL, "TAB_BRITS_FULL": TAB_BRITS_FULL, "TAB_PV": TAB_PV,
}

# --------------------------------------------------------------------------- #
# Emit
# --------------------------------------------------------------------------- #
template = (R / "_slides_body.tex").read_text()


def fill(tex: str) -> str:
    for k, v in TABLES.items():
        tex = tex.replace(f"@{k}@", v)
    for k, v in P.items():
        tex = tex.replace(f"@{k}@", v)
    return tex


body = fill(template)
leftover = sorted({w for w in body.split() if w.startswith("@") and w.endswith("@") and len(w) > 2})
assert not leftover, f"unfilled placeholders: {leftover}"

clean = body.replace("%__NOTES__%", "")
notes = body.replace("%__NOTES__%", r"\setbeameroption{show notes}")
(R / "slides.tex").write_text(clean)
(R / "slides_notes.tex").write_text(notes)

print(f"wrote slides.tex and slides_notes.tex")
print(f"cascade {n_audit} audited / {n_wave} wave-plausible / {n_sel} selected (0.70 thr -> {thr140})")
print(f"per-victim advantage WVHT {adv_w.mean():+.3f} (n={len(adv_w)})  APD {adv_a.mean():+.3f} (n={len(adv_a)})")
print(f"ablation within-noise {abl_within}/{abl_total}   AR skill range {ar_lo:.3f}-{ar_hi:.3f}")
if VERIFY:
    print(f"\n[VERIFY] markers ({len(VERIFY)}):")
    for v in VERIFY:
        print(f"  - {v}")
