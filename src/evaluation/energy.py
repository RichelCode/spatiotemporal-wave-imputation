"""Wave-energy forecasting: convert forecasted WVHT/APD into wave power and score.

The applied payoff. Deep-water wave energy flux (power per unit crest length):

    P = rho*g^2/(64*pi) * Hm0^2 * Te   ->   P ~= 0.49 * H^2 * Te   (kW/m)

with H in metres and Te in seconds. NDBC reports APD (a mean period), not the
energy period Te, so we convert Te = alpha * APD with alpha from config
(``energy.te_apd_alpha``, ~1.12; JONSWAP Te ~= 1.12*Tz). The comparative results
(method rankings, persistence-normalised skill) are INVARIANT to alpha -- it
scales every method's energy identically -- so only absolute kW/m depends on it.

Nonlinearity: P ~ H^2 * Te, so errors in wave HEIGHT are squared while period
enters linearly; height accuracy matters more than period accuracy for energy.

We reuse the SAME leakage-safe forecasting harness. True energy uses the OBSERVED
WVHT and APD at t+h (ground truth); a station-hour is scored only where BOTH
targets are valid (window observed + both targets genuinely observed), so true
energy is real. Forecasted energy for each method (persistence, AR-24,
GraphWaveNet) plugs that method's forecasted WVHT and APD into the power formula.

Outputs: reports/energy_forecast_results.csv and reports/figures/fig11_energy_forecast.{png,pdf}.

Run:  ``python -m src.evaluation.energy``
Importable; nothing runs on import.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..data.download import load_config
from ..features.preprocess import inverse_transform, load_meta
from ..models.forecast_baselines import AR_ORDER, fit_ar_direct
from ..models.forecast_deep import (
    GWN_CHECKPOINT, MODEL_HORIZON, W_IN, GraphForecaster, causal_fill_norm,
)
from .forecast_eval import (
    RAW_TENSOR_NPZ, complete_tensor_with_grin, mae, rmse, skill_vs_persistence,
    valid_origin_mask,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_CSV = PROJECT_ROOT / "reports" / "energy_forecast_results.csv"
HORIZONS = [12, 24]
METHODS = ["persistence", "ar24", "graphwavenet"]


def wave_power(h_m: np.ndarray, apd_s: np.ndarray, coeff: float, alpha: float) -> np.ndarray:
    """Deep-water wave power (kW/m) from height (m) and APD (s): coeff * H^2 * (alpha*APD)."""
    return coeff * h_m ** 2 * (alpha * apd_s)


def _ar_forecast(completed, raw, ts, ss, f, train_end, h) -> np.ndarray:
    """Direct h-step AR(24) forecast (physical) for feature f at origins (ts, ss)."""
    coefs = {int(s): fit_ar_direct(completed[:train_end, s, f], AR_ORDER, h) for s in np.unique(ss)}
    lagmat = np.column_stack([raw[ts - k, ss, f] for k in range(AR_ORDER)])
    C = np.array([coefs[int(s)] if coefs[int(s)] is not None else np.zeros(AR_ORDER + 1) for s in ss])
    pred = C[:, 0] + np.einsum("nk,nk->n", lagmat, C[:, 1:])
    none = np.array([coefs[int(s)] is None for s in ss])
    pred[none] = raw[ts[none], ss[none], f]
    return pred


def run() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    coeff = float(config["energy"]["power_coeff_kw"])
    alpha = float(config["energy"]["te_apd_alpha"])
    feats = meta["feature_names"]
    wf, af = feats.index("WVHT"), feats.index("APD")
    test_start = int(meta["split"]["test"]["start"])
    test_end = int(meta["split"]["test"]["end"])
    train_end = int(meta["split"]["train"]["end"])

    raw = np.load(RAW_TENSOR_NPZ)["tensor"].astype(np.float64)
    obs = ~np.isnan(raw)
    completed = complete_tensor_with_grin().astype(np.float64)  # physical, for AR fits
    causal = causal_fill_norm()                                 # leakage-safe GWN input
    gwn = GraphForecaster(config, device=torch.device("cpu"), adjacency="adjacency_knn_basin",
                          horizon=MODEL_HORIZON).load_checkpoint(GWN_CHECKPOINT)

    print(f"energy: P = {coeff} * H^2 * ({alpha} * APD) kW/m  (alpha-invariant rankings/skill)")
    records = []
    for h in HORIZONS:
        common = valid_origin_mask(obs[:, :, wf], test_start, test_end, W_IN, h) \
            & valid_origin_mask(obs[:, :, af], test_start, test_end, W_IN, h)
        ts, ss = np.where(common)
        logger.info("h=%d: %d origins with BOTH WVHT and APD valid", h, len(ts))

        p_true = wave_power(raw[ts + h, ss, wf], raw[ts + h, ss, af], coeff, alpha)

        # per-method forecasted height and period
        forecasts = {"persistence": (raw[ts, ss, wf], raw[ts, ss, af]),
                     "ar24": (_ar_forecast(completed, raw, ts, ss, wf, train_end, h),
                              _ar_forecast(completed, raw, ts, ss, af, train_end, h))}
        uniq = np.unique(ts)
        preds = gwn.predict_origins(causal, uniq)
        pos = np.searchsorted(uniq, ts)
        forecasts["graphwavenet"] = (
            inverse_transform(preds[pos, h - 1, ss, wf], wf, meta),
            inverse_transform(preds[pos, h - 1, ss, af], af, meta))

        p_pers = wave_power(*forecasts["persistence"], coeff, alpha)
        mae_pers = mae(p_pers, p_true)
        for name in METHODS:
            hh, tt = forecasts[name]
            p_hat = wave_power(hh, tt, coeff, alpha)
            m = mae(p_hat, p_true)
            records.append({"method": name, "horizon": h, "n_origins": len(ts),
                            "MAE_kw": m, "RMSE_kw": rmse(p_hat, p_true),
                            "skill_vs_persistence": skill_vs_persistence(m, mae_pers)})

    df = pd.DataFrame(records)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    saved = _make_figure(df)

    print("\n" + "=" * 62)
    print("WAVE-ENERGY FORECAST (kW/m; leakage-safe; both targets observed)")
    print("=" * 62)
    print(f"{'method':16s}{'h':>4s}{'n':>10s}{'MAE_kW':>10s}{'RMSE_kW':>10s}{'skill':>9s}")
    for h in HORIZONS:
        for name in METHODS:
            r = df[(df.method == name) & (df.horizon == h)].iloc[0]
            print(f"{name:16s}{h:>4d}{int(r['n_origins']):>10,d}{r['MAE_kw']:>10.3f}"
                  f"{r['RMSE_kw']:>10.3f}{r['skill_vs_persistence']:>+9.3f}")
    print(f"\nWrote {RESULTS_CSV} and {', '.join(saved)}")
    return 0


def _make_figure(df: pd.DataFrame) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ..features.eda import FIG_DIR, OKABE_ITO, set_style
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = {"ar24": OKABE_ITO["orange"], "graphwavenet": OKABE_ITO["blue"]}
    width = 0.35
    x = np.arange(len(HORIZONS))
    for i, name in enumerate(["ar24", "graphwavenet"]):
        vals = [float(df[(df.method == name) & (df.horizon == h)]["skill_vs_persistence"].iloc[0])
                for h in HORIZONS]
        ax.bar(x + (i - 0.5) * width, vals, width, label={"ar24": "AR(24)", "graphwavenet": "GraphWaveNet"}[name],
               color=colors[name], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}" for h in HORIZONS])
    ax.set_ylabel("energy skill vs persistence")
    ax.set_title("Figure 11. Wave-energy forecast skill (vs persistence)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig11_energy_forecast.{ext}"
        fig.savefig(p)
        saved.append(p.name)
    plt.close(fig)
    return saved


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
