"""SAITS imputer — PyPOTS SAITS wrapped in the DeepImputer interface.

SAITS (Self-Attention-based Imputation for Time Series) is a TEMPORAL-only deep
model: each station's window is imputed from its own history and its other
features, with NO spatial/graph information. It is therefore the strong
temporal-deep reference point — the number a graph-augmented model must beat on
the station-outage task (where temporal-only cannot recover a fully blacked-out
station).

Reshaping: PyPOTS expects [n_samples, n_steps, n_features]. We treat each
station's window as one independent multivariate sample of 3 features
(WVHT, DPD, APD): [N, W, S, F] -> [N*S, W, F]. DPD stays as an input feature to
help imputation; only WVHT and APD are scored (the harness handles that).

Missingness: NaN marks missing cells, which is exactly PyPOTS' expected format.
Validation for early stopping uses an MCAR holdout (pygrinder) so patience has a
real signal.

Run:  ``python -m src.models.saits_imputer``  (smoke test: few epochs, one mask)
Importable; nothing runs on import.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pygrinder import mcar
from pypots.imputation import SAITS
from pypots.optim import Adam

from ..data.download import load_config
from ..evaluation.masking import load_masks, make_model_input, score
from ..features.preprocess import inverse_transform, load_meta
from .baselines import _config_key, load_truth
from .deep_common import (
    DeepImputer, build_training_windows, evaluate_deep_model, reassemble,
    select_device, window_series,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AXES_JSON = PROJECT_ROOT / "data" / "processed" / "wave_tensor_axes.json"
SAITS_RESULTS_CSV = PROJECT_ROOT / "reports" / "saits_imputation_results.csv"
SAITS_PER_VICTIM_CSV = PROJECT_ROOT / "reports" / "saits_per_victim_outage.csv"

VAL_HOLDOUT_FRACTION = 0.10  # MCAR fraction masked in val for the early-stopping signal


def _windows_to_samples(values: np.ndarray) -> np.ndarray:
    """[N, W, S, F] -> [N*S, W, F] float32 (one sample per station-window)."""
    n, w, s, f = values.shape
    return np.transpose(values, (0, 2, 1, 3)).reshape(n * s, w, f).astype(np.float32)


def _samples_to_windows(samples: np.ndarray, n: int, w: int, s: int, f: int) -> np.ndarray:
    """[N*S, W, F] -> [N, W, S, F]."""
    return samples.reshape(n, s, w, f).transpose(0, 2, 1, 3)


class SAITSImputer(DeepImputer):
    """Temporal-only deep imputer (PyPOTS SAITS) conforming to DeepImputer."""
    name = "saits"

    def __init__(self, config: dict, device: torch.device | None = None,
                 epochs: int | None = None, patience: int | None = -1,
                 n_layers: int = 2, d_model: int = 64, n_heads: int = 4,
                 d_k: int = 16, d_v: int = 16, d_ffn: int = 128, dropout: float = 0.1):
        deep = config["deep"]
        self.W = int(deep["window_hours"])
        self.batch_size = int(deep["batch_size"])
        self.lr = float(deep["learning_rate"])
        self.epochs = int(epochs if epochs is not None else deep["max_epochs"])
        # patience=-1 sentinel -> use config default; patience=None -> disable early stopping.
        self.patience = deep["patience"] if patience == -1 else patience
        self.device = device if device is not None else select_device(deep)
        self.arch = dict(n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                         d_k=d_k, d_v=d_v, d_ffn=d_ffn, dropout=dropout)
        self.model = None
        self.used_device = None

    def _build_model(self, n_features: int, device: torch.device) -> SAITS:
        return SAITS(
            n_steps=self.W, n_features=n_features,
            batch_size=self.batch_size, epochs=self.epochs, patience=self.patience,
            optimizer=Adam(lr=self.lr), num_workers=0, device=device,
            saving_path=None, verbose=True, **self.arch,
        )

    def fit(self, train_windows, val_windows) -> "SAITSImputer":
        F = train_windows.values.shape[3]
        train_set = {"X": _windows_to_samples(train_windows.values)}
        val_ori = _windows_to_samples(val_windows.values)
        np.random.seed(0)  # reproducible val holdout
        val_set = {"X": mcar(val_ori, VAL_HOLDOUT_FRACTION), "X_ori": val_ori}

        try:
            self.model = self._build_model(F, self.device)
            self.model.fit(train_set, val_set)
            self.used_device = self.device
        except Exception as exc:  # e.g. an op unsupported on MPS
            if self.device.type == "cpu":
                raise
            logger.warning("SAITS on %s failed (%s); falling back to CPU.", self.device, exc)
            cpu = torch.device("cpu")
            self.model = self._build_model(F, cpu)
            self.model.fit(train_set, val_set)
            self.used_device = cpu
        return self

    def impute(self, full_masked_tensor: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SAITSImputer.impute called before fit")
        T, S, F = full_masked_tensor.shape
        observed = ~np.isnan(full_masked_tensor)
        win = window_series(full_masked_tensor, observed, 0, T, self.W)
        n = len(win)
        samples = _windows_to_samples(win.values)
        imputed = self.model.predict({"X": samples})["imputation"]
        win_out = _samples_to_windows(np.asarray(imputed), n, self.W, S, F)
        return reassemble(win_out, win.starts, self.W, T)


# ---------------------------------------------------------------------------
# full evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation() -> int:
    """Fit SAITS once on clean 2021-2024, evaluate all masks, save the results CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)
    W = int(config["deep"]["window_hours"])

    device = select_device(config["deep"])
    print(f"Requested device: {device}  | batch_size={config['deep']['batch_size']} "
          f"max_epochs={config['deep']['max_epochs']} patience={config['deep']['patience']}")

    train, val = build_training_windows(truth, observed, meta, W)
    model = SAITSImputer(config, device=device)  # config max_epochs + early stopping

    t0 = time.time()
    model.fit(train, val)
    fit_time = time.time() - t0
    print(f"training device: {model.used_device} | fit time: {fit_time / 60:.1f} min")

    t1 = time.time()
    agg = evaluate_deep_model(model, save_path=SAITS_RESULTS_CSV)
    eval_time = time.time() - t1

    total = fit_time + eval_time
    print(f"\nWall-clock: fit {fit_time / 60:.1f} min + eval {eval_time / 60:.1f} min "
          f"= {total / 60:.1f} min total")
    print(f"Wrote {SAITS_RESULTS_CSV}")
    return 0


# ---------------------------------------------------------------------------
# per-victim errors on station-outage masks (for connectivity analysis)
# ---------------------------------------------------------------------------
def run_outage_per_victim() -> int:
    """Retrain SAITS (seeded) and save per-victim errors for the outage masks.

    Cheap: fits once, imputes ONLY the 6 station-outage masks, and writes
    reports/saits_per_victim_outage.csv so the GRIN-vs-SAITS per-victim
    comparison is never lost.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)
    W = int(config["deep"]["window_hours"])
    device = select_device(config["deep"])
    seed = int(config["deep"].get("seed", config["project"]["seed"]))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)
    print(f"SAITS per-victim: device={device} seed={seed} max_epochs={config['deep']['max_epochs']}")

    train, val = build_training_windows(truth, observed, meta, W)
    model = SAITSImputer(config, device=device)
    t0 = time.time()
    model.fit(train, val)
    print(f"fit time: {(time.time() - t0) / 60:.1f} min")

    station_ids = list(json.load(open(AXES_JSON))["station_ids"])
    masks = [m for m in load_masks() if m.mechanism == "station_outage"]
    records = []
    for mask in masks:
        res = score(model.impute(make_model_input(truth, mask)), mask, meta, truth)
        cfg = _config_key(mask)
        for s_idx, entry in res["per_victim"].items():
            for tgt, mets in entry.items():
                for metric in ("MAE", "RMSE"):
                    records.append({
                        "station_id": station_ids[s_idx], "station_idx": int(s_idx),
                        "config": cfg, "seed": mask.seed, "target": tgt,
                        "metric": metric, "value": mets[metric], "n_cells": mets["n"],
                    })
    pd.DataFrame(records).to_csv(SAITS_PER_VICTIM_CSV, index=False)
    print(f"Wrote {SAITS_PER_VICTIM_CSV} ({len(records)} rows)")
    return 0


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def run_smoke() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)

    device = select_device(config["deep"])
    print(f"Requested device: {device}")

    train, val = build_training_windows(truth, observed, meta, int(config["deep"]["window_hours"]))
    print(f"train samples: {len(train) * truth.shape[1]:,}  "
          f"val samples: {len(val) * truth.shape[1]:,}  (station-windows of [48, 3])")

    SMOKE_EPOCHS = 3
    model = SAITSImputer(config, device=device, epochs=SMOKE_EPOCHS, patience=None)

    t0 = time.time()
    model.fit(train, val)
    fit_time = time.time() - t0
    print(f"\nActual training device: {model.used_device}"
          f"{'  (MPS OK)' if model.used_device.type == 'mps' else '  (fell back)'}")
    print(f"fit time ({SMOKE_EPOCHS} epochs): {fit_time:.1f}s  ->  ~{fit_time / SMOKE_EPOCHS:.1f}s/epoch")

    mask = {m.id: m for m in load_masks()}["station_outage_v14_durfull_seed0"]
    t1 = time.time()
    imputed = model.impute(make_model_input(truth, mask))
    impute_time = time.time() - t1
    res = score(imputed, mask, meta, truth)

    print(f"\nimpute time (1 mask): {impute_time:.1f}s")
    print(f"one-mask score on {mask.id}:")
    for tgt in ("WVHT", "APD"):
        m = res["overall"][tgt]
        print(f"  {tgt}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MRE={m['MRE']:.3f}")

    # Sanity: finite + physical range of the recovered WVHT at hidden cells.
    feats = meta["feature_names"]
    wf = feats.index("WVHT")
    ts, ss = np.where(mask.hidden[:, :, wf])
    wvht_pred = inverse_transform(imputed[ts, ss, wf], wf, meta)
    print(f"\nrecovered WVHT (m) at hidden cells: finite={np.isfinite(wvht_pred).all()}  "
          f"min={wvht_pred.min():.2f}  median={np.median(wvht_pred):.2f}  max={wvht_pred.max():.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SAITS imputer.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full evaluation (fit once, score all masks, save CSV).")
    parser.add_argument("--outage-per-victim", action="store_true",
                        help="Fit once and save per-victim errors for the station-outage masks.")
    args = parser.parse_args()
    if args.outage_per_victim:
        return run_outage_per_victim()
    return run_full_evaluation() if args.full else run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
