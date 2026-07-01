"""BRITS imputer — PyPOTS BRITS wrapped in the DeepImputer interface.

BRITS (Bidirectional Recurrent Imputation for Time Series) is an RNN-based,
TEMPORAL-only deep model — architecturally independent of SAITS (attention).
Having a second temporal-only reference means the central claim, "temporal-only
models fail on station-outage because they cannot reach neighbours," rests on
two different architectures rather than one.

This wrapper mirrors :mod:`src.models.saits_imputer` exactly: same non-
overlapping windowing, the same [N, W, S, F] -> [N*S, W, F] reshape (each
station-window is one independent 3-feature multivariate sample), the same
MPS-with-CPU-fallback, the same MCAR val holdout for early stopping, and the
same float32 cast at the torch boundary. The reshape helpers are reused from the
SAITS wrapper to stay DRY.

Run:  ``python -m src.models.brits_imputer``          (smoke test)
      ``python -m src.models.brits_imputer --full``   (fit once, score all masks)
Importable; nothing runs on import.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from pygrinder import mcar
from pypots.imputation import BRITS
from pypots.optim import Adam

from ..data.download import load_config
from ..evaluation.masking import load_masks, make_model_input, score
from ..features.preprocess import inverse_transform, load_meta
from .baselines import load_truth
from .deep_common import (
    DeepImputer, build_training_windows, evaluate_deep_model, reassemble,
    select_device, window_series,
)
from .saits_imputer import VAL_HOLDOUT_FRACTION, _samples_to_windows, _windows_to_samples

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRITS_RESULTS_CSV = PROJECT_ROOT / "reports" / "brits_imputation_results.csv"


class BRITSImputer(DeepImputer):
    """Temporal-only RNN deep imputer (PyPOTS BRITS) conforming to DeepImputer."""
    name = "brits"

    def __init__(self, config: dict, device: torch.device | None = None,
                 epochs: int | None = None, patience: int | None = -1,
                 rnn_hidden_size: int = 64):
        deep = config["deep"]
        self.W = int(deep["window_hours"])
        self.batch_size = int(deep["batch_size"])
        self.lr = float(deep["learning_rate"])
        self.epochs = int(epochs if epochs is not None else deep["max_epochs"])
        # patience=-1 sentinel -> use config default; patience=None -> disable early stopping.
        self.patience = deep["patience"] if patience == -1 else patience
        self.device = device if device is not None else select_device(deep)
        self.rnn_hidden_size = int(rnn_hidden_size)
        self.model = None
        self.used_device = None

    def _build_model(self, n_features: int, device: torch.device) -> BRITS:
        return BRITS(
            n_steps=self.W, n_features=n_features, rnn_hidden_size=self.rnn_hidden_size,
            batch_size=self.batch_size, epochs=self.epochs, patience=self.patience,
            optimizer=Adam(lr=self.lr), num_workers=0, device=device,
            saving_path=None, verbose=True,
        )

    def fit(self, train_windows, val_windows) -> "BRITSImputer":
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
            logger.warning("BRITS on %s failed (%s); falling back to CPU.", self.device, exc)
            cpu = torch.device("cpu")
            self.model = self._build_model(F, cpu)
            self.model.fit(train_set, val_set)
            self.used_device = cpu
        return self

    def impute(self, full_masked_tensor: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("BRITSImputer.impute called before fit")
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
    """Fit BRITS once on clean 2021-2024, evaluate all masks, save the results CSV."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)
    W = int(config["deep"]["window_hours"])

    # BRITS is a bidirectional RNN; MPS RNN kernels are ~3x slower than CPU on
    # this machine (benchmarked), so the full run uses CPU regardless of the
    # config's auto device.
    device = torch.device("cpu")
    print(f"Device: {device} (CPU: RNN runs ~3x faster than MPS here) | "
          f"batch_size={config['deep']['batch_size']} "
          f"max_epochs={config['deep']['max_epochs']} patience={config['deep']['patience']}")

    train, val = build_training_windows(truth, observed, meta, W)
    model = BRITSImputer(config, device=device)

    t0 = time.time()
    model.fit(train, val)
    fit_time = time.time() - t0
    print(f"training device: {model.used_device} | fit time: {fit_time / 60:.1f} min")

    t1 = time.time()
    agg = evaluate_deep_model(model, save_path=BRITS_RESULTS_CSV)
    eval_time = time.time() - t1

    total = fit_time + eval_time
    print(f"\nWall-clock: fit {fit_time / 60:.1f} min + eval {eval_time / 60:.1f} min "
          f"= {total / 60:.1f} min total")
    print(f"Wrote {BRITS_RESULTS_CSV}")
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
    model = BRITSImputer(config, device=device, epochs=SMOKE_EPOCHS, patience=None)

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

    feats = meta["feature_names"]
    wf = feats.index("WVHT")
    ts, ss = np.where(mask.hidden[:, :, wf])
    wvht_pred = inverse_transform(imputed[ts, ss, wf], wf, meta)
    print(f"\nrecovered WVHT (m) at hidden cells: finite={np.isfinite(wvht_pred).all()}  "
          f"min={wvht_pred.min():.2f}  median={np.median(wvht_pred):.2f}  max={wvht_pred.max():.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BRITS imputer.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full evaluation (fit once, score all masks, save CSV).")
    args = parser.parse_args()
    return run_full_evaluation() if args.full else run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
