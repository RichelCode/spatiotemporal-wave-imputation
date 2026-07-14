"""Deep spatiotemporal forecaster (GraphWaveNet) for long horizons (12h, 24h).

Physical motivation: at long lead times, wave systems PROPAGATE between stations,
so an upstream buoy sees a storm before it reaches the target. A graph forecaster
can exploit that neighbour signal; a per-station AR model structurally cannot.

Model: tsl GraphWaveNetModel consuming the committed basin-aware adjacency
(adjacency_knn_basin) as a FIXED edge_index + edge_weight (learned_adjacency off).
Input: a window of W_IN=24 h of recent history across ALL 140 stations x 3
features. Output: a 24-step horizon; we score steps h=12 and h=24 (one trained
model serves both horizons). Only WVHT and APD are scored.

Leakage discipline (EVALUATION.md sec. 2; SAME harness as the classical baselines):
  * TRAIN on the GRIN-completed tensor (2021-2023). Bidirectional imputation in
    the training data is permitted.
  * VALIDATE on 2024 (completed) for early stopping.
  * TEST on 2025 scored through the committed ``valid_origin_mask`` (input window
    fully observed for the target, target y_{t+h} genuinely observed) -- the
    IDENTICAL origins used for persistence / AR(24).
  * At TEST the model input is the observed window with any gaps filled by
    CAUSAL forward-fill (past-only), NEVER the bidirectional GRIN-completed
    values -- so no future-informed value enters the input. (This differs from
    the completed inputs used in training; the mismatch is documented, not a
    leak.)

Metrics: MAE/RMSE per target in physical units + persistence-normalised skill,
directly comparable to reports/forecast_horizon_sweep.csv.

Run:  ``python -m src.models.forecast_deep``  (smoke test)
Importable; nothing runs on import.
"""

from __future__ import annotations

import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tsl.nn.models.stgn import GraphWaveNetModel

from ..data.download import load_config
from ..evaluation.forecast_eval import (
    INPUT_WINDOW, MODEL_NPZ, RAW_TENSOR_NPZ, mae, rmse, skill_vs_persistence,
    valid_origin_mask,
)
from ..features.preprocess import inverse_transform, load_meta
from .deep_common import select_device
from .grin_imputer import GRIN_CHECKPOINT, GRINImputer, load_graph_edges

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COMPLETED_NORM_NPZ = PROCESSED_DIR / "grin_completed_norm.npz"  # gitignored
GWN_CHECKPOINT = PROCESSED_DIR / "gwn_forecast_checkpoint.pt"   # gitignored (*.pt)
FORECAST_DEEP_CSV = PROJECT_ROOT / "reports" / "forecast_deep_results.csv"

W_IN = INPUT_WINDOW          # input history window (24h) = harness leakage window
MODEL_HORIZON = 24           # trained horizon; score steps 12 and 24
EVAL_HORIZONS = [12, 24]
TARGET_NAMES = ["WVHT", "APD"]


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------
def norm_completed(device=None, force: bool = False) -> np.ndarray:
    """Normalised GRIN-completed tensor (bidirectional; for TRAINING). Cached."""
    if COMPLETED_NORM_NPZ.exists() and not force:
        return np.load(COMPLETED_NORM_NPZ)["completed_norm"]
    config = load_config()
    dev = device if device is not None else select_device(config["deep"])
    norm = np.load(MODEL_NPZ)["tensor"]  # normalised, NaN at missing
    model = GRINImputer(config, device=dev, adjacency="adjacency_knn_basin").load_checkpoint(GRIN_CHECKPOINT)
    completed = model.impute(norm).astype(np.float32)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COMPLETED_NORM_NPZ, completed_norm=completed)
    return completed


def _ffill_axis0(a: np.ndarray) -> np.ndarray:
    """Causal forward-fill of NaNs down time for a [T, S] array (past-only)."""
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(a.shape[0])[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return a[idx, np.arange(a.shape[1])[None, :]]


def causal_fill_norm() -> np.ndarray:
    """Leakage-safe TEST input: normalised observed, gaps causally forward-filled."""
    norm = np.load(MODEL_NPZ)["tensor"].astype(np.float32)  # normalised, NaN at missing
    out = norm.copy()
    for f in range(out.shape[2]):
        out[:, :, f] = _ffill_axis0(out[:, :, f])
    out[np.isnan(out)] = 0.0  # leading gaps (no past) -> 0 == normalised mean
    return out


# ---------------------------------------------------------------------------
# forecaster
# ---------------------------------------------------------------------------
class GraphForecaster:
    """GraphWaveNet direct multi-horizon forecaster on the basin-aware graph."""
    name = "graphwavenet"

    def __init__(self, config: dict, device: torch.device | None = None,
                 adjacency: str = "adjacency_knn_basin", horizon: int = MODEL_HORIZON,
                 win: int = W_IN, hidden_size: int = 32, ff_size: int = 128,
                 n_layers: int = 4, epochs: int | None = None, patience: int | None = -1,
                 batch_size: int = 32, seed: int | None = None):
        deep = config["deep"]
        self.win, self.horizon = int(win), int(horizon)
        self.lr = float(deep["learning_rate"])
        self.epochs = int(epochs if epochs is not None else deep["max_epochs"])
        self.patience = deep["patience"] if patience == -1 else patience
        self.device = device if device is not None else select_device(deep)
        self.batch_size = int(batch_size)
        self.seed = int(seed if seed is not None else deep.get("seed", config["project"]["seed"]))
        self.arch = dict(hidden_size=hidden_size, ff_size=ff_size, n_layers=n_layers)
        self.edge_index, self.edge_weight, self.n_nodes = load_graph_edges(adjacency)
        self.target_idx = [load_meta()["feature_names"].index(t) for t in TARGET_NAMES]
        self.model = None

    def _build(self, n_features: int) -> GraphWaveNetModel:
        return GraphWaveNetModel(
            input_size=n_features, output_size=n_features, horizon=self.horizon,
            n_nodes=self.n_nodes, learned_adjacency=False, **self.arch)

    def _windows(self, tensor: torch.Tensor, origins: torch.Tensor):
        """Gather input [B,win,N,F] and target [B,horizon,N,F] for a batch of origins."""
        in_idx = origins[:, None] + torch.arange(-self.win + 1, 1, device=origins.device)
        out_idx = origins[:, None] + torch.arange(1, self.horizon + 1, device=origins.device)
        return tensor[in_idx], tensor[out_idx]

    def fit(self, completed_norm: np.ndarray, train_end: int, val_start: int, val_end: int) -> "GraphForecaster":
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(self.seed)
        dev = self.device
        F = completed_norm.shape[2]
        self.model = self._build(F).to(dev)
        ei, ew = self.edge_index.to(dev), self.edge_weight.to(dev)
        data = torch.tensor(completed_norm, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        tgt = self.target_idx

        train_o = torch.arange(self.win - 1, train_end - self.horizon, device=dev)
        val_o = torch.arange(val_start, val_end - self.horizon, device=dev)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            perm = train_o[torch.randperm(len(train_o), device=dev)]
            running = 0.0
            for s in range(0, len(perm), self.batch_size):
                o = perm[s:s + self.batch_size]
                x, y = self._windows(data, o)
                pred = self.model(x, ei, ew)
                loss = torch.abs(pred[..., tgt] - y[..., tgt]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
            vloss = self._val_loss(data, val_o, ei, ew, tgt)
            logger.info("epoch %d - train %.4f - val %.4f", epoch, running / max(1, len(range(0, len(perm), self.batch_size))), vloss)
            if vloss < best:
                best, best_state, bad = vloss, copy.deepcopy(self.model.state_dict()), 0
            elif self.patience is not None:
                bad += 1
                if bad >= self.patience:
                    logger.info("early stop at epoch %d (best %.4f)", epoch, best)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _val_loss(self, data, val_o, ei, ew, tgt) -> float:
        self.model.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(val_o), self.batch_size):
                o = val_o[s:s + self.batch_size]
                x, y = self._windows(data, o)
                pred = self.model(x, ei, ew)
                total += torch.abs(pred[..., tgt] - y[..., tgt]).mean().item()
                nb += 1
        return total / max(1, nb)

    def save_checkpoint(self, path=GWN_CHECKPOINT) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info("saved GWN checkpoint -> %s", path)

    def load_checkpoint(self, path=GWN_CHECKPOINT, n_features: int = 3) -> "GraphForecaster":
        self.model = self._build(n_features).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def predict_origins(self, input_source: np.ndarray, origin_times: np.ndarray) -> np.ndarray:
        """Predict [len(origins), horizon, N, F] (normalised) for the given origin times."""
        dev = self.device
        ei, ew = self.edge_index.to(dev), self.edge_weight.to(dev)
        data = torch.tensor(input_source, dtype=torch.float32, device=dev)
        origins = torch.tensor(origin_times, dtype=torch.long, device=dev)
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(origins), self.batch_size):
                o = origins[s:s + self.batch_size]
                in_idx = o[:, None] + torch.arange(-self.win + 1, 1, device=dev)
                pred = self.model(data[in_idx], ei, ew)  # [B, horizon, N, F]
                out.append(pred.cpu().numpy())
        return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# scoring (leakage-safe, identical origins to the classical baselines)
# ---------------------------------------------------------------------------
def score_forecaster(model: GraphForecaster, meta: dict, horizons=EVAL_HORIZONS,
                     sample: int | None = None, seed: int = 0) -> list[dict]:
    """Score on the committed valid origins (identical to persistence/AR).

    Predicts once on the UNION of required origin times (leakage-safe causal
    inputs), then extracts each (target, horizon) subset. Returns rows with the
    same schema as the horizon sweep (method, target, horizon, n_origins, ...).
    """
    raw = np.load(RAW_TENSOR_NPZ)["tensor"].astype(np.float64)
    obs = ~np.isnan(raw)
    causal = causal_fill_norm()
    feats = meta["feature_names"]
    test_start = int(meta["split"]["test"]["start"])
    test_end = int(meta["split"]["test"]["end"])
    rng = np.random.default_rng(seed)

    combos, all_times = {}, set()
    for tname in TARGET_NAMES:
        f = feats.index(tname)
        for h in horizons:
            ts, ss = np.where(valid_origin_mask(obs[:, :, f], test_start, test_end, W_IN, h))
            if sample is not None and len(ts) > sample:
                sel = rng.choice(len(ts), sample, replace=False)
                ts, ss = ts[sel], ss[sel]
            combos[(tname, h)] = (f, ts, ss)
            all_times.update(ts.tolist())

    uniq_t = np.array(sorted(all_times))
    preds = model.predict_origins(causal, uniq_t)  # [U, horizon, N, F], predicted ONCE

    results = []
    for (tname, h), (f, ts, ss) in combos.items():
        pos = np.searchsorted(uniq_t, ts)                     # uniq_t sorted; ts subset of it
        pred_phys = inverse_transform(preds[pos, h - 1, ss, f], f, meta)
        truth = raw[ts + h, ss, f]
        pers = raw[ts, ss, f]
        m = mae(pred_phys, truth)
        results.append({
            "method": model.name, "target": tname, "horizon": h, "n_origins": len(ts),
            "MAE": m, "RMSE": rmse(pred_phys, truth),
            "skill_vs_persistence": skill_vs_persistence(m, mae(pers, truth)),
            "pred_min": float(pred_phys.min()), "pred_max": float(pred_phys.max()),
        })
    return results


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _bench_one_epoch(config, completed, bounds, dev_name: str) -> str:
    try:
        dev = torch.device(dev_name)
        m = GraphForecaster(config, device=dev, horizon=MODEL_HORIZON, epochs=1, patience=None)
        t0 = time.time()
        m.fit(completed, *bounds)
        return f"{dev_name}: 1-epoch fit {time.time() - t0:.1f}s  [ok]"
    except Exception as exc:
        return f"{dev_name}: FAILED - {type(exc).__name__}: {str(exc)[:140]}"


def run_smoke() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    train_end = int(meta["split"]["train"]["end"])
    val_start = int(meta["split"]["val"]["start"])
    val_end = int(meta["split"]["val"]["end"])
    bounds = (train_end, val_start, val_end)

    ei, ew, n_nodes = load_graph_edges("adjacency_knn_basin")
    print(f"graph: {n_nodes} nodes, {ei.shape[1]} edges | input window {W_IN}h | model horizon {MODEL_HORIZON}h")
    completed = norm_completed()  # normalised GRIN-completed (cached)
    print(f"completed-norm tensor {completed.shape} (train origins ~{train_end - W_IN - MODEL_HORIZON:,})")

    print("\n--- per-epoch device benchmark (1 epoch each, full train set) ---")
    print("  " + _bench_one_epoch(config, completed, bounds, "cpu"))
    print("  " + _bench_one_epoch(config, completed, bounds, "mps"))

    print("\n--- smoke train (3 epochs, CPU) + rough score at h=12,24 ---")
    model = GraphForecaster(config, device=torch.device("cpu"), horizon=MODEL_HORIZON,
                            epochs=3, patience=None)
    t0 = time.time()
    model.fit(completed, *bounds)
    print(f"3-epoch train: {time.time() - t0:.1f}s | params: "
          f"{sum(p.numel() for p in model.model.parameters()):,}")

    t1 = time.time()
    results = score_forecaster(model, meta, EVAL_HORIZONS, sample=3000)
    print(f"score (sampled 3000/origin-set): {time.time() - t1:.1f}s\n")
    print(f"{'target':6s}{'h':>4s}{'n':>7s}{'MAE':>9s}{'RMSE':>9s}{'skill':>9s}{'phys range':>18s}")
    for r in results:
        print(f"{r['target']:6s}{r['horizon']:>4d}{r['n_origins']:>7d}{r['MAE']:>9.4f}{r['RMSE']:>9.4f}"
              f"{r['skill_vs_persistence']:>+9.3f}   [{r['pred_min']:.2f}, {r['pred_max']:.2f}]")
    print("\n(3-epoch UNDERTRAINED; compare skill sign/scale, not final quality.)")
    return 0


# ---------------------------------------------------------------------------
# full evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation() -> int:
    """Fit GraphWaveNet on CPU (seeded, capped, checkpointed); score full origin set."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    train_end = int(meta["split"]["train"]["end"])
    val_start = int(meta["split"]["val"]["start"])
    val_end = int(meta["split"]["val"]["end"])
    seed = int(config["deep"].get("seed", config["project"]["seed"]))

    # GraphWaveNet cannot run on MPS (channels_last op unsupported); CPU is forced.
    device = torch.device("cpu")
    print(f"Device: {device} (CPU forced; GWN unsupported on MPS) | seed={seed} | "
          f"max_epochs=30 patience={config['deep']['patience']} | horizon={MODEL_HORIZON} win={W_IN}")

    completed = norm_completed()
    model = GraphForecaster(config, device=device, adjacency="adjacency_knn_basin",
                            horizon=MODEL_HORIZON, epochs=30, seed=seed)

    t0 = time.time()
    model.fit(completed, train_end, val_start, val_end)
    fit_time = time.time() - t0
    model.save_checkpoint(GWN_CHECKPOINT)
    print(f"fit time: {fit_time / 60:.1f} min | checkpoint saved")

    t1 = time.time()
    results = score_forecaster(model, meta, EVAL_HORIZONS, sample=None)
    eval_time = time.time() - t1

    df = pd.DataFrame(results)[["method", "target", "horizon", "n_origins",
                                "MAE", "RMSE", "skill_vs_persistence"]]
    FORECAST_DEEP_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FORECAST_DEEP_CSV, index=False)

    print(f"\nWall-clock: fit {fit_time / 60:.1f} min + eval {eval_time / 60:.1f} min")
    print(f"Wrote {FORECAST_DEEP_CSV}")
    for r in results:
        print(f"  {r['target']} h={r['horizon']:>2d}: MAE={r['MAE']:.4f} skill={r['skill_vs_persistence']:+.3f} "
              f"(n={r['n_origins']:,})")
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Deep spatiotemporal forecaster.")
    parser.add_argument("--full", action="store_true", help="Full run (fit, checkpoint, score full set).")
    args = parser.parse_args()
    return run_full_evaluation() if args.full else run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
