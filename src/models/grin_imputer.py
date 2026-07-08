"""GRIN imputer — tsl's GRINModel wrapped in the DeepImputer interface.

GRIN (Graph Recurrent Imputation Network) is the spatiotemporal GRAPH model: it
combines bidirectional recurrence (temporal) with message passing over the buoy
graph (spatial), so a blacked-out station is imputed from its neighbours. It is
the first model that should be strong in BOTH regimes — beating spatial IDW on
station-outage AND the temporal deep models (SAITS/BRITS) on MCAR/block.

Three pieces of wiring beyond the PyPOTS wrappers:

1. Adjacency -> edges. The basin-aware weighted adjacency (adjacency_knn_basin,
   140x140) is converted to GRIN's edge_index [2, E] (long) + edge_weight [E]
   (the Gaussian weights). Station order is asserted to match the tensor axis.
   The graph is FIXED across all training and imputation.
2. Whole-network windows. Unlike SAITS (per-station samples), GRIN needs
   x = [batch, time, nodes, features] = [B, W, 140, 3] so stations message-pass
   to neighbours. deep_common's [N, W, S, F] windows are exactly this, batched
   over N. embedding_size must be a positive int (None crashes GRINModel).
3. Masked-reconstruction training (Lightning-free). Each step WHITENS a random
   fraction of observed cells (hides them from the input) and reconstructs the
   observed values; the loss on observed cells forces the model to predict the
   whitened cells from spatiotemporal context, i.e. to impute. Early stopping on
   a fixed val whitening.

Note on device: torch_scatter / torch_sparse (used by the graph convs) are
CPU/CUDA C++ extensions with no MPS kernels, so GRIN typically runs on CPU. The
smoke test benchmarks both.

Run:  ``python -m src.models.grin_imputer``  (smoke test)
Importable; nothing runs on import.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tsl.nn.models.stgn.grin_model import GRINModel

from ..data.download import load_config
from ..evaluation.masking import TARGET_FEATURES, load_masks, make_model_input, score
from ..features.preprocess import inverse_transform, load_meta
from .baselines import _config_key, load_truth
from .deep_common import (
    DeepImputer, build_training_windows, reassemble, select_device, window_series,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
GRAPH_NPZ = PROCESSED_DIR / "wave_graph.npz"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
GRIN_RESULTS_CSV = PROJECT_ROOT / "reports" / "grin_imputation_results.csv"
GRIN_CHECKPOINT = PROCESSED_DIR / "grin_checkpoint.pt"          # gitignored (*.pt)
GRIN_OUTAGE_NPZ = PROCESSED_DIR / "grin_outage_imputations.npz"  # gitignored artifact
GRIN_PER_VICTIM_CSV = PROJECT_ROOT / "reports" / "grin_per_victim_outage.csv"  # analysis data (commits)

# GRIN batches WHOLE-NETWORK windows [B, W, 140, 3], so the batch is far heavier
# than SAITS's per-station samples; the config's batch_size=256 is inappropriate
# here. Use a small graph-window batch.
GRIN_BATCH_SIZE = 16
WHITEN_PROB = 0.25  # fraction of observed cells hidden from the input each step


def load_graph_edges(adjacency_key: str = "adjacency_knn_basin"):
    """Load a weighted adjacency as GRIN edge_index [2,E] (long) + edge_weight [E]."""
    z = np.load(GRAPH_NPZ)
    graph_order = [str(s) for s in z["station_order"]]
    axes_order = [str(s) for s in json.load(open(AXES_JSON))["station_ids"]]
    if graph_order != axes_order:
        raise RuntimeError("graph station order does not match tensor station order")
    adjacency = z[adjacency_key].astype(np.float32)
    src, dst = np.nonzero(adjacency)                      # only nonzero -> edges
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    edge_weight = torch.tensor(adjacency[src, dst], dtype=torch.float32)
    return edge_index, edge_weight, adjacency.shape[0]


def _masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean absolute error over the cells where ``mask`` is True."""
    return torch.abs(pred - target)[mask].mean()


def _prep(windows) -> tuple[torch.Tensor, torch.Tensor]:
    """Windows -> (x float32 with 0 at missing, observed-mask float32)."""
    vals = windows.values
    observed = (~np.isnan(vals)).astype(np.float32)
    x = np.nan_to_num(vals, nan=0.0).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(observed)


class GRINImputer(DeepImputer):
    """Spatiotemporal graph imputer (tsl GRIN) conforming to DeepImputer."""
    name = "grin"

    def __init__(self, config: dict, device: torch.device | None = None,
                 adjacency: str = "adjacency_knn_basin",
                 hidden_size: int = 32, ff_size: int = 64, embedding_size: int = 8,
                 n_layers: int = 1, kernel_size: int = 2,
                 epochs: int | None = None, patience: int | None = -1,
                 batch_size: int = GRIN_BATCH_SIZE, whiten_prob: float = WHITEN_PROB,
                 seed: int | None = None):
        deep = config["deep"]
        self.W = int(deep["window_hours"])
        self.lr = float(deep["learning_rate"])
        self.epochs = int(epochs if epochs is not None else deep["max_epochs"])
        self.patience = deep["patience"] if patience == -1 else patience
        self.device = device if device is not None else select_device(deep)
        self.seed = int(seed if seed is not None else deep.get("seed", config["project"]["seed"]))
        self.batch_size = int(batch_size)
        self.whiten_prob = float(whiten_prob)
        self.arch = dict(hidden_size=hidden_size, ff_size=ff_size,
                         embedding_size=embedding_size, n_layers=n_layers,
                         kernel_size=kernel_size)
        self.edge_index, self.edge_weight, self.n_nodes = load_graph_edges(adjacency)
        self.model = None

    def _build_model(self, n_features: int) -> GRINModel:
        return GRINModel(input_size=n_features, n_nodes=self.n_nodes, **self.arch)

    def _forward(self, x_in: torch.Tensor, mask: torch.Tensor, ei, ew) -> torch.Tensor:
        out = self.model(x_in, ei, edge_weight=ew, mask=mask)
        return out[0] if isinstance(out, (tuple, list)) else out

    def _seed(self) -> None:
        """Fix torch/numpy RNGs for reproducibility (MPS is best-effort)."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(self.seed)
        logger.info("seeded torch/numpy with seed=%d", self.seed)

    def save_checkpoint(self, path=GRIN_CHECKPOINT) -> None:
        """Save the trained model state_dict (reproducible from seed+script)."""
        torch.save(self.model.state_dict(), path)
        logger.info("saved GRIN checkpoint -> %s", path)

    def load_checkpoint(self, path=GRIN_CHECKPOINT, n_features: int = 3) -> "GRINImputer":
        """Rebuild the model and load a saved state_dict (skips retraining)."""
        self.model = self._build_model(n_features).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def fit(self, train_windows, val_windows) -> "GRINImputer":
        self._seed()
        dev = self.device
        F = train_windows.values.shape[3]
        self.model = self._build_model(F).to(dev)
        ei, ew = self.edge_index.to(dev), self.edge_weight.to(dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        Xtr, Mtr = _prep(train_windows)
        Xva, Mva = _prep(val_windows)
        # Fixed val whitening -> a deterministic imputation-quality signal.
        gen = torch.Generator().manual_seed(0)
        val_whiten = (torch.rand(Mva.shape, generator=gen) < self.whiten_prob) & (Mva > 0.5)

        best, best_state, bad = float("inf"), None, 0
        n_train = Xtr.shape[0]
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            perm = torch.randperm(n_train)
            running, n_batches = 0.0, 0
            for s in range(0, n_train, self.batch_size):
                idx = perm[s:s + self.batch_size]
                x, m = Xtr[idx].to(dev), Mtr[idx].to(dev)
                obs = m > 0.5
                whiten = (torch.rand_like(m) < self.whiten_prob) & obs
                input_mask = obs & ~whiten
                pred = self._forward(x * input_mask.float(), input_mask.float(), ei, ew)
                loss = _masked_mae(pred, x, obs)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
                n_batches += 1

            val_loss = self._val_loss(Xva, Mva, val_whiten, ei, ew, dev)
            logger.info("epoch %d - train %.4f - val %.4f", epoch, running / max(n_batches, 1), val_loss)
            if val_loss < best:
                best, best_state, bad = val_loss, copy.deepcopy(self.model.state_dict()), 0
            elif self.patience is not None:
                bad += 1
                if bad >= self.patience:
                    logger.info("early stopping at epoch %d (best %.4f)", epoch, best)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _val_loss(self, Xva, Mva, val_whiten, ei, ew, dev) -> float:
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for s in range(0, Xva.shape[0], self.batch_size):
                sl = slice(s, s + self.batch_size)
                x, m = Xva[sl].to(dev), Mva[sl].to(dev)
                whit = val_whiten[sl].to(dev)
                input_mask = (m > 0.5) & ~whit
                pred = self._forward(x * input_mask.float(), input_mask.float(), ei, ew)
                d = torch.abs(pred - x)[whit]
                total += d.sum().item()
                count += whit.sum().item()
        return total / max(count, 1)

    def impute(self, full_masked_tensor: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GRINImputer.impute called before fit")
        dev = self.device
        T = full_masked_tensor.shape[0]
        observed = ~np.isnan(full_masked_tensor)
        win = window_series(full_masked_tensor, observed, 0, T, self.W)
        X, M = _prep(win)
        ei, ew = self.edge_index.to(dev), self.edge_weight.to(dev)

        self.model.eval()
        preds = []
        with torch.no_grad():
            for s in range(0, X.shape[0], self.batch_size):
                sl = slice(s, s + self.batch_size)
                x, m = X[sl].to(dev), M[sl].to(dev)
                pred = self._forward(x, m, ei, ew)
                preds.append(pred.cpu().numpy())
        pred_windows = np.concatenate(preds, axis=0)
        pred_full = reassemble(pred_windows, win.starts, self.W, T)
        # Keep observed values; fill missing/hidden cells with GRIN's prediction.
        return np.where(np.isnan(full_masked_tensor), pred_full, full_masked_tensor)


# ---------------------------------------------------------------------------
# full evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation() -> int:
    """Canonical GRIN run: seeded, capped, checkpointed, per-victim instrumented.

    Fits GRIN once on clean 2021-2024, saves the checkpoint, then imputes every
    mask ONCE — collecting the aggregated results (all 24 masks), the per-victim
    station errors for the station-outage masks, and the outage imputed tensors
    (so downstream analyses re-score without re-imputing).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)
    W = int(config["deep"]["window_hours"])
    station_ids = list(json.load(open(AXES_JSON))["station_ids"])

    device = select_device(config["deep"])
    seed = int(config["deep"].get("seed", config["project"]["seed"]))
    print(f"Device: {device} | graph=adjacency_knn_basin | seed={seed} | batch_size={GRIN_BATCH_SIZE} "
          f"max_epochs={config['deep']['max_epochs']} patience={config['deep']['patience']}")

    train, val = build_training_windows(truth, observed, meta, W)
    model = GRINImputer(config, device=device, adjacency="adjacency_knn_basin")

    t0 = time.time()
    model.fit(train, val)
    fit_time = time.time() - t0
    model.save_checkpoint(GRIN_CHECKPOINT)
    print(f"training device: {device} | fit time: {fit_time / 60:.1f} min | checkpoint saved")

    # Impute every mask ONCE; capture aggregated + per-victim + outage tensors.
    t1 = time.time()
    overall, per_victim, outage_imputations = [], [], {}
    for i, mask in enumerate(load_masks(), start=1):
        logger.info("[%d/24] impute+score %s", i, mask.id)
        imputed = model.impute(make_model_input(truth, mask))
        res = score(imputed, mask, meta, truth)
        cfg = _config_key(mask)
        for tgt in TARGET_FEATURES:
            if tgt not in res["overall"]:
                continue
            for metric, value in res["overall"][tgt].items():
                if metric == "n":
                    continue
                overall.append({"baseline": "grin", "config": cfg, "mechanism": mask.mechanism,
                                "seed": mask.seed, "target": tgt, "metric": metric, "value": value})
        if mask.mechanism == "station_outage":
            outage_imputations[mask.id] = imputed.astype(np.float32)
            for s_idx, entry in res["per_victim"].items():
                for tgt, mets in entry.items():
                    for metric in ("MAE", "RMSE"):
                        per_victim.append({
                            "station_id": station_ids[s_idx], "station_idx": int(s_idx),
                            "config": cfg, "seed": mask.seed, "target": tgt,
                            "metric": metric, "value": mets[metric], "n_cells": mets["n"],
                        })
    eval_time = time.time() - t1

    # Aggregated results (same schema as the other model CSVs).
    agg = (pd.DataFrame(overall)
           .groupby(["baseline", "config", "mechanism", "target", "metric"])["value"]
           .agg(["mean", "std", "count"]).reset_index().rename(columns={"count": "n_seeds"}))
    agg.to_csv(GRIN_RESULTS_CSV, index=False)
    pd.DataFrame(per_victim).to_csv(GRIN_PER_VICTIM_CSV, index=False)
    np.savez_compressed(GRIN_OUTAGE_NPZ, **outage_imputations)

    total = fit_time + eval_time
    print(f"\nWall-clock: fit {fit_time / 60:.1f} min + eval {eval_time / 60:.1f} min "
          f"= {total / 60:.1f} min total")
    print(f"Wrote {GRIN_RESULTS_CSV}")
    print(f"Wrote {GRIN_PER_VICTIM_CSV}  ({len(per_victim)} rows)")
    print(f"Wrote {GRIN_OUTAGE_NPZ}  ({len(outage_imputations)} outage masks)")
    print(f"Wrote {GRIN_CHECKPOINT}")
    return 0


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def _benchmark_one_epoch(config, train, val, dev_name: str) -> str:
    """Time a single training epoch on a device; report failures gracefully."""
    try:
        dev = torch.device(dev_name)
        model = GRINImputer(config, device=dev, epochs=1, patience=None)
        t0 = time.time()
        model.fit(train, val)
        return f"{dev_name}: 1-epoch fit {time.time() - t0:.1f}s  [ok]"
    except Exception as exc:
        return f"{dev_name}: FAILED - {type(exc).__name__}: {str(exc)[:140]}"


def run_smoke() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    truth = load_truth()
    observed = ~np.isnan(truth)
    W = int(config["deep"]["window_hours"])

    ei, ew, n_nodes = load_graph_edges("adjacency_knn_basin")
    print(f"basin-aware graph: {n_nodes} nodes, {ei.shape[1]} directed edges "
          f"(weight range {ew.min():.2e}..{ew.max():.2e})")

    train, val = build_training_windows(truth, observed, meta, W)
    print(f"windows -> train: {len(train)}  val: {len(val)}  (whole-network [B,{W},{n_nodes},3])")

    print("\n--- per-epoch device benchmark (1 epoch each) ---")
    print("  " + _benchmark_one_epoch(config, train, val, "cpu"))
    print("  " + _benchmark_one_epoch(config, train, val, "mps"))

    print("\n--- smoke train (3 epochs, CPU) + score one mask ---")
    model = GRINImputer(config, device=torch.device("cpu"), epochs=3, patience=None)
    t0 = time.time()
    model.fit(train, val)
    print(f"3-epoch train: {time.time() - t0:.1f}s | params: "
          f"{sum(p.numel() for p in model.model.parameters()):,}")

    mask = {m.id: m for m in load_masks()}["station_outage_v14_durfull_seed0"]
    t1 = time.time()
    imputed = model.impute(make_model_input(truth, mask))
    print(f"impute (1 mask): {time.time() - t1:.1f}s")
    res = score(imputed, mask, meta, truth)
    for tgt in ("WVHT", "APD"):
        m = res["overall"][tgt]
        print(f"  {tgt}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MRE={m['MRE']:.3f}")

    wf = meta["feature_names"].index("WVHT")
    ts, ss = np.where(mask.hidden[:, :, wf])
    wvht = inverse_transform(imputed[ts, ss, wf], wf, meta)
    print(f"\nrecovered WVHT (m) at hidden cells: finite={np.isfinite(wvht).all()}  "
          f"min={wvht.min():.2f}  median={np.median(wvht):.2f}  max={wvht.max():.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GRIN imputer.")
    parser.add_argument("--full", action="store_true",
                        help="Run the full evaluation (fit once, score all masks, save CSV).")
    args = parser.parse_args()
    return run_full_evaluation() if args.full else run_smoke()


if __name__ == "__main__":
    raise SystemExit(main())
