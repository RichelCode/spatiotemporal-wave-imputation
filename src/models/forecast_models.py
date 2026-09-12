"""Additional deep forecasters, leakage-safe, sharing the GraphWaveNet harness.

Every model here is a direct multi-horizon forecaster: input a window of W_IN=24 h
across all stations and features, predict the next MODEL_HORIZON=24 h, and score at
horizons 1/3/6/12/24 through the SAME leakage-safe origins and metrics as
persistence, AR(24) and GraphWaveNet (see forecast_deep.score_forecaster). Trained
on the GRIN-completed 2021-2024 tensor; scored on causal forward-filled test inputs.

Temporal models (channel-independent / cross-variate, no graph):
  DLinear       -- decomposition + linear map (Zeng et al. 2023)
  PatchTST      -- patching + transformer encoder, channel-independent (Nie et al. 2023)
  iTransformer  -- inverted attention over variates (Liu et al. 2024)

Spatiotemporal graph models (reuse the basin-aware kNN graph, via tsl):
  DCRNN         -- diffusion-convolutional recurrent network (Li et al. 2018)
  AGCRN         -- adaptive graph convolutional recurrent network (Bai et al. 2020)

Each model trains once (seeded, capped, checkpointed to data/processed/, gitignored)
and writes reports/forecast_<name>_results.csv, which build_paper.py and
make_forecast_plots.py pick up automatically.

Run one:   python -m src.models.forecast_models --model dlinear
Run all:   python -m src.models.forecast_models --all
Re-score a saved checkpoint (no retraining):
           python -m src.models.forecast_models --model dlinear --rescore
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.download import load_config
from ..features.preprocess import load_meta
from .deep_common import select_device
from .forecast_deep import (
    EVAL_HORIZONS, MODEL_HORIZON, PROCESSED_DIR, W_IN, causal_fill_norm,
    load_graph_edges, norm_completed, score_forecaster,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
TARGET_NAMES = ["WVHT", "APD"]


def _ckpt(name: str) -> Path:
    return PROCESSED_DIR / f"forecast_{name}_checkpoint.pt"


def _csv(name: str) -> Path:
    return REPORTS_DIR / f"forecast_{name}_results.csv"


# ===========================================================================
# model architectures
# ===========================================================================
class _SeriesDecomp(nn.Module):
    """Moving-average trend/seasonal decomposition (edge-padded, length-preserving)."""

    def __init__(self, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=0)

    def forward(self, x):  # x [B*, L]
        pad = (self.kernel - 1) // 2
        front = x[:, :1].repeat(1, pad)
        end = x[:, -1:].repeat(1, self.kernel - 1 - pad)
        padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(padded.unsqueeze(1)).squeeze(1)
        return x - trend, trend


class DLinearModel(nn.Module):
    """DLinear: channel-independent decomposition-linear forecaster (Zeng et al. 2023)."""

    def __init__(self, win: int, horizon: int, kernel: int = 25, **_):
        super().__init__()
        self.win, self.horizon = win, horizon
        self.decomp = _SeriesDecomp(kernel)
        self.lin_seasonal = nn.Linear(win, horizon)
        self.lin_trend = nn.Linear(win, horizon)

    def forward(self, x):  # x [B, win, N, F] -> [B, horizon, N, F]
        B, L, N, Fd = x.shape
        z = x.permute(0, 2, 3, 1).reshape(B * N * Fd, L)
        seasonal, trend = self.decomp(z)
        out = self.lin_seasonal(seasonal) + self.lin_trend(trend)      # [B*N*F, horizon]
        return out.reshape(B, N, Fd, self.horizon).permute(0, 3, 1, 2)


# ===========================================================================
# forecaster harness (non-graph; mirrors forecast_deep.GraphForecaster)
# ===========================================================================
class TemporalForecaster:
    """Direct multi-horizon forecaster for non-graph models (forward is model(x))."""

    uses_graph = False

    def __init__(self, name: str, build_fn, config: dict, device=None,
                 horizon: int = MODEL_HORIZON, win: int = W_IN,
                 epochs: int | None = None, patience: int | None = -1,
                 batch_size: int = 32, lr: float | None = None, seed: int | None = None,
                 arch: dict | None = None):
        deep = config["deep"]
        self.name = name
        self._build_fn = build_fn
        self.win, self.horizon = int(win), int(horizon)
        self.lr = float(lr if lr is not None else deep["learning_rate"])
        self.epochs = int(epochs if epochs is not None else deep["max_epochs"])
        self.patience = deep["patience"] if patience == -1 else patience
        self.device = device if device is not None else select_device(deep)
        self.batch_size = int(batch_size)
        self.seed = int(seed if seed is not None else deep.get("seed", config["project"]["seed"]))
        self.arch = arch or {}
        self.target_idx = [load_meta()["feature_names"].index(t) for t in TARGET_NAMES]
        self.model = None

    # -- graph hook (overridden by the graph subclass) --
    def _prep_graph(self):
        pass

    def _forward(self, x):
        return self.model(x)

    def _build(self, F):
        return self._build_fn(win=self.win, horizon=self.horizon, n_features=F,
                              n_nodes=getattr(self, "n_nodes", None), **self.arch)

    def _windows(self, tensor, origins):
        in_idx = origins[:, None] + torch.arange(-self.win + 1, 1, device=origins.device)
        out_idx = origins[:, None] + torch.arange(1, self.horizon + 1, device=origins.device)
        return tensor[in_idx], tensor[out_idx]

    def _seed_all(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.manual_seed(self.seed)

    def fit(self, completed_norm, train_end, val_start, val_end):
        self._seed_all()
        dev = self.device
        F = completed_norm.shape[2]
        self.model = self._build(F).to(dev)
        self._prep_graph()
        data = torch.tensor(completed_norm, dtype=torch.float32, device=dev)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        tgt = self.target_idx
        train_o = torch.arange(self.win - 1, train_end - self.horizon, device=dev)
        val_o = torch.arange(val_start, val_end - self.horizon, device=dev)

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            perm = train_o[torch.randperm(len(train_o), device=dev)]
            running, nb = 0.0, 0
            for s in range(0, len(perm), self.batch_size):
                o = perm[s:s + self.batch_size]
                x, y = self._windows(data, o)
                pred = self._forward(x)
                loss = torch.abs(pred[..., tgt] - y[..., tgt]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
                nb += 1
            vloss = self._val_loss(data, val_o, tgt)
            logger.info("[%s] epoch %d - train %.4f - val %.4f", self.name, epoch, running / max(1, nb), vloss)
            if vloss < best:
                best, best_state, bad = vloss, copy.deepcopy(self.model.state_dict()), 0
            elif self.patience is not None:
                bad += 1
                if bad >= self.patience:
                    logger.info("[%s] early stop at epoch %d (best %.4f)", self.name, epoch, best)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _val_loss(self, data, val_o, tgt):
        self.model.eval()
        total, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(val_o), self.batch_size):
                o = val_o[s:s + self.batch_size]
                x, y = self._windows(data, o)
                pred = self._forward(x)
                total += torch.abs(pred[..., tgt] - y[..., tgt]).mean().item()
                nb += 1
        return total / max(1, nb)

    def save_checkpoint(self, path=None):
        torch.save(self.model.state_dict(), path or _ckpt(self.name))

    def load_checkpoint(self, path=None, n_features: int = 3):
        self.model = self._build(n_features).to(self.device)
        self._prep_graph()
        self.model.load_state_dict(torch.load(path or _ckpt(self.name), map_location=self.device))
        return self

    def predict_origins(self, input_source, origin_times):
        dev = self.device
        data = torch.tensor(input_source, dtype=torch.float32, device=dev)
        origins = torch.tensor(origin_times, dtype=torch.long, device=dev)
        self.model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(origins), self.batch_size):
                o = origins[s:s + self.batch_size]
                in_idx = o[:, None] + torch.arange(-self.win + 1, 1, device=dev)
                out.append(self._forward(data[in_idx]).cpu().numpy())
        return np.concatenate(out, axis=0)


# ===========================================================================
# registry
# ===========================================================================
def _dlinear_build(win, horizon, n_features, **_):
    return DLinearModel(win=win, horizon=horizon)


REGISTRY = {
    "dlinear": dict(build=_dlinear_build, epochs=40, batch_size=64, arch={}),
}


def make_forecaster(name: str, config, device=None, **over):
    spec = REGISTRY[name]
    return TemporalForecaster(name, spec["build"], config, device=device,
                              epochs=over.get("epochs", spec.get("epochs")),
                              batch_size=over.get("batch_size", spec.get("batch_size", 32)),
                              arch=spec.get("arch", {}))


# ===========================================================================
# train + score
# ===========================================================================
def run_model(name: str, rescore: bool = False) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    meta = load_meta()
    device = torch.device("cpu")  # CPU: reproducible and avoids MPS OOM on full-tensor predict
    model = make_forecaster(name, config, device=device)

    if rescore:
        model.load_checkpoint()
        print(f"[{name}] loaded checkpoint; re-scoring at {EVAL_HORIZONS}")
    else:
        train_end = int(meta["split"]["train"]["end"])
        val_start = int(meta["split"]["val"]["start"])
        val_end = int(meta["split"]["val"]["end"])
        completed = norm_completed()
        print(f"[{name}] training on CPU (seed={model.seed}, epochs={model.epochs}, win={W_IN}, horizon={MODEL_HORIZON})")
        t0 = time.time()
        model.fit(completed, train_end, val_start, val_end)
        model.save_checkpoint()
        print(f"[{name}] fit {(time.time() - t0) / 60:.1f} min; checkpoint saved")

    results = score_forecaster(model, meta, EVAL_HORIZONS, sample=None)
    df = pd.DataFrame(results)[["method", "target", "horizon", "n_origins",
                                "MAE", "RMSE", "skill_vs_persistence"]]
    df["method"] = name  # score_forecaster tags with model.name
    _csv(name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_csv(name), index=False)
    print(f"[{name}] wrote {_csv(name)}")
    for r in results:
        print(f"  {r['target']} h={r['horizon']:>2d}: MAE={r['MAE']:.4f} "
              f"skill={r['skill_vs_persistence']:+.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Additional deep forecasters.")
    parser.add_argument("--model", choices=sorted(REGISTRY), help="model to train/score")
    parser.add_argument("--all", action="store_true", help="run every registered model")
    parser.add_argument("--rescore", action="store_true", help="load checkpoint and re-score only")
    args = parser.parse_args()
    if args.all:
        for name in REGISTRY:
            run_model(name, rescore=args.rescore)
        return 0
    if not args.model:
        parser.error("pass --model <name> or --all")
    return run_model(args.model, rescore=args.rescore)


if __name__ == "__main__":
    raise SystemExit(main())
