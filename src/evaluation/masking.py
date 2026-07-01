"""Hide-and-recover evaluation harness (EVALUATION.md, section 1a).

Artificially hides genuinely-observed target values so any imputation model can
be scored against known truth. Every model — baseline or deep — is evaluated
through the SAME saved masks, for reproducibility and fair comparison.

Targets scored: WVHT and APD (2 responses). DPD is an input feature and is
never scored. All masks touch only genuinely-observed cells and are confined to
the TEST period (2025) for leakage-safety.

Mechanisms
----------
station_outage (PRIMARY): black out ALL channels of a set of victim stations
    over an interval (whole test period or a fixed duration). This is the
    sensor-outage / recover-from-neighbours scenario the spatial graph exists to
    help with: a victim's own values (including its DPD input) are removed, so
    the model must lean on graph neighbours. Only the victim's target cells are
    scored, and per-victim errors are reported.
mcar (secondary): hide random observed target cells at 10/30/50%.
block (secondary): hide one contiguous temporal gap per station+target, sized
    to 10/30/50% of the test period.

Key objects
-----------
MaskSpec        one mask: mechanism, params, seed, and a boolean ``hidden``
                array [T, S, F] (cells removed from the model input; a subset of
                the observed cells). ``scored`` cells are derived at score time
                as the target-feature subset of ``hidden``.
generate_masks  (mechanism, params, seed) -> MaskSpec
score           (imputed, mask, meta) -> per-target MAE/RMSE/MRE in physical
                units, plus per-victim errors for station_outage.
save_masks / load_masks  persist the suite to data/processed/ (gitignored).

Self-test: ``python -m src.evaluation.masking`` builds the default suite on the
real tensor, saves it, and prints mask statistics. Nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.preprocess import inverse_transform, load_meta

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_NPZ = PROCESSED_DIR / "wave_tensor_model.npz"
META_JSON = PROCESSED_DIR / "preprocess_meta.json"
MASKS_NPZ = PROCESSED_DIR / "eval_masks.npz"
MASKS_MANIFEST = PROCESSED_DIR / "eval_masks_manifest.json"

# The two response variables that are scored. DPD is an input feature only.
TARGET_FEATURES = ["WVHT", "APD"]

# Per-mechanism code, mixed into the seed so identical base seeds give
# independent draws across mechanisms.
_MECH_CODE = {"station_outage": 1, "mcar": 2, "block": 3}


@dataclass
class MaskSpec:
    """One evaluation mask over the [time, station, feature] tensor."""
    id: str
    mechanism: str
    seed: int
    params: dict
    hidden: np.ndarray                      # bool [T, S, F]; removed from model input
    victims: list = field(default_factory=list)   # station indices (station_outage)
    interval: tuple | None = None           # (start, end) time indices, if applicable

    def n_hidden(self) -> int:
        return int(self.hidden.sum())

    def n_scored(self, target_idx: list[int]) -> int:
        return int(sum(self.hidden[:, :, f].sum() for f in target_idx))

    def to_manifest(self, target_idx: list[int]) -> dict:
        return {
            "id": self.id, "mechanism": self.mechanism, "seed": self.seed,
            "params": self.params, "victims": [int(v) for v in self.victims],
            "interval": [int(self.interval[0]), int(self.interval[1])] if self.interval else None,
            "n_hidden": self.n_hidden(), "n_scored": self.n_scored(target_idx),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load_observed() -> np.ndarray:
    return np.load(MODEL_NPZ)["mask"]


def _target_indices(meta: dict) -> list[int]:
    return [meta["feature_names"].index(t) for t in TARGET_FEATURES]


def _test_bounds(meta: dict) -> tuple[int, int]:
    test = meta["split"]["test"]
    return int(test["start"]), int(test["end"])


def _rng(mechanism: str, seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, _MECH_CODE[mechanism]])


# ---------------------------------------------------------------------------
# mechanisms
# ---------------------------------------------------------------------------
def _station_outage(params: dict, seed: int, observed: np.ndarray, meta: dict) -> MaskSpec:
    T, S, F = observed.shape
    t0, t1 = _test_bounds(meta)
    duration = params.get("duration_hours")
    ws, we = (t0, t1) if duration is None else (t0, min(t0 + int(duration), t1))

    n_victims = params.get("n_victims")
    if n_victims is None:
        n_victims = int(round(params["victim_fraction"] * S))
    victims = sorted(int(s) for s in _rng("station_outage", seed).choice(S, n_victims, replace=False))

    hidden = np.zeros((T, S, F), dtype=bool)
    for s in victims:
        # Blackout ALL channels (incl. DPD input) where genuinely observed.
        hidden[ws:we, s, :] = observed[ws:we, s, :]

    dur_label = "full" if duration is None else str(int(duration))
    return MaskSpec(
        id=f"station_outage_v{n_victims}_dur{dur_label}_seed{seed}",
        mechanism="station_outage", seed=seed, params=dict(params),
        hidden=hidden, victims=victims, interval=(ws, we),
    )


def _mcar(params: dict, seed: int, observed: np.ndarray, meta: dict) -> MaskSpec:
    ratio = float(params["ratio"])
    t0, t1 = _test_bounds(meta)
    target_idx = _target_indices(meta)

    candidate = np.zeros_like(observed, dtype=bool)
    for f in target_idx:
        candidate[t0:t1, :, f] = observed[t0:t1, :, f]

    flat = np.flatnonzero(candidate.ravel())
    k = int(round(ratio * flat.size))
    chosen = _rng("mcar", seed).choice(flat, size=k, replace=False)
    hidden = np.zeros(candidate.size, dtype=bool)
    hidden[chosen] = True
    hidden = hidden.reshape(candidate.shape)

    return MaskSpec(
        id=f"mcar_r{int(ratio * 100)}_seed{seed}", mechanism="mcar", seed=seed,
        params=dict(params), hidden=hidden,
    )


def _block(params: dict, seed: int, observed: np.ndarray, meta: dict) -> MaskSpec:
    ratio = float(params["ratio"])
    t0, t1 = _test_bounds(meta)
    target_idx = _target_indices(meta)
    _, S, F = observed.shape
    test_len = t1 - t0
    block_len = min(int(round(ratio * test_len)), test_len)

    rng = _rng("block", seed)
    hidden = np.zeros_like(observed, dtype=bool)
    if block_len > 0:
        for s in range(S):
            for f in target_idx:
                max_start = t1 - block_len
                start = int(rng.integers(t0, max_start + 1)) if max_start > t0 else t0
                seg = slice(start, start + block_len)
                # Hide only the observed cells inside the contiguous window.
                hidden[seg, s, f] = observed[seg, s, f]

    return MaskSpec(
        id=f"block_r{int(ratio * 100)}_seed{seed}", mechanism="block", seed=seed,
        params=dict(params), hidden=hidden,
    )


_GENERATORS = {"station_outage": _station_outage, "mcar": _mcar, "block": _block}


def generate_masks(mechanism: str, params: dict, seed: int,
                   observed: np.ndarray | None = None, meta: dict | None = None) -> MaskSpec:
    """Generate a single MaskSpec for one (mechanism, params, seed).

    ``observed`` (the tensor's boolean observed-mask) and ``meta`` are loaded
    from data/processed/ if not supplied.
    """
    if mechanism not in _GENERATORS:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if observed is None:
        observed = _load_observed()
    if meta is None:
        meta = load_meta()
    return _GENERATORS[mechanism](params, seed, observed, meta)


def default_suite() -> list[tuple[str, dict, int]]:
    """The standard (mechanism, params, seed) grid: >=3 seeds each."""
    specs = []
    for seed in (0, 1, 2):
        specs.append(("station_outage", {"victim_fraction": 0.10, "duration_hours": None}, seed))
        specs.append(("station_outage", {"victim_fraction": 0.10, "duration_hours": 720}, seed))
        for ratio in (0.1, 0.3, 0.5):
            specs.append(("mcar", {"ratio": ratio}, seed))
            specs.append(("block", {"ratio": ratio}, seed))
    return specs


def generate_mask_suite(specs=None) -> list[MaskSpec]:
    """Generate every mask in ``specs`` (default: :func:`default_suite`)."""
    specs = specs or default_suite()
    observed = _load_observed()
    meta = load_meta()
    return [generate_masks(mech, params, seed, observed, meta) for mech, params, seed in specs]


# ---------------------------------------------------------------------------
# model input + scoring
# ---------------------------------------------------------------------------
def make_model_input(tensor: np.ndarray, mask: MaskSpec) -> np.ndarray:
    """Return a copy of ``tensor`` with the mask's hidden cells set to NaN.

    Naturally-missing cells are already NaN; this additionally removes the
    artificially-hidden cells the model must recover.
    """
    x = tensor.astype(np.float64).copy()
    x[mask.hidden] = np.nan
    return x


def _metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    abs_err = np.abs(err)
    denom = np.abs(true).sum()
    return {
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MRE": float(abs_err.sum() / denom) if denom > 0 else float("nan"),
        "n": int(true.size),
    }


def score(imputed: np.ndarray, mask: MaskSpec, meta: dict | None = None,
          truth: np.ndarray | None = None) -> dict:
    """Score a model's imputed tensor on a mask's scored cells (physical units).

    ``imputed`` is the model's full [T, S, F] output in NORMALISED space.
    ``truth`` (the original normalised tensor) and ``meta`` are loaded from
    data/processed/ if not supplied. Metrics (MAE, RMSE, MRE) are per target,
    computed after inverse_transform to physical units (m for WVHT, s for APD).
    For station_outage, per-victim-station errors are also returned.
    """
    meta = meta or load_meta()
    if truth is None:
        truth = np.load(MODEL_NPZ)["tensor"]
    features = meta["feature_names"]
    target_idx = _target_indices(meta)

    result = {"mask_id": mask.id, "mechanism": mask.mechanism, "seed": mask.seed, "overall": {}}
    for f in target_idx:
        cells = mask.hidden[:, :, f]
        if cells.sum() == 0:
            continue
        ts, ss = np.where(cells)
        pred = inverse_transform(imputed[ts, ss, f], f, meta)
        true = inverse_transform(truth[ts, ss, f], f, meta)
        result["overall"][features[f]] = _metrics(pred, true)

    if mask.mechanism == "station_outage":
        per_victim = {}
        for s in mask.victims:
            entry = {}
            for f in target_idx:
                ts = np.where(mask.hidden[:, s, f])[0]
                if ts.size == 0:
                    continue
                pred = inverse_transform(imputed[ts, s, f], f, meta)
                true = inverse_transform(truth[ts, s, f], f, meta)
                entry[features[f]] = _metrics(pred, true)
            per_victim[int(s)] = entry
        result["per_victim"] = per_victim

    return result


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_masks(masks: list[MaskSpec], npz_path: Path = MASKS_NPZ,
               manifest_path: Path = MASKS_MANIFEST) -> None:
    """Persist masks (bit-packed hidden arrays) + a JSON manifest."""
    meta = load_meta()
    target_idx = _target_indices(meta)
    shape = masks[0].hidden.shape
    arrays = {f"hidden__{m.id}": np.packbits(m.hidden.ravel()) for m in masks}
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, shape=np.array(shape, dtype=np.int64), **arrays)
    manifest = {
        "shape": list(shape),
        "target_features": TARGET_FEATURES,
        "masks": [m.to_manifest(target_idx) for m in masks],
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Saved %d masks to %s (+ manifest).", len(masks), npz_path)


def load_masks(npz_path: Path = MASKS_NPZ, manifest_path: Path = MASKS_MANIFEST) -> list[MaskSpec]:
    """Reconstruct the mask suite saved by :func:`save_masks`."""
    z = np.load(npz_path)
    shape = tuple(int(x) for x in z["shape"])
    size = int(np.prod(shape))
    manifest = json.load(open(manifest_path))
    masks = []
    for spec in manifest["masks"]:
        hidden = np.unpackbits(z[f"hidden__{spec['id']}"])[:size].astype(bool).reshape(shape)
        masks.append(MaskSpec(
            id=spec["id"], mechanism=spec["mechanism"], seed=spec["seed"],
            params=spec["params"], hidden=hidden, victims=spec["victims"],
            interval=tuple(spec["interval"]) if spec["interval"] else None,
        ))
    return masks


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    meta = load_meta()
    target_idx = _target_indices(meta)

    masks = generate_mask_suite()
    save_masks(masks)

    rows = []
    for m in masks:
        interval_h = (m.interval[1] - m.interval[0]) if m.interval else None
        rows.append({
            "id": m.id, "mechanism": m.mechanism, "seed": m.seed,
            "n_hidden": m.n_hidden(), "n_scored": m.n_scored(target_idx),
            "n_victims": len(m.victims), "interval_h": interval_h,
        })
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)

    print("\n" + "=" * 72)
    print(f"HIDE-AND-RECOVER MASK SUITE — {len(masks)} masks")
    print(f"targets scored: {TARGET_FEATURES}  (DPD hidden for station-outage but not scored)")
    print("all masks confined to the TEST period, observed cells only")
    print("=" * 72)
    print(table.to_string(index=False))

    print("\n--- per-mechanism totals ---")
    for mech, grp in table.groupby("mechanism"):
        print(f"  {mech:15s}: {len(grp)} masks | mean n_scored={grp['n_scored'].mean():,.0f}")

    # Scoring-path self-test (no model): perfect imputation must give ~0 error;
    # a predict-the-training-mean baseline must give a sensible nonzero error.
    truth = np.load(MODEL_NPZ)["tensor"]
    so = next(m for m in masks if m.mechanism == "station_outage")
    perfect = score(truth, so, meta, truth)
    naive_input = truth.copy()
    naive_input[so.hidden] = 0.0  # 0 in normalised space == training mean
    naive = score(naive_input, so, meta, truth)

    print(f"\n--- scoring self-test on {so.id} ({len(so.victims)} victims) ---")
    for name, res in [("perfect (truth)", perfect), ("predict-train-mean", naive)]:
        parts = ", ".join(f"{t}: MAE={res['overall'][t]['MAE']:.4f}" for t in TARGET_FEATURES
                          if t in res["overall"])
        print(f"  {name:20s}: {parts}")
    worst = sorted(perfect["per_victim"].items(),
                   key=lambda kv: kv[1].get("WVHT", {}).get("MAE", 0), reverse=True)[:1]
    print(f"  per-victim keys present: {len(perfect['per_victim'])} stations "
          f"(perfect WVHT MAE max = {worst[0][1]['WVHT']['MAE']:.2e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
