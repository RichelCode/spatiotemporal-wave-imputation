"""Construct the spatial adjacency for the 140-station network.

Builds a Gaussian-weighted k-nearest-neighbour graph over the buoy locations as
a standalone, reusable artifact (plus a diagnostic figure). Two adjacency
variants are produced so we can later ablate the value of basin boundaries:

  adjacency_knn        symmetric kNN graph, Gaussian edge weights
  adjacency_knn_basin  the same graph with every cross-basin edge zeroed

Construction:
  * full 140x140 haversine distance matrix (km),
  * for each node, its k nearest neighbours (k from config, default 8),
  * edge weight = exp(-d^2 / (sigma_i * sigma_j)) with LOCAL self-tuning scale
    sigma_i = distance from node i to its k-th nearest neighbour. Local scaling
    (Zelnik-Manor & Perona) keeps every kNN edge meaningfully weighted whether
    the node sits in a dense coastal cluster or is a remote offshore buoy; a
    single global sigma would underflow long edges to zero,
  * symmetrise by union: an edge survives if i is a neighbour of j OR vice
    versa. sigma_i*sigma_j is symmetric, so the "average of directed weights"
    is just that weight on the union.

Outputs (gitignored build artifacts, in ``data/processed/``):
  wave_graph.npz    adjacency_knn, adjacency_knn_basin, distance_km, station_order
  graph_meta.json   k, sigma, basin-aware flag, station order, per-variant stats

Diagnostic figure: ``reports/figures/fig8_spatial_graph.{png,pdf}``.

Run:  ``python -m src.features.build_graph``
Importable; nothing runs on import.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.download import load_config
from .eda import BASIN_COLORS, FIG_DIR, haversine, set_style

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AXES_JSON = PROCESSED_DIR / "wave_tensor_axes.json"
SELECTED_CSV = PROJECT_ROOT / "reports" / "selected_stations.csv"
GRAPH_NPZ = PROCESSED_DIR / "wave_graph.npz"
GRAPH_META_JSON = PROCESSED_DIR / "graph_meta.json"


def load_stations_in_tensor_order():
    """Station table reordered to exactly match the tensor's station axis."""
    station_ids = list(json.load(open(AXES_JSON))["station_ids"])
    stations = (
        pd.read_csv(SELECTED_CSV, dtype={"station_id": str})
        .set_index("station_id")
        .loc[station_ids]
        .reset_index()
    )
    return stations, station_ids


def distance_matrix(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Full symmetric haversine distance matrix (km), zero diagonal."""
    n = len(lats)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(lats[i], lons[i], lats[j], lons[j])
            D[i, j] = D[j, i] = d
    return D


def build_knn_gaussian(D: np.ndarray, k: int):
    """Symmetric locally-scaled Gaussian kNN adjacency.

    Returns (adjacency, sigma_local, knn_mask). ``sigma_local[i]`` is the
    distance from i to its k-th nearest neighbour (the local scale);
    ``knn_mask[i, j]`` is True if j is among i's k nearest.
    """
    n = D.shape[0]
    knn_mask = np.zeros((n, n), dtype=bool)
    sigma_local = np.zeros(n, dtype=np.float64)
    for i in range(n):
        order = np.argsort(D[i])
        neighbours = order[order != i][:k]
        knn_mask[i, neighbours] = True
        sigma_local[i] = D[i, neighbours[-1]]  # distance to the k-th neighbour

    # Guard against a zero local scale (coincident stations) to avoid 0/0.
    positive = sigma_local[sigma_local > 0]
    if positive.size:
        sigma_local[sigma_local == 0] = positive.min()

    scale = np.outer(sigma_local, sigma_local)  # sigma_i * sigma_j
    weights = np.exp(-(D ** 2) / scale)

    union = knn_mask | knn_mask.T  # keep edge if either direction selected it
    adjacency = weights * union
    np.fill_diagonal(adjacency, 0.0)
    return adjacency, sigma_local, knn_mask


def apply_basin_constraint(adjacency: np.ndarray, basins: np.ndarray) -> np.ndarray:
    """Zero every edge that crosses a basin boundary."""
    same_basin = basins[:, None] == basins[None, :]
    return adjacency * same_basin


def graph_stats(adjacency: np.ndarray) -> dict:
    """Edge count, degree summary and isolated nodes for an adjacency."""
    present = adjacency > 0
    degree = present.sum(axis=1)
    isolated = np.where(degree == 0)[0]
    return {
        "n_edges": int(present.sum() // 2),
        "degree": degree,
        "isolated": isolated,
        "min_degree": int(degree.min()),
        "median_degree": float(np.median(degree)),
        "max_degree": int(degree.max()),
    }


def _max_edge_distance(adjacency: np.ndarray, D: np.ndarray) -> float:
    """Longest edge length (km) present in an adjacency — a cross-ocean smell test."""
    present = adjacency > 0
    return float(D[present].max()) if present.any() else 0.0


def build_graph():
    """Build both adjacency variants and persist the graph artifact + meta."""
    config = load_config()
    k = int(config["graph"]["k_neighbors"])

    stations, station_ids = load_stations_in_tensor_order()
    lats = stations["lat"].to_numpy()
    lons = stations["lon"].to_numpy()
    basins = stations["basin"].to_numpy()

    D = distance_matrix(lats, lons)
    A_knn, sigma_local, _ = build_knn_gaussian(D, k)
    A_basin = apply_basin_constraint(A_knn, basins)

    stats_knn = graph_stats(A_knn)
    stats_basin = graph_stats(A_basin)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        GRAPH_NPZ,
        adjacency_knn=A_knn.astype(np.float32),
        adjacency_knn_basin=A_basin.astype(np.float32),
        distance_km=D.astype(np.float32),
        station_order=np.array(station_ids),
    )
    meta = {
        "k_neighbors": k,
        "sigma_local_km": {
            "min": float(sigma_local.min()),
            "median": float(np.median(sigma_local)),
            "max": float(sigma_local.max()),
        },
        "sigma_heuristic": config["graph"]["sigma_heuristic"],
        "weight_formula": "exp(-d^2 / (sigma_i * sigma_j))",
        "basin_aware": bool(config["graph"]["basin_aware"]),
        "station_order": station_ids,
        "variants": {
            "adjacency_knn": {
                "n_edges": stats_knn["n_edges"],
                "min_degree": stats_knn["min_degree"],
                "median_degree": stats_knn["median_degree"],
                "max_degree": stats_knn["max_degree"],
                "n_isolated": int(len(stats_knn["isolated"])),
            },
            "adjacency_knn_basin": {
                "n_edges": stats_basin["n_edges"],
                "min_degree": stats_basin["min_degree"],
                "median_degree": stats_basin["median_degree"],
                "max_degree": stats_basin["max_degree"],
                "n_isolated": int(len(stats_basin["isolated"])),
            },
        },
    }
    with open(GRAPH_META_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Wrote %s and %s", GRAPH_NPZ, GRAPH_META_JSON)

    return stations, D, A_knn, A_basin, sigma_local, k, stats_knn, stats_basin


def plot_graph(stations, A_basin, saved):
    """Diagnostic map: nodes coloured by basin, basin-aware edges by weight."""
    set_style()
    lons = stations["lon"].to_numpy()
    lats = stations["lat"].to_numpy()
    n = len(stations)
    wmax = A_basin.max() if A_basin.max() > 0 else 1.0

    fig, ax = plt.subplots(figsize=(11, 7))
    # Edges underneath, opacity/width scaled by (normalised) weight.
    for i in range(n):
        for j in range(i + 1, n):
            w = A_basin[i, j]
            if w > 0:
                rel = w / wmax
                ax.plot([lons[i], lons[j]], [lats[i], lats[j]], color="0.35",
                        linewidth=0.2 + 1.8 * rel, alpha=0.08 + 0.55 * rel, zorder=1)
    # Nodes on top, coloured by basin.
    for basin in sorted(stations["basin"].unique()):
        sub = stations[stations["basin"] == basin]
        ax.scatter(sub["lon"], sub["lat"], s=32, color=BASIN_COLORS.get(basin, "#777777"),
                   edgecolor="white", linewidth=0.4, label=f"{basin} (n={len(sub)})", zorder=3)

    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title("Figure 8. Basin-aware spatial graph (kNN + Gaussian weights)")
    ax.legend(loc="lower left", title="Basin", ncol=2)
    ax.grid(True, alpha=0.25)
    mean_lat = np.radians(stations["lat"].mean())
    ax.set_aspect(1.0 / max(np.cos(mean_lat), 0.2))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"fig8_spatial_graph.{ext}"
        fig.savefig(path)
        saved.append(path.name)
    plt.close(fig)


def print_diagnostics(D, A_knn, A_basin, sigma_local, k, stats_knn, stats_basin, stations):
    """Print the diagnostics the reviewer needs to sanity-check the graph."""
    print("\n=== Spatial graph diagnostics ===")
    print(f"k (neighbours)        : {k}")
    print(f"sigma_local (k-th NN)  : min={sigma_local.min():.1f} / "
          f"median={np.median(sigma_local):.1f} / max={sigma_local.max():.1f} km")

    print("\n-- adjacency_knn (no basin constraint) --")
    print(f"  edges                : {stats_knn['n_edges']}")
    print(f"  degree min/median/max: {stats_knn['min_degree']} / "
          f"{stats_knn['median_degree']:.1f} / {stats_knn['max_degree']}")
    print(f"  longest edge         : {_max_edge_distance(A_knn, D):.0f} km")
    iso_knn = stats_knn["isolated"]
    print(f"  isolated nodes       : {len(iso_knn)}")

    print("\n-- adjacency_knn_basin (cross-basin edges removed) --")
    removed = stats_knn["n_edges"] - stats_basin["n_edges"]
    print(f"  edges                : {stats_basin['n_edges']}  "
          f"({removed} removed by basin constraint)")
    print(f"  degree min/median/max: {stats_basin['min_degree']} / "
          f"{stats_basin['median_degree']:.1f} / {stats_basin['max_degree']}")
    print(f"  longest edge         : {_max_edge_distance(A_basin, D):.0f} km")

    iso_basin = stats_basin["isolated"]
    if len(iso_basin) == 0:
        print("  isolated nodes       : 0  [OK]")
    else:
        print(f"  *** ISOLATED NODES: {len(iso_basin)} — FLAG ***")
        for idx in iso_basin:
            row = stations.iloc[idx]
            print(f"      [{idx}] {row['station_id']}  {row['basin']}  "
                  f"({row['lat']:.2f}, {row['lon']:.2f})  {row['name']}")

    print("\nstations per basin (for context):")
    for basin, count in stations["basin"].value_counts().items():
        print(f"  {basin:14s}: {count}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    stations, D, A_knn, A_basin, sigma, k, stats_knn, stats_basin = build_graph()
    saved: list[str] = []
    plot_graph(stations, A_basin, saved)
    print_diagnostics(D, A_knn, A_basin, sigma, k, stats_knn, stats_basin, stations)
    print(f"\nsaved figure: {', '.join(saved)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
