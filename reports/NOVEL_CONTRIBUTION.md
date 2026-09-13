# Novel contribution: swell propagation explains (and improves) graph-based wave forecasting

**One-line thesis.** A national buoy network's wave statistics *recover the physics of swell
propagation*, and that propagation is the mechanism behind the well-known advantage of
spatiotemporal graph models — an explanation, validated three ways, that turns our existing
empirical study into a physically-grounded, publishable contribution.

---

## 1. The gap we must close
Our current study is careful and complete, but its headline claim — "spatiotemporal graph
models beat temporal-only methods" — is **already established** in the ML literature (traffic,
air quality, sensor networks). Applying known models (GRIN, GraphWaveNet, DCRNN, AGCRN) to a
new dataset (NDBC waves) is a strong *application*, not a *novel* contribution. A top venue
reviewer will say "known methods, new data." We need one genuine scientific or methodological
contribution on top.

## 2. The contribution
Instead of chasing a fractional accuracy win over SOTA, we **explain and physically validate
why the graph wins**, and derive a principled modeling implication. Proposed framing:

> **"Swell propagation is why graphs beat temporal models: recovering wave dynamics from a
> national buoy network for wave-energy nowcasting and forecasting."**

Three pillars — **all now supported by de-risking experiments we have run**:

### Pillar 1 — The network recovers wave physics *(the discovery)*
For 1,068 within-basin buoy pairs, the lag that maximises wave-height cross-correlation tracks
the deep-water group-velocity prediction `tau = d / c_g`, `c_g = gT/4pi`:
- **Spearman rho = 0.80** (p ~ 1e-238) between observed lag and physics prediction.
- **99%** of pairs obey the causal bound `|tau_obs| <= d/c_g` — waves never outrun their group
  velocity, and the data knows it.
- Coherent **within every basin**, strongest on the West Coast (**rho = 0.89**, the long Pacific
  swell corridor).
- The below-diagonal spread is itself a **dispersion signature** (fast long-period swell arrives
  ahead of the mean-period prediction).

No one has shown a national buoy network recovers the dispersion relation from data alone. This
is a genuinely new, physically meaningful result. **(Figure: observed vs predicted lag.)**

### Pillar 2 — Propagation is the mechanism behind the graph advantage *(the explanation)*
The graph edge that matters is an upstream buoy acting as a **causal preview**: a buoy ~`h` hours
upstream lets us see, from already-observed data, the swell that will arrive `h` hours from now.
Two independent lines of evidence:
- **Our own existing results** already show forecast skill *grows monotonically with horizon*
  (fig10/11) — exactly what the preview mechanism predicts (longer horizon => more distant
  upstream buoys become usable previews).
- A controlled test: adding **propagation-lag-aligned** upstream features beats adding
  **same-time** neighbour features at 24 h (MAE 0.492 vs 0.502 vs 0.523 for own-history alone).
  The effect is modest and horizon-dependent (own-history dominates at 6-12 h), which is *itself*
  the honest, physically-sensible finding: propagation matters once swell has travelled far
  enough to matter.

This reframes the entire 8-model comparison (persistence, AR, DLinear, PatchTST, iTransformer,
GraphWaveNet, DCRNN, AGCRN) as **evidence for the mechanism**: the two tiers we found — every
graph model above every temporal model at horizons >= 6 h — are what swell propagation predicts.

### Pillar 3 — Practical payoff for wave energy *(the application)*
- **Sensor-free reconstruction works.** Recovering a totally dark buoy's full wave-energy record
  from neighbours (our station-outage regime) beats climatology (GRIN **0.346** m vs **0.378** m).
  Reframed: we reconstruct the wave-energy resource at an ungauged site from the network.
- **Swell-corridor maps.** The directed propagation graph (721 edges, 133/140 buoys) is a
  data-derived map of basin-scale swell pathways — a scientific product in its own right.
  **(Figure: swell-corridor map.)**
- **Modeling implication.** A **propagation-aware graph** (directed, lag-carrying edges built from
  the empirical lead-lags) is the principled successor to the generic distance graph. We present
  it as the mechanism-motivated design; the honest scope is that its accuracy gain over a learned
  distance graph is modest, so we lead with the *explanation*, not a leaderboard win.

## 3. Why this clears the bar
- **Novelty:** a mechanistic, physics-validated *explanation*, not new-method-on-new-data.
- **Rigor:** every claim is backed by an experiment already run; the existing study is the
  evidence base.
- **Depth + impact:** real ocean physics (dispersion, group velocity, swell corridors) meets a
  concrete renewable-energy payoff (resource assessment at ungauged sites).
- **Honesty:** we state plainly where propagation matters (long horizons) and where it does not
  (short horizons), and we do not oversell the propagation-graph's accuracy.

## 4. Money figures
1. Observed vs predicted propagation lag (rho = 0.80) — data recovers wave physics.
2. Swell-corridor map — directed propagation graph over the basins.
3. Skill vs horizon for the two tiers (existing fig10) — the mechanism's fingerprint.
4. Sensor-free reconstruction vs climatology at ungauged sites (existing station-outage).

## 5. Paper outline
1. Introduction — the gap, the mechanism thesis, contributions.
2. Data & network (existing).
3. **Swell propagation in the network** — lead-lag vs group velocity; corridor maps. *(new)*
4. **Propagation as the graph mechanism** — preview test; horizon dependence; reinterpreting the
   8-model comparison. *(new framing of existing results)*
5. Sensor-free reconstruction for wave-energy resource assessment. *(reframed)*
6. Propagation-aware graph — the modeling implication, honest scope. *(new)*
7. Discussion, limitations, conclusion.

## 6. Honest scope and remaining work
- Formalise Experiments 1-4 into committed, reproducible analysis in the repo.
- Polish the corridor map (coastlines, per-basin panels).
- Optional (higher effort, uncertain payoff): train a full lag-aware graph GNN and quantify any
  accuracy gain; if it helps materially, promote it; if not, it remains the principled-design
  section.
- Confirm the `T_e = alpha * APD` and deep-water assumptions with co-authors (already flagged).

## 7. Evidence log (numbers)
| Experiment | Result |
| --- | --- |
| Lag vs `d/c_g` (within-basin, n=1068) | Spearman rho = 0.80; causal bound 99%; West Coast rho = 0.89 |
| Propagation preview vs same-time neighbours, h=24 (WVHT MAE) | 0.492 (preview) < 0.502 (same-time) < 0.523 (own) |
| Sensor-free reconstruction (station-outage, WVHT MAE) | GRIN 0.346 < climatology 0.378 |
| Directed swell-propagation graph | 721 edges, 133/140 buoys, median lag 5 h, median coupling 0.69 |
| Off-the-shelf sensor-free *forecasting* (honest negative) | worse than climatology; needs purpose-training, not claimed |
