# Evaluation Protocol

This document is written before any model is built. It binds all
imputation and forecasting experiments. No result is valid unless it
conforms to the protocol below.

## 1. Two distinct tasks, two distinct evaluations

### 1a. Imputation — hide-and-recover
We evaluate imputation by masking OBSERVED values, imputing them, and
scoring recovery against the held-out ground truth.

- Masking is applied only to cells that are genuinely observed.
- Naturally-missing cells are never scored (no ground truth exists).
- Missingness mechanisms reported:
  - MCAR: points removed completely at random.
  - Block/structured: contiguous temporal gaps per station (mimics real sensor outages).
  - Spatial: whole-station blackouts over intervals (tests recovery of a station from its neighbours).
- Masking ratios reported: 10%, 30%, 50%.
- Results averaged over 3+ seeds; mask seed fixed and logged.

### 1b. Forecasting — temporal origin-based split
Forecasting is evaluated by strict chronological partition. No future
information may inform a prediction at origin t.

- Train: 2021–2023. Validation: 2024. Test: 2025.
- Split is by time only; never random over pooled timestamps.
- Horizons: 1, 3, 6, 12 h ahead.

## 2. The leakage boundary (critical)

Imputation and forecasting must not contaminate each other.

- For any forecast at origin t, every input feature — including any imputed value used as input — must be derivable from data at or before t.
- A bidirectional/diffusion imputer using full-series context may build the TRAINING tensor, but at TEST time forecast inputs are imputed using past-only context relative to each origin.
- Imputation test masks and the forecasting test period are kept disjoint where they could interact; any overlap is documented.
- We explicitly report whether the forecaster consumes (a) raw + mask channel, or (b) a pre-imputed series, and justify the choice.

## 3. Metrics

- Imputation: MAE, RMSE, MRE per target (WVHT, DPD/APD).
- Forecasting: MAE, RMSE per target per horizon; plus persistence-normalised skill score (vs. last-observed-value baseline).
- All metrics reported in physical units (m, s), not normalised space.

## 4. Baselines (a model beats these or it is not reported)

- Imputation: mean fill, forward-fill, linear interpolation, spatial IDW/kriging.
- Forecasting: persistence (t+h = t), seasonal-naive, ARIMA per station.

## 5. Reproducibility

- Every reported number traces to a config + seed + git commit.
- Metrics computed only on the held-out partitions defined above.
