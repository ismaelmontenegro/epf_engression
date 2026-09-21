# Probabilistic intraday electricity price forecasting using engression

Code for the bachelor thesis of the same name (Karlsruhe Institute
of Technology, Institute of Statistics, Statistical Methods and Econometrics) under supervision of Dr. Sam Allen.

The thesis evaluates a conditional noise-scaling extension of Engression for
forecasting the ten-step German continuous intraday price path, against a
LASSO-bootstrap benchmark and the conditional generative model (CGM) of
Chen et al. (2025), on two chronologically disjoint evaluation windows.

---

## Data availability

**This repository reproduces the old window (2017–2019) only.**

The recent-window dataset (December 2023 – April 2026) and the proprietary
MetDesk forecast-update features belong to the industry partner and are not
redistributable. This was agreed in advance. The code
paths are identical across the two windows (before adding the MetDesk features as explained in section 4.5); only the input data differs.

Reproducible here:

| Thesis result | Content | Produced by |
|---|---|---|
| Table 4 | per-horizon CRPS, raw block, full vs. reduced history | `train_engression_from_cgm.py`, `eval_proper_old.py` |
| Table 6, old-window rows | architecture search: block × history representation | `train_engression_from_cgm.py`, `engression_experiments_old.py`, `eval_proper_old.py` |
| Section 4.1.4, old window | HTS encoder (ES in text) | `train_engression_hts_old.py`, `eval_proper_old.py` |
| Table 8 | main model comparison | all main models, `eval_proper_old.py` |
| Table 10 | Diebold–Mariano tests | `eval_dm.py` |
| Table 13 | threshold-weighted energy score | `eval_proper_old.py` |
| Table 15 | per-horizon PIT calibration | `eval_proper_old.py` |
| Table 17 | pooled tail calibration | `eval_proper_old.py` |
| Tables 18 and 19, old-window rows | target variance by horizon, resid_last and raw | `resid_last_var_by_h.py` |
| Table 20 | spread–skill ratios | `spread_vs_error_h.py` |
| Table 23 | matched fixed-window comparison | `lasso_stationary.py`, `lasso_bootstrap_stationary.py`, `eval_proper_old.py` |
| Section 4.3.4, old window | CGM–Engression hybrid (scores in text) | `train_hybrid_cgm.py`, `eval_proper_old.py` |
| Table 28 | realized trading potential by regime | `eval_proper_old.py` |
| Table 30 | scalar index forecasts | `eval_proper_old.py` (path-collapsed, naive), `engression_experiments_indexes.py` + `eval_engression_experiments_indexes.py` (direct univariate), `naive_probabilistic_benchmark.py` (naive probabilistic) |
| Table 32, old-window rows | naive-forecast ensemble | `eval_proper_old.py` |
| Section A.1 | appendix figures, old window | `eval_proper_old.py`, `spread_vs_error_h.py` |

Not reproducible without the proprietary data: every recent-window table and
figure, all MetDesk results (Sections 4.5 and 4.6.5), and the experiments that
were run on the recent window only (Table 11, output-space scale head;
Table 24, ES^β sweep; the GKS loss; Tables 25 and 26, MCB/TMCB fine-tuning; the
rolling-window Engression rows of Table 22).

Table and section numbers refer to the submitted version of the thesis.

---

## Environment

Python 3.13 (the reported runs used 3.13.15).

```bash
# CUDA 12.8 build of PyTorch, as used for the reported runs
# (omit this line for the default CPU/platform build)
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # from the repository root
cd epf_engression                 # all scripts are run from here
```

Exact versions are pinned in `requirements.txt`. Both PyTorch (Engression) and
TensorFlow/Keras (CGM, hybrid, HTS encoder) are required.

The directory `epf_engression/engression_module/` is a modified copy of the
`engression` package by Xinwei Shen and Nicolai Meinshausen
(<https://github.com/xwshen51/engression>, v0.1.14, BSD 3-Clause License). It
is imported as a local package and is not installed via its `setup.py`.

**Run every script from inside `epf_engression/`.** Several scripts use
`./` as their data directory, and the scripts import each other by module
name.

---

## Repository layout

All paths are relative to `epf_engression/`.

```
engression_module/
  models.py                           StoLayer / StoNet, incl. the conditional scale head
  engression.py                       training loop, sampling, optional tail-weighted / GKS losses
  loss_func.py                        energy score losses (two-sample training estimator,
                                      m/(m-1) estimator for evaluation), Gaussian kernel score
  data/                               upstream utilities, not used by the thesis scripts

# Engression
engression_experiments_old.py         Engression training entry point (main results)
train_engression_hts_old.py           Engression with a learned history encoder (HTS)
train_engression_from_cgm.py          Engression on the raw CGM inputs, no feature engineering
engression_experiments_indexes.py     Engression trained directly on scalar ID1/ID2/ID3 proxies

# Benchmarks
cgm_epf.py, cgm_models.py             CGM benchmark (Chen et al. 2025); also writes lasso_y.feather
train_hybrid_cgm.py                   CGM–Engression hybrid, training entry point
latent_perturbation_cgm_models.py     CGM–Engression hybrid, model definition
lasso.py                              LASSO point forecasts (Chen et al. 2025), rolling 397-day calibration window
lasso_bootstrap.py                    bootstrap path ensemble from rolling LASSO residuals (Chen et al. 2025)
lasso_stationary.py                   LASSO point forecasts, fixed training window
lasso_bootstrap_stationary.py         bootstrap path ensemble, fixed residual pool
naive_probabilistic_benchmark.py      naive same-hour bootstrap benchmark

# Evaluation
eval_proper_old.py                    scoring, calibration, trading and scalar-index evaluation
tail_calibration.py                   tail-calibration diagnostics (port of sallen12/TailCalibration)
eval_dm.py                            Diebold–Mariano tests and RTP baselines
eval_engression_experiments_indexes.py  evaluation of the dedicated scalar-index runs
spread_vs_error_h.py                  spread–skill check by horizon
lag_decay.py                          lag–target correlation diagnostic
resid_last_var_by_h.py                target variance by horizon, raw vs. resid_last

ID_DATA/, EXOG_DATA/                  input data (see "Data availability")
lasso_y.feather                       copy of realised targets in LASSO layout (also written by cgm_epf.py)
```

---

## Running the locked Engression model

There is no config-file loader. Settings live in the `ExperimentConfig`
dataclass at the top of `engression_experiments_old.py`. To reproduce a
particular run, edit the fields listed under "Configurations" below and run:

```bash
python engression_experiments_old.py
```

**Set `data_dir` first.** The default is the author's local path and will not
exist on your machine; point it at `epf_engression/` (the directory containing
`ID_DATA/` and `EXOG_DATA/`). Note that `OUT_DIR.mkdir()` executes at import
time, so this path is created as soon as the module is loaded, including when
`lag_decay.py` or `resid_last_var_by_h.py` import from it.

Each run writes to `<data_dir>/<out_subdir>/<run_name>/`, where `run_name`
encodes the configuration:

| File | Contents |
|---|---|
| `pred.npy` | `(4800, 10, 1000)` predictive path samples, test set |
| `y_test.npy` | `(4800, 10)` realised test paths |
| `y_train.npy` | realised training paths (used for tail-calibration thresholds) |
| `pred_norm.npy`, `y_test_norm.npy` | the same in model-internal units |
| `anchor_test.npy` | the `last_p` anchor, for non-raw target modes |
| `config.json` | the exact configuration used |
| `feature_columns.json` | the 20 engineered input columns, in order |
| `normalization_stats.json` | training-block means and standard deviations |
| `manifest.json` | run name, paths, observation count, device |

---

## Configurations

The four runs below are the Engression rows of Table 8 and the four
single-component ablation comparisons of Table 10. They form a 2×2 design over
the two locked design choices: the target transformation and the conditional
noise-scaling head.

| Run | `target_mode` | `hetero_noise` | Thesis label |
|---|---|---|---|
| 1 | `"raw"` | `False` | Engression, fixed-scale noise, raw target |
| 2 | `"resid_last"` | `False` | Engression, fixed-scale noise |
| 3 | `"raw"` | `True` | Engression, raw target, conditional noise-scaling |
| 4 | `"resid_last"` | `True` | **Engression, locked configuration** |

Settings shared by all four runs (set these explicitly; do not rely on the
dataclass defaults):

- `benchmark_mode="compact"`, `feature_set="compact_v1"` — 27 inputs: 20
  engineered columns, of which 19 are distinct (see "Data conventions"), plus
  a 7-dimensional weekday one-hot
- `num_epochs=2000` (the old-window schedule; the recent window uses 200)
- 2 layers, hidden width 128, noise dimension 32, scale-head hidden width 256
- the scale head conditions on the full 27-dimensional input
  (`scale_cond_idx=None`)
- learning rate 1e-4, batch size 1024, no early stopping
- 10-member ensemble, 100 samples per member, 1,000 pooled draws per forecast
  origin
- 24,362 trainable parameters per ensemble member with the scale head, 8,970
  without

**Seeds.** `torch.manual_seed(seed_base + k)` and `np.random.seed(seed_base + k)`
for `k = 0..9`, so `seed_base = 1000` gives seeds **1000–1009**. All reported
runs use this base.


---

## Benchmarks and further models

### LASSO bootstrap

```bash
python lasso.py            # rolling LASSO point forecasts -> lasso_xy.feather
python lasso_bootstrap.py  # -> lasso_bootstrap.npy, shape (4800, 10, 10000)
```

`lasso.py` fits one `LassoCV` model per delivery hour, forecast day and
sub-period on a rolling 397-day calibration window, with median/MAD scaling
and an asinh transform on its inputs, and back-transforms by smearing over the
in-sample residuals. It runs in parallel via `multiprocessing`.
Set `home_dir` at the top of the file first.

`lasso_bootstrap.py` adds out-of-sample LASSO residuals from the 240 preceding
forecast days of the same hour to the point forecast, 10,000 draws per forecast
origin. The same resampled day is used for all ten horizons, so the empirical
dependence across the path is preserved.

The fixed-window variant used for the robustness check (Table 23) is:

```bash
python lasso_stationary.py            # fixed training days 7–510 -> lasso_stationary_xy.feather
python lasso_bootstrap_stationary.py  # residual pool: validation block -> lasso_bootstrap_stationary.npy
```

Here the LASSO is trained once per hour and sub-period on the same 504-day
training block as the neural models, and the residual pool is the fixed set of
out-of-sample residuals on the 126-day validation block.

### CGM

```bash
python cgm_epf.py          # -> pred_cgm_esloss.npy, shape (4800, 10, 10000)
```

Energy-score loss only (`loss_index_w = 0.0`), latent dimension 100, 200
training samples, learning rate 1e-4, batch size 1024, at most 100 epochs with
early stopping (patience 10) on a validation split equal to the last 20 % of the
training data, i.e. the same validation block the Engression runs hold out.
10 runs × 1,000 draws. `cgm_epf.py` also writes `lasso_y.feather`, which the
LASSO scripts read; a copy is included in the repository.

### CGM–Engression hybrid

```bash
mkdir hybrid_outputs
python train_hybrid_cgm.py # -> hybrid_outputs/pred_hybrid_esloss_2.npy
```

Same inputs and training setup as the CGM. The CGM's scaled latent noise is
passed through a linear layer and added to a deterministic latent state
computed from the inputs (Engression-style pre-additive noise), and the sum is
decoded by a nonlinear network. Create `hybrid_outputs/`
before running; `np.save` does not create it and the script fails only after
training. This script is fully seeded (`SEED = 42`, `SEED + ens` per member)
and enables TensorFlow op determinism.

### Architecture-search variants (Tables 4 and 6, Section 4.1.4)

All architecture-search runs use fixed-scale noise, the raw target and a fixed
200-epoch schedule.

- CGM raw block rows of Tables 4 and 6: `train_engression_from_cgm.py` with
  `BENCHMARK_MODE = "compact"` (reduced history, 224 inputs) or `"full"`
  (flattened history, 3,359 inputs).
- Engineered block rows of Table 6: `engression_experiments_old.py` with
  `num_epochs=200`, `hetero_noise=False`, `target_mode="raw"`, and
  `benchmark_mode="compact"` (27 inputs) or `"full"` (3,327 inputs; appends
  the flattened 20 × 165 history tensor to the engineered features).
- HTS encoder (Section 4.1.4): `train_engression_hts_old.py`. A Keras history
  encoder trained with MAE on the 20 × 165 history tensor (early stopping,
  patience 10) compresses it to a 16-dimensional embedding, which is appended
  to `compact_v1`. The reported
  result uses the raw target.

### Scalar index forecasts (Tables 30 and 32)

Table 30 compares three forecasts of each proxy:

- **Path-collapsed**: `eval_proper_old.py` collapses the locked model's path
  samples into duration-weighted averages of the sub-periods
  and also computes the deterministic naive (`last_p`) MAE/RMSE and the
  naive-forecast ensemble of Table 32.
- **Direct univariate**: `engression_experiments_indexes.py` trains the locked
  architecture on one scalar proxy (`target_name = "id1_p"`, `"id2_p"` or
  `"id3_p"`, one run each), and `eval_engression_experiments_indexes.py`
  evaluates a run. The reported runs use the script's default of 200 epochs,
  not the 2,000 of the locked path model; 200 epochs performed clearly better
  for the univariate targets.
- **Naive probabilistic**: `naive_probabilistic_benchmark.py` (next section).

### Naive probabilistic benchmark (Table 30)

`naive_probabilistic_benchmark.py` uses the same-contract `last_p` as a point
forecast for the whole path and adds bootstrapped residual paths
(realised path − `last_p`) of the same delivery hour from a rolling window of
preceding days, 10,000 trajectories, seed 123. The window length is set by
`BOOTSTRAP_WINDOW_DAYS` (currently 30). Its CRPS and 80% empirical coverage are reported in table 30.

---

## Evaluation

Once the prediction files exist:

```bash
python eval_proper_old.py  # scores, calibration, trading, scalar indices
python eval_dm.py          # Diebold–Mariano tests and RTP baselines
python spread_vs_error_h.py
```

Adjust the path constants at the top of each file first; they point at the
original machine. `eval_proper_old.py` skips any model whose prediction file is
missing, so partial reproductions work.

Evaluation conventions:

- All models are subsampled to 1,000 draws per forecast origin
  (`N_EVAL_SAMPLES`), and to 200 draws (`N_SCORE_SAMPLES`) for the energy,
  CRPS, Dawid–Sebastiani and variogram scores, with a fixed seed (123), so
  models with 10,000 draws are scored on the same footing as Engression.
- Thresholds are the top and bottom 10 % of test observations by the
  mean price of the realised path.
- The thresholds for the threshold-weighted scores are the 10 % and 90 %
  quantiles of the pooled training-block paths (`y_train.npy`).
- Diebold–Mariano tests use per-observation energy scores and a Newey–West
  variance with 24 lags.

Main outputs of `eval_proper_old.py`, written to `OUT_DIR`:

| File | Contents |
|---|---|
| `distributional_scores_tw.csv` | energy score, CRPS, MAE, DSS, variogram scores; unweighted and threshold-weighted |
| `per_horizon_crps_tw.csv` | CRPS by horizon and weighting |
| `trading_by_regime.csv`, `naive_baselines.csv` | realized trading potential by regime |
| `scalar_id1_id2_id3_summary_metrics*.csv` | scalar index evaluation |
| `pit_calibration/` | PIT metrics, histograms, per-horizon and pooled tail calibration |
| `dispersion_ratio_*.csv` | model versus empirical spread by regime |
| `marginal_distribution_overlay/` | marginal distribution plots by regime |

---

## Data conventions

- `start_date = 2017-06-14` is the day before the first delivery
  day: the `day` column is computed as `(delivery_date − start_date).days`.
  The data cover delivery days 2017-06-15 to 2019-09-29 (837 days).
- `start_index = 168` discards the first 7 days as burn-in for the 168-hour lag
  window.
- Of the remaining 830 days: 504 training (2017-06-22 – 2018-11-07), 126
  validation (2018-11-08 – 2019-03-13), 200 test (2019-03-14 – 2019-09-29), in
  chronological order, never reshuffled. The validation block is held out from
  Engression training and was used for model development; the reported scores
  are all computed on the test block.
- `lead = 4` — inputs are taken at least 4 hours before delivery, except the
  day-ahead price and the day-ahead wind and load forecasts, which are known in
  advance, and the `last_p` anchor at a 3-hour lead.
- Normalisation statistics are computed on the **training block only** and
  saved to `normalization_stats.json`.
- The target is `id_3 … id_12`: sub-period VWAPs spanning three hours to
  30 minutes before delivery, with horizon 1 closest to delivery and horizon 10
  closest to the forecast origin.
- `compact_v1` contains 20 columns representing 19 distinct quantities:
  `id_std_hist_last` and `id_std` are both `input_std[:, 0]` and are therefore
  numerically identical. The duplicate is retained so that this code matches
  the runs reported in the thesis; it is disclosed in the Implementation Note
  of Section 2.2.2.

---

## Reproducibility

Three sources of run-to-run variation are worth noting.

1. **Seeds.** Engression runs are seeded as described above. This fixes weight
   initialisation, batch ordering and noise draws.
2. **CUDA nondeterminism.** The reported runs were trained on a GPU, and cuDNN
   kernels are not bitwise deterministic, so identical seeds on different
   hardware do not reproduce identical outputs. A replication on separate
   hardware produced small numerical differences that left every reported
   ranking and per-horizon profile unchanged.
3. **Unseeded benchmarks.** `lasso_bootstrap.py`,
   `lasso_bootstrap_stationary.py`, `cgm_epf.py` and the HTS encoder in
   `train_engression_hts_old.py` do not set random seeds (as in the original implementations of the LASSO Bootstrap and CGM from the paper by Chen et al. (2025)), so their outputs are
   reproducible only up to Monte Carlo and training error. For the bootstraps,
   at 10,000 resampled trajectories this error is orders of magnitude below the
   score differences reported in the thesis. `train_hybrid_cgm.py` is seeded.

---

## Options in `engression_module` not used by the old-window results

`engression_module` supports several training options that are switched off
in every script here. They were used only for recent-window experiments or
not at all in the thesis:

- `hetero_output` (output-space scale head, Table 11) is hardcoded to `False`
  in `fit_single_engression`.
- `beta` (the ES^β loss, Table 24) is left at its default of 1.
- `gks_pure`, `gks_lambda`, `gks_sigma` (Gaussian kernel score loss,
  Section 4.3.1 footnote) are not passed.
- `upper_twes_lambda` (upper-tail threshold-weighted training loss) is
  hardcoded to `0.0`; no result in the thesis uses it.


---

## References

Chen, J., Lerch, S., Schienle, M., Serafin, T. & Weron, R. (2025), *Probabilistic intraday electricity
price forecasting using generative machine learning*, arXiv:2506.00044. 

Shen, X. and Meinshausen, N. (2025), 'Engression: extrapolation through the
lens of distributional regression', *Journal of the Royal Statistical Society
Series B* 87(3), 653–677.