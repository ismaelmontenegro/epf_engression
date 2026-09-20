import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest
from scoringrules import twcrps_ensemble as twcrps
from scoringrules import twes_ensemble as twes
from tail_calibration import (tc_prob_ensemble, tail_cal_curves_pooled, plot_tail_cal_curves,
                                             plot_occ_by_horizon)
from scipy.stats import norm


# ============================================================
# Configuration
# ============================================================
from pathlib import Path
Y_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
CGM_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression")
ENG_EXP_DIR_1 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_False_HS_256")
ENG_EXP_DIR_1_E200 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E200_ENS10_HETERO_False_HS_256")
ENG_EXP_DIR_1_FULL_E200 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\full__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E200_ENS10_HETERO_False_HS_256")
ENG_EXP_DIR_2 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_False_HS_256")
CGM_HYBRID_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\hybrid_outputs")
ENG_HTS_DIR_E16 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_hts\engression_hts_corr_corr__embed16__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E200_ENS10")
ENG_EXP_DIR_2_HETERO_ALL_256 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
ENG_EXP_DIR_2_COMPLETE = Path(r"C:\Users\montenegrof\PycharmProjects\epf_cgm\engression_outputs_experiments\full__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10")
ENG_EXP_DIR_1_HETERO_ALL_256 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
ENG_DIR_NO_FE = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs")
N_EVAL_SAMPLES = 1000

MODEL_FILES = {
    "engression_experiment_1": ENG_EXP_DIR_1 / "pred.npy", #engression, fixed scale, raw target, engineered features, best configuration
    "engression_experiment_1_e200": ENG_EXP_DIR_1_E200 / "pred.npy", #engression, fixed scale, raw target, engineered features, 200 epochs (architecture search)
    "engression_experiment_1_full_e200": ENG_EXP_DIR_1_FULL_E200 / "pred.npy", #engression, fixed scale, raw target, engineered features, full history, 200 epochs (architecture search)
    "engression_experiment_2": ENG_EXP_DIR_2 / "pred.npy", #engression, fixed scale, engineered features, best configuration
    "engression_experiment_2_hetero_all_256": ENG_EXP_DIR_2_HETERO_ALL_256 / "pred.npy", #engression, conditional scale, engineered features, best configuration
    "engression_experiment_1_hetero_all_256": ENG_EXP_DIR_1_HETERO_ALL_256 / "pred.npy", #engression, conditional scale, raw target, engineered features, best configuration
    "engression_compact": ENG_DIR_NO_FE / "pred_engression_compact_e200.npy", #engression, fixed scale, cgm raw, reduced history, 200 epochs (architecture search)
    "engression_full": ENG_DIR_NO_FE / "pred_engression_full_e200.npy", #engression, fixed scale, cgm raw, full history, 200 epochs (architecture search)
    "engression_hts_16": ENG_HTS_DIR_E16 / "pred.npy", #engression, fixed scale, raw target, embedding from learned encoder as additional input, 200 epochs (architecture search)
    "cgm_hybrid": CGM_HYBRID_DIR / "pred_hybrid_esloss_2.npy", #cgm-engression hybrid
    "cgm_esloss": CGM_DIR / "pred_cgm_esloss.npy", # cgm with ES-loss
    "lasso_bootstrap": CGM_DIR / "lasso_bootstrap.npy", #standard lasso bootstrap
    "lasso_bootstrap_stationary": CGM_DIR / "lasso_bootstrap_stationary.npy" #lasso bootstrap without rolling window
}

Y_TEST_FILE = ENG_EXP_DIR_1_HETERO_ALL_256 / "y_test.npy"
OUT_DIR = CGM_DIR / "evaluation_outputs_calib_proper_old_cleaned_eval"
OUT_DIR.mkdir(exist_ok=True)


# For multivariate scores, exact pairwise sample terms with many samples can be heavy.
# Use the same fixed number of samples for all models for fairness.
N_SCORE_SAMPLES = 200

# Batch size for multivariate score computations
OBS_BATCH_SIZE = 32
# ============================================================
# PIT Calibration constants
# ============================================================
PIT_N_BINS = 20
# Nominal quantile levels used throughout PIT reliability computations.
PIT_QUANTILE_LEVELS = np.linspace(0.01, 0.99, 99)

# ============================================================
# Threshold-weighted proper scoring config
# ============================================================

TAIL_PCT = 0.10

TW_TAIL_PCT = 0.10                     # decile tails
TW_THRESHOLD_SOURCE = "train"          # "train" (preferred) | "test"
Y_TRAIN_FILE = Y_DIR / "y_train.npy"

def compute_tw_thresholds(y_source: np.ndarray, tail_pct: float = TW_TAIL_PCT):
    """Marginal (componentwise) price thresholds from flattened paths.
    Componentwise because the chaining function is applied per subperiod."""
    flat = np.asarray(y_source, dtype=float).reshape(-1)
    t_lo = float(np.quantile(flat, tail_pct))
    t_hi = float(np.quantile(flat, 1.0 - tail_pct))
    return t_lo, t_hi

def make_chaining_functions(t_lo: float, t_hi: float) -> dict:
    """Chaining functions v (v' = emphasis weight). Identity == ordinary ES/CRPS."""
    return {
        "unweighted": (lambda x: x),
        "upper_tail": (lambda x: np.maximum(x, t_hi)),                       # w = 1{x > t_hi}
        "lower_tail": (lambda x: np.minimum(x, t_lo)),                       # w = 1{x < t_lo}
        "both_tails": (lambda x: np.maximum(x, t_hi) + np.minimum(x, t_lo)), # w = 1{x>t_hi}+1{x<t_lo}
    }

def tw_energy_score(fct, y_true, v_func, batch_size=32):
    """
    Memory-safe threshold-weighted Energy Score.

    Same argument order as the original function:
        fct     : forecasts
        y_true  : observations
        v_func  : chaining/threshold function
    """

    total = 0.0
    n_total = 0

    for start in range(0, len(y_true), batch_size):
        end = min(start + batch_size, len(y_true))

        f_batch = fct[start:end]
        y_batch = y_true[start:end]

        scores = twes(
            y_batch,
            f_batch,
            v_func,
            m_axis=-1,
            v_axis=-2,
            estimator="nrg",
        )

        total += np.sum(scores)
        n_total += scores.size

    return float(total / n_total)
def tw_crps_by_horizon(pred, y_true, v_func, batch_size=32):
    """
    Memory-safe threshold-weighted CRPS by horizon.

    pred shape:   (n_obs, d, n_members)
    y_true shape: (n_obs, d)

    Returns:
        (d,) mean TWCRPS for each horizon
    """

    total = np.zeros(y_true.shape[1], dtype=np.float64)
    n_total = 0

    for start in range(0, len(y_true), batch_size):
        end = min(start + batch_size, len(y_true))

        pred_batch = pred[start:end]       # (batch, d, members)
        y_batch = y_true[start:end]        # (batch, d)

        per = twcrps(
            y_batch,
            pred_batch,
            m_axis=-1,
            v_func=v_func,
            estimator="nrg",
        )                                  # (batch, d)

        total += np.sum(per, axis=0)
        n_total += per.shape[0]

    return total / n_total

def _tc_cell(y, dat, t, tail, seed, n_rand):
    return {
        "occ":     tc_prob_ensemble(y, dat, t, "occ", tail=tail),
        "sev_sup": tc_prob_ensemble(y, dat, t, "sev", tail=tail, sup=True,
                                    rng=np.random.default_rng(seed), n_rand=n_rand),
        "com_sup": tc_prob_ensemble(y, dat, t, "com", tail=tail, sup=True,
                                    rng=np.random.default_rng(seed), n_rand=n_rand),
    }

def tail_cal_per_horizon(loaded_preds, y_true, y_train,
                         levels_upper=(0.90, 0.95, 0.99),
                         levels_lower=(0.10, 0.05, 0.01),
                         n_rand=10, seed=0):
    """Per-horizon probabilistic tail calibration. Thresholds = train marginal quantiles."""
    y_true = ensure_y_shape(y_true); y_train = ensure_y_shape(y_train)
    rows = []
    for model, pred in loaded_preds.items():
        for h in range(y_true.shape[1]):
            dat, y, ytr = pred[:, h, :], y_true[:, h], y_train[:, h]
            for tail, levels in (("upper", levels_upper), ("lower", levels_lower)):
                for lvl in levels:
                    t = float(np.quantile(ytr, lvl))
                    rows.append({"model": model, "horizon": h + 1, "tail": tail,
                                 "q": lvl, "t": t, **_tc_cell(y, dat, t, tail, seed, n_rand)})
    return pd.DataFrame(rows)

# ============================================================
# Scalar ID1 / ID2 / ID3 proxy evaluation settings
# ============================================================

# y_test columns = [id_3, id_4, ..., id_12]
SCALAR_SUBPERIOD_MINUTES = np.array(
    [15, 15, 15, 15, 15, 15, 15, 15, 15, 10],
    dtype=float,
)

SCALAR_TARGET_SPECS = {
    # Approx. last 1h before delivery: id_9..id_12
    "id1_proxy_time_weighted": np.array([0, 1, 2, 3], dtype=int),

    # Approx. last 2h before delivery: id_5..id_12
    "id2_proxy_time_weighted": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),

    # Full forecasted ID3 proxy: id_3..id_12
    "id3_proxy_time_weighted": np.arange(10, dtype=int),
}
START_INDEX_OBS = 7 * 24  # first 7 days removed by the CGM/Engression preprocessing

# ============================================================
# Utility functions
# ============================================================
def ensure_pred_shape(arr: np.ndarray) -> np.ndarray:
    """
    Force predictions into shape (n_obs, horizon=10, n_samples).
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D prediction array, got shape {arr.shape}")

    # expected already: (n_obs, 10, n_samples)
    if arr.shape[1] == 10:
        return arr

    # maybe (n_obs, n_samples, 10)
    if arr.shape[2] == 10:
        return np.transpose(arr, (0, 2, 1))

    raise ValueError(f"Cannot infer forecast axis order from shape {arr.shape}")


def ensure_y_shape(arr: np.ndarray) -> np.ndarray:
    """
    Force y into shape (n_obs, 10).
    """
    if arr.ndim == 2 and arr.shape[1] == 10:
        return arr
    if arr.ndim == 3 and arr.shape[1] == 10 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 10:
        return arr[:, 0, :]
    raise ValueError(f"Cannot infer y shape from {arr.shape}")

def build_stress_masks(y_true: np.ndarray, tail_pct: float = 0.10):
    """
    Stress = either tail (very high or very negative prices).
    price_level: mean across subperiods to capture the regime of the whole market.
    """
    price_level = y_true.mean(axis=1)

    high_thresh = np.quantile(price_level, 1 - tail_pct)
    low_thresh  = np.quantile(price_level, tail_pct)

    masks = {
        "all":           np.ones(len(y_true), dtype=bool),
        "upper_stress":  price_level >= high_thresh,
        "lower_stress":  price_level <= low_thresh,
        "both_tails":    (price_level >= high_thresh) | (price_level <= low_thresh),
        "normal":        (price_level > low_thresh) & (price_level < high_thresh),
    }
    return masks


def maybe_subsample_samples(pred: np.ndarray, n_samples: int, seed: int = 123) -> np.ndarray:
    """
    Subsample along sample axis if needed.
    pred shape: (n_obs, horizon, n_samples_total)
    """
    n_total = pred.shape[2]
    if n_total <= n_samples:
        return pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total, size=n_samples, replace=False)
    idx.sort()
    return pred[:, :, idx]


def naive_ensemble(point_forecast, last_p_scalar, w=0.5):
    """
    Marcjasz-style point ensemble: blend a model's POINT forecast with the naive (last_p).
    Operates ONLY on the point estimate; the predictive distribution is never modified.

    point_forecast : (n_obs,)  the model's point (mean OR median of its samples)
    last_p_scalar  : (n_obs,)  the naive forecast == last_p, aligned per observation
    w              : weight on the naive (paper uses 0.5; sweep 0..1 for their Fig 6)
    """
    point_forecast = np.asarray(point_forecast, dtype=float)
    last_p = np.asarray(last_p_scalar, dtype=float)
    return (1.0 - w) * point_forecast + w * last_p

# ============================================================
# Marginal metrics
# ============================================================
def median_mae_by_horizon(pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    MAE of the sample median forecast at each horizon.
    Returns shape (10,)
    """
    median_fcst = np.median(pred, axis=2)
    return np.mean(np.abs(median_fcst - y_true), axis=0)


def crps_from_samples_1d(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Efficient CRPS for empirical samples.
    samples shape: (n_obs, n_samples)
    y shape: (n_obs,)
    returns shape: (n_obs,)
    Formula:
      CRPS = mean |X - y| - (1 / (2 M^2)) sum_{i,j} |X_i - X_j|
    The pairwise term is computed efficiently from sorted samples.
    """
    n_obs, m = samples.shape

    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)

    xs = np.sort(samples, axis=1)
    coeff = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)  # shape (m,)
    # second term = 1/(2m^2) * sum_{i,j}|xi-xj|
    term2 = np.sum(xs * coeff[None, :], axis=1) / (m ** 2)

    return term1 - term2


# ============================================================
# Scalar ID1 / ID2 / ID3 proxy metrics
# ============================================================

def get_scalar_target_weights(idxs: np.ndarray) -> np.ndarray:
    """
    Time weights renormalized over selected target subperiods.
    """
    weights = SCALAR_SUBPERIOD_MINUTES[idxs].astype(float)
    return weights / weights.sum()


def collapse_pred_to_scalar_target(pred: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Collapse path forecast samples to scalar IDx proxy samples.

    Input:
        pred shape: (n_obs, 10, n_samples)

    Output:
        scalar samples shape: (n_obs, n_samples)
    """
    pred = ensure_pred_shape(pred)
    weights = get_scalar_target_weights(idxs)
    return np.average(pred[:, idxs, :], axis=1, weights=weights)


def collapse_y_to_scalar_target(y_true_path: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Collapse realized path to scalar IDx proxy.

    Input:
        y_true_path shape: (n_obs, 10)

    Output:
        y_scalar shape: (n_obs,)
    """
    y_true_path = ensure_y_shape(y_true_path)
    weights = get_scalar_target_weights(idxs)
    return np.average(y_true_path[:, idxs], axis=1, weights=weights)


def pinball_loss(samples: np.ndarray, y: np.ndarray, q: float) -> float:
    """
    Pinball loss for empirical q-quantile forecast.
    """
    q_pred = np.quantile(samples, q, axis=1)
    err = y - q_pred
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.mean(loss))


def evaluate_scalar_prob_forecast(samples: np.ndarray, y: np.ndarray) -> dict:
    """
    Scalar probabilistic forecast evaluation.

    samples shape: (n_obs, n_samples)
    y shape: (n_obs,)
    """
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(y) == 0:
        return {
            "mae_median": np.nan,
            "rmse_median": np.nan,
            "bias_median": np.nan,
            "mae_mean": np.nan,
            "rmse_mean": np.nan,
            "bias_mean": np.nan,
            "crps": np.nan,
            "coverage_80": np.nan,
            "width_80": np.nan,
            "coverage_60": np.nan,
            "width_60": np.nan,
            "pinball_q10": np.nan,
            "pinball_q50": np.nan,
            "pinball_q90": np.nan,
        }

    median = np.median(samples, axis=1)
    mean = np.mean(samples, axis=1)

    err_median = median - y
    err_mean = mean - y

    crps_obs = crps_from_samples_1d(samples, y)

    q10 = np.quantile(samples, 0.10, axis=1)
    q20 = np.quantile(samples, 0.20, axis=1)
    q80 = np.quantile(samples, 0.80, axis=1)
    q90 = np.quantile(samples, 0.90, axis=1)

    return {
        "mae_median": float(np.mean(np.abs(err_median))),
        "rmse_median": float(np.sqrt(np.mean(err_median ** 2))),
        "bias_median": float(np.mean(err_median)),

        "mae_mean": float(np.mean(np.abs(err_mean))),
        "rmse_mean": float(np.sqrt(np.mean(err_mean ** 2))),
        "bias_mean": float(np.mean(err_mean)),

        "crps": float(np.mean(crps_obs)),

        "coverage_80": float(np.mean((y >= q10) & (y <= q90))),
        "width_80": float(np.mean(q90 - q10)),

        "coverage_60": float(np.mean((y >= q20) & (y <= q80))),
        "width_60": float(np.mean(q80 - q20)),

        "pinball_q10": pinball_loss(samples, y, 0.10),
        "pinball_q50": pinball_loss(samples, y, 0.50),
        "pinball_q90": pinball_loss(samples, y, 0.90),
    }


def build_stress_masks_scalar(y_scalar: np.ndarray, tail_pct: float = TAIL_PCT) -> dict:
    """
    Scalar stress masks based on the evaluated scalar target itself.
    """
    y_scalar = np.asarray(y_scalar, dtype=float)

    low_thresh = np.quantile(y_scalar, tail_pct)
    high_thresh = np.quantile(y_scalar, 1.0 - tail_pct)

    return {
        "all": np.ones(len(y_scalar), dtype=bool),
        "lower_stress": y_scalar <= low_thresh,
        "upper_stress": y_scalar >= high_thresh,
        "both_tails": (y_scalar <= low_thresh) | (y_scalar >= high_thresh),
        "normal": (y_scalar > low_thresh) & (y_scalar < high_thresh),
    }


def evaluate_scalar_targets_for_forecasts(
    forecast_sets: dict,
    y_true_path: np.ndarray,
    last_p_test: np.ndarray | None = None,
    target_specs: dict = SCALAR_TARGET_SPECS,
    threshold_scope: str = "test",
    ens_weight: float = 0.5,
) -> pd.DataFrame:
    rows = []
    y_true_path = ensure_y_shape(y_true_path)
    ens_tag = f"__ens_w{int(round(ens_weight * 100)):02d}"

    for target_name, idxs in target_specs.items():
        y_scalar = collapse_y_to_scalar_target(y_true_path, idxs)
        masks = build_stress_masks_scalar(y_scalar, tail_pct=TAIL_PCT)

        for model_name, pred_path in forecast_sets.items():
            pred_path = ensure_pred_shape(pred_path)
            if pred_path.shape[0] != y_true_path.shape[0]:
                raise ValueError(
                    f"{model_name} / {target_name}: n_obs mismatch. "
                    f"pred={pred_path.shape[0]}, y={y_true_path.shape[0]}"
                )
            scalar_samples = collapse_pred_to_scalar_target(pred_path, idxs)

            for period_name, mask in masks.items():
                rows.append({
                    "target": target_name, "model": model_name,
                    "period": period_name, "threshold_scope": threshold_scope,
                    "n_obs": int(mask.sum()),
                    **evaluate_scalar_prob_forecast(scalar_samples[mask], y_scalar[mask]),
                })

            # Point-only naive ensemble (full set), once per model/target
            if last_p_test is not None and model_name != "naive_same_contract_last_p":
                e_mean = naive_ensemble(scalar_samples.mean(axis=1), last_p_test, ens_weight) - y_scalar
                e_med = naive_ensemble(np.median(scalar_samples, axis=1), last_p_test, ens_weight) - y_scalar
                rows.append({
                    "target": target_name, "model": model_name + ens_tag,
                    "period": "all", "threshold_scope": threshold_scope,
                    "n_obs": int(len(y_scalar)),
                    "mae_mean": float(np.mean(np.abs(e_mean))),
                    "rmse_mean": float(np.sqrt(np.mean(e_mean ** 2))),
                    "bias_mean": float(np.mean(e_mean)),
                    "mae_median": float(np.mean(np.abs(e_med))),
                    "rmse_median": float(np.sqrt(np.mean(e_med ** 2))),
                    "bias_median": float(np.mean(e_med)),
                })

    return pd.DataFrame(rows)


# ============================================================
# Multivariate scores
# ============================================================

def dawid_sebastiani_score(pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-6, obs_batch_size: int = 128) -> float:
    """
    DSS = log det(Sigma) + (y - mu)^T Sigma^{-1} (y - mu)
    pred shape: (n_obs, d, m)
    """
    n_obs, d, m = pred.shape
    out = []

    for start in range(0, n_obs, obs_batch_size):
        stop = min(start + obs_batch_size, n_obs)
        xb = pred[start:stop]   # (b, d, m)
        yb = y_true[start:stop] # (b, d)

        mu = xb.mean(axis=2)    # (b, d)
        xc = xb - mu[:, :, None]

        cov = np.einsum("bdm,bem->bde", xc, xc) / max(m - 1, 1)
        cov = cov + eps * np.eye(d)[None, :, :]

        diff = (yb - mu)[:, :, None]  # (b, d, 1)

        sign, logdet = np.linalg.slogdet(cov)
        # if something numerically goes wrong, penalize heavily
        bad = sign <= 0
        logdet[bad] = 1e6

        inv_cov_diff = np.linalg.solve(cov, diff)      # (b, d, 1)
        quad = np.matmul(np.transpose(diff, (0, 2, 1)), inv_cov_diff).reshape(-1)

        dss = logdet + quad
        out.append(dss)

    return float(np.mean(np.concatenate(out)))


def variogram_score(pred: np.ndarray, y_true: np.ndarray, p: float = 1.0) -> float:
    """
    Unweighted variogram score with weight 1/d^2:
      VS = sum_{i,j} w_ij ( |y_i-y_j|^p - mean_m |x_i^m-x_j^m|^p )^2
    pred shape: (n_obs, d, m)
    """
    n_obs, d, m = pred.shape
    w = 1.0 / (d ** 2)

    # true pairwise differences: (n_obs, d, d)
    ydiff = np.abs(y_true[:, :, None] - y_true[:, None, :]) ** p

    # forecast expected pairwise differences: mean over sample axis
    # pred: (n_obs, d, m)
    xdiff = np.abs(pred[:, :, None, :] - pred[:, None, :, :]) ** p  # (n_obs, d, d, m)
    xdiff_mean = xdiff.mean(axis=3)

    vs = w * np.sum((ydiff - xdiff_mean) ** 2, axis=(1, 2))
    return float(np.mean(vs))


# ============================================================
# PIT Calibration
# ============================================================

def compute_pit_values(pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Compute marginal Probability Integral Transform (PIT) values.

    For each (obs i, horizon h):
        PIT[i, h] = (1 / M) * sum_m  I(pred[i, h, m] <= y_true[i, h])

    Under a calibrated forecast the PIT ~ Uniform[0, 1] marginally at
    every horizon. Note this is a marginal check and does not capture
    cross-horizon dependence failures.

    pred   shape : (n_obs, 10, n_samples)
    y_true shape : (n_obs, 10)
    Returns      : (n_obs, 10)  — values in [0, 1]
    """
    return np.mean(pred <= y_true[:, :, np.newaxis], axis=2)


def pit_calibration_metrics(
    pit_values: np.ndarray,
    quantile_levels: np.ndarray = None,
) -> dict:
    """
    Summary PIT calibration metrics (marginal, pooled over all obs × horizons).

    Args:
        pit_values     : shape (n_obs, horizon) or (n,) — values in [0, 1]
        quantile_levels: nominal levels for reliability curve;
                         defaults to PIT_QUANTILE_LEVELS (0.05..0.95 step 0.05)

    Returns dict with keys:
        pit_mean      : mean of pooled PIT values  (ideal: 0.500)
        pit_std       : std of pooled PIT values   (ideal: 0.289 = 1/sqrt(12))
        ks_stat       : Kolmogorov-Smirnov statistic vs Uniform[0, 1]
        ks_pval       : two-sided KS p-value  (small = miscalibrated)
        mace          : Mean Absolute Calibration Error = mean |empirical_q - q|
        ace           : Average Calibration Error (signed; positive = overconfident)
        coverage_at_q : empirical coverage at each quantile level (array)
        quantile_levels: quantile levels used (array)
    """
    if quantile_levels is None:
        quantile_levels = PIT_QUANTILE_LEVELS

    flat = np.asarray(pit_values, dtype=float).ravel()

    empirical = np.array([np.mean(flat <= q) for q in quantile_levels])
    cal_error = empirical - quantile_levels

    ks_stat, ks_pval = kstest(flat, "uniform")

    return {
        "pit_mean": float(np.mean(flat)),
        "pit_std": float(np.std(flat)),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "mace": float(np.mean(np.abs(cal_error))),
        "ace": float(np.mean(cal_error)),
        "coverage_at_q": empirical,
        "quantile_levels": quantile_levels,
    }


def pit_metrics_by_horizon(pit_values: np.ndarray) -> dict:
    """
    Per-horizon PIT metrics.

    pit_values shape: (n_obs, horizon)

    Returns dict with arrays of shape (horizon,):
        mean    : mean PIT per horizon   (ideal: 0.5 everywhere)
        std     : std PIT per horizon    (ideal: 0.289 everywhere)
        ks_stat : KS statistic per horizon
    """
    pit_values = np.asarray(pit_values, dtype=float)
    horizon = pit_values.shape[1]
    mean_h = np.empty(horizon)
    std_h = np.empty(horizon)
    ks_h = np.empty(horizon)

    for h in range(horizon):
        v = pit_values[:, h]
        mean_h[h] = v.mean()
        std_h[h] = v.std()
        ks_h[h], _ = kstest(v, "uniform")

    return {"mean": mean_h, "std": std_h, "ks_stat": ks_h}

def pit_histograms_by_horizon(
    pit_dict: dict,
    n_bins: int = PIT_N_BINS,
) -> pd.DataFrame:
    """
    Per-horizon PIT histogram bin data for every model, in tidy long form.

    Under a calibrated forecast each horizon's PIT ~ Uniform[0, 1], so the
    density should sit at ~1.0 in every bin. Returned frame reconstructs the
    plotted histograms exactly and is CSV-ready.

    Columns:
        model, horizon, bin_index, bin_left, bin_right, bin_center,
        count, density, uniform_density, density_dev
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    uniform_density = 1.0  # density of Uniform[0, 1]

    rows = []
    for model_name, pit in pit_dict.items():
        pit = np.asarray(pit, dtype=float)
        horizon = pit.shape[1]
        for h in range(horizon):
            v = pit[:, h]
            counts, _ = np.histogram(v, bins=edges)
            density, _ = np.histogram(v, bins=edges, density=True)
            for b in range(n_bins):
                rows.append({
                    "model": model_name,
                    "horizon": h + 1,
                    "bin_index": b + 1,
                    "bin_left": float(edges[b]),
                    "bin_right": float(edges[b + 1]),
                    "bin_center": float(centers[b]),
                    "count": int(counts[b]),
                    "density": float(density[b]),
                    "uniform_density": uniform_density,
                    "density_dev": float(density[b] - uniform_density),
                })
    return pd.DataFrame(rows)

def plot_pit_diagnostics(
    pit_dict: dict,
    out_dir: Path,
    mask_label: str = "all",
    n_bins: int = PIT_N_BINS,
    quantile_levels: np.ndarray = None,
):
    """
    Produce three batches of PIT diagnostic figures for a given stress regime.

    (a) Per-model 3-panel figure
        [1] Marginal PIT histogram vs Uniform reference
        [2] Reliability diagram (empirical vs nominal coverage)
        [3] Mean PIT ± 1 std by horizon

    (b) All-models reliability overlay — one figure, all models on same axes

    (c) All-models PIT histogram grid

    Args:
        pit_dict      : {model_name: pit_array (n_obs, 10)}
        out_dir       : output directory (created if absent)
        mask_label    : regime label used in titles / filenames
        n_bins        : number of histogram bins
        quantile_levels: nominal quantile levels for reliability
    """
    if quantile_levels is None:
        quantile_levels = PIT_QUANTILE_LEVELS

    out_dir.mkdir(exist_ok=True, parents=True)

    # ----------------------------------------------------------------
    # (a) per-model 3-panel figure
    # ----------------------------------------------------------------
    for model_name, pit in pit_dict.items():
        flat = pit.ravel()
        m = pit_calibration_metrics(flat, quantile_levels)
        by_h = pit_metrics_by_horizon(pit)
        x_h = np.arange(1, pit.shape[1] + 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(
            f"{model_name}  |  PIT [{mask_label}]   "
            f"KS={m['ks_stat']:.3f}  MACE={m['mace']:.4f}  "
            f"mean={m['pit_mean']:.3f}  std={m['pit_std']:.3f}",
            fontsize=10,
        )

        # [1] PIT histogram
        ax = axes[0]
        ax.hist(
            flat, bins=n_bins, range=(0, 1), density=True,
            color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.80,
        )
        ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="Uniform")
        ax.set_xlabel("PIT value")
        ax.set_ylabel("Density")
        ax.set_title("Marginal PIT histogram")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

        # [2] Reliability diagram
        ax = axes[1]
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
        ax.fill_between(
            quantile_levels,
            np.clip(quantile_levels - 0.05, 0, 1),
            np.clip(quantile_levels + 0.05, 0, 1),
            alpha=0.12, color="gray", label="±5% tolerance",
        )
        ax.plot(
            quantile_levels, m["coverage_at_q"],
            marker="o", markersize=4, linewidth=1.5, color="steelblue",
            label=f"Empirical  MACE={m['mace']:.4f}",
        )
        # shade over/under-coverage regions
        ax.fill_between(
            quantile_levels,
            quantile_levels, m["coverage_at_q"],
            where=m["coverage_at_q"] > quantile_levels,
            alpha=0.10, color="red", label="Over-coverage",
        )
        ax.fill_between(
            quantile_levels,
            quantile_levels, m["coverage_at_q"],
            where=m["coverage_at_q"] < quantile_levels,
            alpha=0.10, color="blue", label="Under-coverage",
        )
        ax.set_xlabel("Nominal quantile")
        ax.set_ylabel("Empirical coverage")
        ax.set_title("Reliability diagram")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

        # [3] Mean PIT by horizon
        ax = axes[2]
        # uniform-std reference band: ± (1/sqrt(12)) / sqrt(n_obs) roughly; show ±1 marginal std
        ax.axhline(0.5, color="red", linewidth=1.2, linestyle="--", label="Ideal (0.5)")
        ax.plot(x_h, by_h["mean"], marker="o", linewidth=1.8, color="steelblue", label="Mean PIT")
        ax.fill_between(
            x_h,
            np.clip(by_h["mean"] - by_h["std"], 0, 1),
            np.clip(by_h["mean"] + by_h["std"], 0, 1),
            alpha=0.15, color="steelblue", label="±1 std",
        )
        # also show KS stat on secondary axis
        ax2 = ax.twinx()
        ax2.bar(x_h, by_h["ks_stat"], alpha=0.25, color="orange", width=0.4, label="KS stat")
        ax2.set_ylabel("KS stat", fontsize=8, color="orange")
        ax2.tick_params(axis="y", labelcolor="orange", labelsize=7)
        ax2.set_ylim(0, max(by_h["ks_stat"].max() * 2, 0.1))

        ax.set_xlabel("Horizon")
        ax.set_ylabel("Mean PIT")
        ax.set_title("Mean PIT & KS by horizon")
        ax.set_xlim(0.5, pit.shape[1] + 0.5)
        ax.set_ylim(0, 1)
        ax.set_xticks(x_h)

        # merge legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

        plt.tight_layout()
        plt.savefig(
            out_dir / f"pit_{model_name}_{mask_label}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # ----------------------------------------------------------------
    # (b) combined reliability diagram — all models on one plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Ideal")
    ax.fill_between(
        quantile_levels,
        np.clip(quantile_levels - 0.05, 0, 1),
        np.clip(quantile_levels + 0.05, 0, 1),
        alpha=0.10, color="gray", label="±5% tolerance",
    )
    for model_name, pit in pit_dict.items():
        flat = pit.ravel()
        emp = np.array([np.mean(flat <= q) for q in quantile_levels])
        mace = float(np.mean(np.abs(emp - quantile_levels)))
        ax.plot(
            quantile_levels, emp,
            marker="o", markersize=3, linewidth=1.5,
            label=f"{model_name}  (MACE={mace:.4f})",
        )
    ax.set_xlabel("Nominal quantile")
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Reliability diagram — all models [{mask_label}]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"pit_reliability_all_{mask_label}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # ----------------------------------------------------------------
    # (c) PIT histogram grid — all models, one subplot each
    # ----------------------------------------------------------------
    n_models = len(pit_dict)
    if n_models == 0:
        return
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for idx, (model_name, pit) in enumerate(pit_dict.items()):
        ax = axes_flat[idx]
        flat = pit.ravel()
        ax.hist(
            flat, bins=n_bins, range=(0, 1), density=True,
            color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.80,
        )
        ax.axhline(1.0, color="red", linewidth=1.5, linestyle="--", label="Uniform")
        m = pit_calibration_metrics(flat, quantile_levels)
        ax.set_title(
            f"{model_name}\nKS={m['ks_stat']:.3f}  MACE={m['mace']:.4f}"
            f"  mean={m['pit_mean']:.3f}  std={m['pit_std']:.3f}",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle(f"PIT histograms — all models [{mask_label}]", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        out_dir / f"pit_histograms_all_{mask_label}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close()

def plot_pit_histograms_by_horizon(
    pit_dict: dict,
    out_dir: Path,
    mask_label: str = "all",
    n_bins: int = PIT_N_BINS,
):
    """
    One figure per model: a grid of PIT histograms, one panel per horizon,
    each against the Uniform[0, 1] reference. Complements the pooled
    histogram grid in plot_pit_diagnostics, which collapses all horizons.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    for model_name, pit in pit_dict.items():
        pit = np.asarray(pit, dtype=float)
        horizon = pit.shape[1]
        ncols = min(5, horizon)
        nrows = int(np.ceil(horizon / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False
        )
        axes_flat = axes.ravel()

        for h in range(horizon):
            ax = axes_flat[h]
            v = pit[:, h]
            ax.hist(
                v, bins=n_bins, range=(0, 1), density=True,
                color="steelblue", edgecolor="white", linewidth=0.4, alpha=0.80,
            )
            ax.axhline(1.0, color="red", linewidth=1.2, linestyle="--")
            ks, _ = kstest(v, "uniform")
            ax.set_title(
                f"Horizon {h + 1}  KS={ks:.3f}  mean={v.mean():.3f}",
                fontsize=8,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("PIT", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)

        for idx in range(horizon, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(
            f"{model_name} | per-horizon PIT histograms [{mask_label}]",
            fontsize=11,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(
            out_dir / f"pit_histograms_by_horizon_{model_name}_{mask_label}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()



# ============================================================
# Trading metrics
# ============================================================
def mode_argmax(pred: np.ndarray) -> np.ndarray:
    """
    Majority-vote strategy:
    choose the most frequent argmax horizon across forecast paths.
    pred shape: (n_obs, 10, n_samples)
    returns indices in 0..9
    """
    argmaxes = np.argmax(pred, axis=1)  # (n_obs, n_samples)

    out = np.zeros(argmaxes.shape[0], dtype=int)
    for i in range(argmaxes.shape[0]):
        counts = np.bincount(argmaxes[i], minlength=pred.shape[1])
        out[i] = np.argmax(counts)
    return out


def profit_from_chosen_horizon(y_true: np.ndarray, chosen_idx: np.ndarray) -> float:
    """
    Assumes selling 1 MWh at chosen subperiod price and sums over all observations.
    """
    return float(np.sum(y_true[np.arange(len(y_true)), chosen_idx]))


def naive_strategy_profit(y_true: np.ndarray, which: str) -> float:
    """
    which in {"first", "last", "avg"}
    """
    if which == "last":
        return float(np.sum(y_true[:, 0]))
    if which == "first":
        return float(np.sum(y_true[:, -1]))
    if which == "avg":
        return float(np.sum(y_true.mean(axis=1)))
    raise ValueError("which must be 'first', 'last', or 'avg'")


def crystal_ball_bounds(y_true: np.ndarray):
    """
    Returns (cb_min, cb_max)
    """
    cb_max = float(np.sum(np.max(y_true, axis=1)))
    cb_min = float(np.sum(np.min(y_true, axis=1)))
    return cb_min, cb_max


def realized_trading_potential(profit: float, cb_min: float, cb_max: float) -> float:
    return 100.0 * (profit - cb_min) / (cb_max - cb_min)


def _resolve_hourly_file(folder: Path, prefix: str, hour: int) -> Path:
    """
    Find hourly raw ID file, allowing both extensionless and .csv filenames.
    """
    candidates = [
        folder / f"{prefix}_{hour:02d}",
        folder / f"{prefix}_{hour:02d}.csv",
    ]

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Could not find hourly file for hour={hour:02d}. "
        f"Tried: {candidates}"
    )


def read_last_p_full_from_raw_id_data(data_dir: Path) -> np.ndarray:
    """
    Reconstruct same-contract last available VWAP from raw ID price files.

    Expected raw files:
        data_dir / ID_DATA / prices_hourly_00
        ...
        data_dir / ID_DATA / prices_hourly_23

    The price files are assumed to contain 13 price columns after the timestamp:
        id_1, ..., id_12, last_p

    Returns:
        last_p_full shape: (n_days * 24,)
        ordered as day-major, hour-minor:
        day1 h0, day1 h1, ..., day1 h23, day2 h0, ...
    """
    folder = data_dir / "ID_DATA"

    per_hour_last_p = []

    for h in range(24):
        fpath = _resolve_hourly_file(folder, "prices_hourly", h)

        with open(fpath) as f:
            lines = f.readlines()

        # skip header
        vals = []
        for line in lines[1:]:
            parts = line.strip().split(",")
            nums = [float(x) for x in parts[1:]]

            if len(nums) < 13:
                raise ValueError(
                    f"{fpath} has fewer than 13 numeric price columns. "
                    f"Found {len(nums)} columns."
                )

            # 13th kept price column = last_p
            vals.append(nums[12])

        per_hour_last_p.append(np.asarray(vals, dtype=float))

    n_days_set = {len(x) for x in per_hour_last_p}
    if len(n_days_set) != 1:
        raise ValueError(
            f"Hourly price files have inconsistent day counts: {n_days_set}"
        )

    n_days = per_hour_last_p[0].shape[0]

    # shape (n_days, 24)
    last_p_panel = np.zeros((n_days, 24), dtype=float)
    for h in range(24):
        last_p_panel[:, h] = per_hour_last_p[h]

    return last_p_panel.reshape(-1)


def build_same_contract_last_p_test_baseline(
    data_dir: Path,
    n_test_obs: int,
    start_index_obs: int = START_INDEX_OBS,
) -> np.ndarray:
    """
    Build same-contract last_p baseline aligned to the test period.

    This assumes the evaluated test set is the final n_test_obs observations
    after removing the first 7 days, which matches your current 837-day setup.
    """
    last_p_full = read_last_p_full_from_raw_id_data(data_dir)

    if len(last_p_full) <= start_index_obs:
        raise ValueError(
            f"Raw last_p series too short: {len(last_p_full)} observations, "
            f"START_INDEX_OBS={start_index_obs}"
        )

    last_p_trimmed = last_p_full[start_index_obs:]

    if len(last_p_trimmed) < n_test_obs:
        raise ValueError(
            f"Trimmed last_p series has only {len(last_p_trimmed)} observations, "
            f"but y_true has {n_test_obs}."
        )

    # test period = final block
    return last_p_trimmed[-n_test_obs:]


def build_constant_path_samples_from_point(
    point_forecast: np.ndarray,
    horizon: int = 10,
    n_samples: int = N_EVAL_SAMPLES,
) -> np.ndarray:
    """
    Convert scalar point forecast into a degenerate path forecast.

    Output:
        shape (n_obs, horizon, n_samples)

    For scalar ID1/ID2/ID3 collapse, every target proxy becomes equal
    to the same-contract last_p baseline.
    """
    point_forecast = np.asarray(point_forecast, dtype=float)

    if point_forecast.ndim != 1:
        raise ValueError(f"Expected 1D point forecast, got {point_forecast.shape}")

    return np.repeat(
        point_forecast[:, None, None],
        repeats=horizon,
        axis=1,
    ).repeat(n_samples, axis=2)

def plot_marginal_distribution_overlay(
    loaded_preds: dict,
    y_true: np.ndarray,
    out_dir: Path,
    mask: np.ndarray = None,
    mask_label: str = "all",
    n_bins: int = 60,
    clip_quantiles: tuple = (0.005, 0.995),
):
    """
    Pooled marginal distribution check: per subperiod, overlay each model's
    pooled generated samples against the empirical histogram of realized
    prices. This is the literal "learned distribution vs empirical
    distribution" comparison -- complementary to, not a replacement for,
    the PIT/tail-calibration diagnostics above.

    Unlike PIT, this collapses the conditioning information (x) entirely,
    so a model can match the pooled marginal well while still being badly
    calibrated point-by-point. Use alongside the PIT diagnostics, not
    instead of them.

    loaded_preds: {model_name: pred array (n_obs, 10, n_samples)}
    y_true      : (n_obs, 10)
    mask        : optional boolean mask (n_obs,) -- pass a stress-regime
                  mask from build_stress_masks() to see if the mismatch
                  is regime-specific.
    clip_quantiles: x-axis range per subperiod is clipped to these
                  quantiles of the empirical data so extreme spikes
                  (already covered by your tail-calibration work) don't
                  compress the bulk of the histogram.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    if mask is None:
        mask = np.ones(len(y_true), dtype=bool)

    horizon = y_true.shape[1]
    ncols = min(5, horizon)
    nrows = int(np.ceil(horizon / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    handles, labels = None, None
    for h in range(horizon):
        ax = axes_flat[h]
        y_h = y_true[mask, h]
        lo, hi = np.quantile(y_h, clip_quantiles)

        ax.hist(y_h, bins=n_bins, range=(lo, hi), density=True,
                 color="black", alpha=0.30, label="empirical")
        for model_name, pred in loaded_preds.items():
            samples_h = pred[mask, h, :].ravel()
            ax.hist(samples_h, bins=n_bins, range=(lo, hi), density=True,
                     histtype="step", linewidth=1.4, label=model_name)

        ax.set_title(f"Horizon {h + 1}", fontsize=9)
        ax.set_xlabel("price")
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    for idx in range(horizon, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(f"Pooled marginal distribution — all models [{mask_label}]", fontsize=12)
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_dir / f"marginal_overlay_{mask_label}.png", dpi=150, bbox_inches="tight")
    plt.close()

def dispersion_ratio_by_horizon(loaded_preds, y_true, mask):
    """
    std(pooled model samples) / std(empirical) per subperiod, within mask.
    ratio < 1  -> model underdispersed relative to truth
    ratio > 1  -> model overdispersed relative to truth
    This replaces eyeballing peak heights with an actual number.
    """
    horizon = y_true.shape[1]
    rows = []
    for h in range(horizon):
        emp_std = y_true[mask, h].std()
        for model_name, pred in loaded_preds.items():
            model_std = pred[mask, h, :].std()
            rows.append({
                "horizon": h + 1,
                "model": model_name,
                "emp_std": emp_std,
                "model_std": model_std,
                "dispersion_ratio": model_std / emp_std,
            })
    return pd.DataFrame(rows)


# ======================================================================
# 1. Per-origin loss vectors
# ======================================================================

def energy_score_per_obs(pred, y_true, obs_batch_size=32, fair=True):
    """
    Per-forecast-origin energy score.  Returns (n_obs,).

    fair=True uses the unbiased 1/(m(m-1)) normalisation for the spread
    term, matching energy_loss() in loss_func.py.  fair=False reproduces
    the 1/m^2 ("nrg") convention currently used by energy_score() in
    eval_proper_old.py.  The biased version inflates ES by
    E||X-X'||/(2m), an amount proportional to each model's own spread,
    so it penalises wider forecasts more than sharper ones.
    """
    n_obs, d, m = pred.shape
    pred_t = np.transpose(pred, (0, 2, 1))  # (n_obs, m, d)
    out = []

    for start in range(0, n_obs, obs_batch_size):
        stop = min(start + obs_batch_size, n_obs)
        xb = pred_t[start:stop]  # (b, m, d)
        yb = y_true[start:stop]  # (b, d)

        term1 = np.linalg.norm(xb - yb[:, None, :], axis=2).mean(axis=1)

        diff = xb[:, :, None, :] - xb[:, None, :, :]
        pdist = np.linalg.norm(diff, axis=3)  # (b, m, m)
        if fair:
            # sum over off-diagonal pairs / (m(m-1)); diagonal is zero
            spread = pdist.sum(axis=(1, 2)) / (m * (m - 1))
        else:
            spread = pdist.mean(axis=(1, 2))
        out.append(term1 - 0.5 * spread)

    return np.concatenate(out)


def crps_per_obs(pred, y_true, fair=True):
    """
    Per-origin CRPS, averaged over the ten horizons.  Returns (n_obs,).
    """
    n_obs, d, m = pred.shape
    total = np.zeros(n_obs)
    for h in range(d):
        s = pred[:, h, :]
        y = y_true[:, h]
        term1 = np.mean(np.abs(s - y[:, None]), axis=1)
        xs = np.sort(s, axis=1)
        coeff = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
        pair_sum = np.sum(xs * coeff[None, :], axis=1)  # = sum_{i,j}|xi-xj| / 2
        denom = m * (m - 1) if fair else m ** 2
        total += term1 - pair_sum / denom
    return total / d


def variogram_per_obs(pred, y_true, p=1.0):
    """
    Per-origin variogram score with weights 1/d^2.  Returns (n_obs,).
    """
    n_obs, d, m = pred.shape
    w = 1.0 / (d ** 2)
    ydiff = np.abs(y_true[:, :, None] - y_true[:, None, :]) ** p
    xdiff = np.abs(pred[:, :, None, :] - pred[:, None, :, :]) ** p
    xmean = xdiff.mean(axis=3)
    return w * np.sum((ydiff - xmean) ** 2, axis=(1, 2))


# ======================================================================
# 2. Diebold-Mariano with HAC standard errors
# ======================================================================

def _newey_west_var(d, lag):
    """Long-run variance of d with Bartlett weights."""
    n = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    acc = gamma0
    for j in range(1, lag + 1):
        gj = np.dot(dc[j:], dc[:-j]) / n
        acc += 2.0 * (1.0 - j / (lag + 1.0)) * gj
    return acc


def dm_test(loss_a, loss_b, lag=24):
    """
    Two-sided Diebold-Mariano test on the loss differential
    d_t = loss_a - loss_b, with Newey-West HAC variance.

    Negative statistic  => model A has the lower (better) loss.
    lag=24 covers one day of serial dependence across hourly products;
    check sensitivity to lag in {0, 12, 24, 48}.
    """
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    n = len(d)
    mean_d = d.mean()
    lrv = _newey_west_var(d, lag)
    se = np.sqrt(max(lrv, 1e-300) / n)
    stat = mean_d / se
    return {
        "mean_diff": float(mean_d),
        "se": float(se),
        "stat": float(stat),
        "p_value": float(2 * norm.sf(abs(stat))),
        "n": int(n),
        "lag": int(lag),
    }


def dm_matrix(losses_by_model, lag=24):
    """
    losses_by_model: {model_name: (n_obs,) loss vector}
    Returns (stat_df, p_df); entry [A, B] tests A against B, so a
    negative statistic means row A beats column B.
    """
    names = list(losses_by_model)
    stat = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    pval = pd.DataFrame(np.nan, index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            if a == b:
                continue
            r = dm_test(losses_by_model[a], losses_by_model[b], lag=lag)
            stat.loc[a, b] = r["stat"]
            pval.loc[a, b] = r["p_value"]
    return stat, pval


def dm_vs_reference(losses_by_model, reference, lag=24):
    """One row per model, each tested against a single reference model."""
    rows = []
    for name, loss in losses_by_model.items():
        if name == reference:
            continue
        r = dm_test(loss, losses_by_model[reference], lag=lag)
        rows.append({
            "model": name,
            "mean_loss": float(np.mean(loss)),
            "vs_ref_diff": r["mean_diff"],
            "DM_stat": r["stat"],
            "p_value": r["p_value"],
        })
    return pd.DataFrame(rows).sort_values("mean_loss").reset_index(drop=True)


def dm_latex(stat, pval, caption, label, float_fmt="{:.2f}"):
    """Lower-triangular DM statistic table with significance stars."""
    names = list(stat.index)
    short = [n if len(n) <= 22 else n[:21] + "." for n in names]
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\scriptsize",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{l" + "c" * (len(names) - 1) + "}", r"\toprule",
        " & " + " & ".join(short[:-1]) + r" \\", r"\midrule",
    ]
    for i, a in enumerate(names):
        if i == 0:
            continue
        cells = []
        for j, b in enumerate(names[:-1]):
            if j >= i:
                cells.append("")
                continue
            s, p = stat.loc[a, b], pval.loc[a, b]
            star = "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""
            cells.append("$" + float_fmt.format(s) + star + "$")
        lines.append(short[i] + " & " + " & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ]
    return "\n".join(lines)


# ======================================================================
# 3. Across-seed spread
# ======================================================================

def seed_spread(member_preds, y_true, score_fn=None, n_score_samples=200, seed=123):
    """
    member_preds: list of (n_obs, 10, m_member) arrays, one per seed.
    Returns mean / sd / min / max of the score across seeds, so you can
    say whether a gap like 18.730 vs 18.739 is inside run-to-run noise.
    """
    if score_fn is None:
        score_fn = lambda p, y: energy_score_per_obs(p, y).mean()
    rng = np.random.default_rng(seed)
    vals = []
    for p in member_preds:
        if p.shape[2] > n_score_samples:
            idx = np.sort(rng.choice(p.shape[2], n_score_samples, replace=False))
            p = p[:, :, idx]
        vals.append(float(score_fn(p, y_true)))
    v = np.asarray(vals)
    return {"mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(),
            "max": v.max(), "n_seeds": len(v), "values": vals}


# ======================================================================
# 4. Naive / random baselines for the RTP tables
# ======================================================================

def rtp_baselines(y_true, masks=None):
    """
    Reference rows for Tables 24/25.

    Note: selling at a uniformly random sub-period has expected profit
    sum_i mean_j y[i,j], which is exactly the 'avg' strategy already in
    eval_proper_old.py.  So the random-selection RTP needs no simulation
    -- it is the 'Random / average' row below, computed in closed form.

    masks: optional {name: boolean (n_obs,)} from build_stress_masks().
    """
    if masks is None:
        masks = {"all": np.ones(len(y_true), dtype=bool)}

    strategies = {
        "Random selection (= expected uniform pick)": lambda y: y.mean(axis=1),
        "Always h1 (farthest from anchor)": lambda y: y[:, 0],
        "Always h10 (closest to anchor)": lambda y: y[:, -1],
    }

    rows = []
    for label, fn in strategies.items():
        row = {"strategy": label}
        for mname, mask in masks.items():
            ysub = y_true[mask]
            profit = float(np.sum(fn(ysub)))
            cb_min = float(np.sum(np.min(ysub, axis=1)))
            cb_max = float(np.sum(np.max(ysub, axis=1)))
            row[mname] = 100.0 * (profit - cb_min) / (cb_max - cb_min)
        rows.append(row)

    # best fixed sub-period chosen in hindsight -- an upper bound on any
    # non-adaptive rule, useful as a ceiling for "does timing help at all"
    row = {"strategy": "Best fixed sub-period (ex post)"}
    for mname, mask in masks.items():
        ysub = y_true[mask]
        cb_min = float(np.sum(np.min(ysub, axis=1)))
        cb_max = float(np.sum(np.max(ysub, axis=1)))
        best = max(100.0 * (float(np.sum(ysub[:, j])) - cb_min) / (cb_max - cb_min)
                   for j in range(ysub.shape[1]))
        row[mname] = best
    rows.append(row)

    return pd.DataFrame(rows)

def main():
    loaded_preds = {}

    # ----- load truth -----
    y_true = np.load(Y_TEST_FILE)
    y_true = ensure_y_shape(y_true)
    print("Loaded y_true:", y_true.shape)

    stress_masks = build_stress_masks(y_true, tail_pct=0.1)
    PERIODS = list(stress_masks.items())

    # ----- naive baselines -----
    print("\nNaive / crystal-ball baselines:")
    naive_rows = []
    for label, mask in PERIODS:
        cb_min, cb_max = crystal_ball_bounds(y_true[mask])
        for s in ["first", "last", "avg"]:
            p = naive_strategy_profit(y_true[mask], s)
            naive_rows.append({
                "period": label,
                "strategy": f"naive_{s}",
                "n_obs": int(mask.sum()),
                "profit": p,
                "rtp": realized_trading_potential(p, cb_min, cb_max),
                "cb_min": cb_min,
                "cb_max": cb_max,
            })
            print(f"  [{label}] naive_{s}_profit={p:.4f}  rtp={realized_trading_potential(p, cb_min, cb_max):.4f}")


    # ----- threshold-weighted scoring setup (once) -----
    if TW_THRESHOLD_SOURCE == "train" and Y_TRAIN_FILE.exists():
        y_thr_src = ensure_y_shape(np.load(Y_TRAIN_FILE));
        thr_label = "train"
    else:
        y_thr_src = y_true;
        thr_label = "test"
        if TW_THRESHOLD_SOURCE == "train":
            print(f"WARNING: {Y_TRAIN_FILE} not found, falling back to test-set thresholds.")
    t_lo, t_hi = compute_tw_thresholds(y_thr_src, TW_TAIL_PCT)
    chaining = make_chaining_functions(t_lo, t_hi)
    print(f"\ntwScore thresholds ({thr_label}): t_lo={t_lo:.3f}, t_hi={t_hi:.3f}")

    tw_rows = []
    trading_rows = []
    per_horizon_rows = []

    for model_name, file_path in MODEL_FILES.items():
        if not file_path.exists():
            print(f"\nSkipping {model_name}: file not found -> {file_path}")
            continue

        pred = ensure_pred_shape(np.load(file_path))
        pred = maybe_subsample_samples(pred, N_EVAL_SAMPLES, seed=123)
        loaded_preds[model_name] = pred

        if pred.shape[0] != y_true.shape[0] or pred.shape[1] != y_true.shape[1]:
            raise ValueError(f"shape mismatch for {model_name}: {pred.shape} vs {y_true.shape}")

        print(f"\nEvaluating {model_name} | pred shape = {pred.shape}")
        pred_score = maybe_subsample_samples(pred, N_SCORE_SAMPLES, seed=123)

        for scheme, v_func in chaining.items():
            es = tw_energy_score(pred_score, y_true, v_func)
            crps_h = tw_crps_by_horizon(pred_score, y_true, v_func)
            row = {
                "model": model_name, "weighting": scheme, "n_obs": int(len(y_true)),
                "energy_score": es, "crps_mean": float(crps_h.mean()),
                "t_lo": t_lo, "t_hi": t_hi, "threshold_src": thr_label,
            }
            if scheme == "unweighted":
                row["mae_mean"] = float(median_mae_by_horizon(pred, y_true).mean())
                row["dss"] = dawid_sebastiani_score(pred_score, y_true, obs_batch_size=128)
                row["vs_p1"] = variogram_score(pred_score, y_true, p=1.0)
                row["vs_p05"] = variogram_score(pred_score, y_true, p=0.5)
            tw_rows.append(row)
            for h in range(y_true.shape[1]):
                per_horizon_rows.append({
                    "model": model_name, "weighting": scheme,
                    "horizon": h + 1, "crps": float(crps_h[h]),
                })

        for label, mask in PERIODS:
            chosen = mode_argmax(pred[mask])
            profit = profit_from_chosen_horizon(y_true[mask], chosen)
            cb_min, cb_max = crystal_ball_bounds(y_true[mask])
            trading_rows.append({
                "model": model_name, "regime": label, "n_obs": int(mask.sum()),
                "profit_majority_vote": profit,
                "rtp_majority_vote": realized_trading_potential(profit, cb_min, cb_max),
                "mae_mean": float(median_mae_by_horizon(pred[mask], y_true[mask]).mean()),
            })

    # --------------------------------------------------------
    # Scalar ID1 / ID2 / ID3 proxy evaluation
    # --------------------------------------------------------
    print("\nEvaluating scalar ID1 / ID2 / ID3 proxy targets...")

    scalar_forecast_sets = dict(loaded_preds)

    # Same-contract last available VWAP baseline.
    # This is only added to scalar evaluation, not to the path/portal plots.
    try:
        last_p_test = build_same_contract_last_p_test_baseline(
            data_dir=CGM_DIR,
            n_test_obs=len(y_true),
            start_index_obs=START_INDEX_OBS,
        )

        pred_last_p_path = build_constant_path_samples_from_point(
            point_forecast=last_p_test,
            horizon=y_true.shape[1],
            n_samples=N_EVAL_SAMPLES,
        )

        scalar_forecast_sets["naive_same_contract_last_p"] = pred_last_p_path

        print(
            "Added scalar naive baseline: naive_same_contract_last_p "
            f"| shape={pred_last_p_path.shape}"
        )

    except Exception as e:
        print(
            "\nWARNING: Could not build naive_same_contract_last_p baseline. "
            "Scalar model evaluation will continue without it."
        )
        print(f"Reason: {e}")

    scalar_results_df = evaluate_scalar_targets_for_forecasts(
        forecast_sets=scalar_forecast_sets,
        y_true_path=y_true,
        target_specs=SCALAR_TARGET_SPECS,
        threshold_scope="test",
    )

    scalar_results_df = scalar_results_df.sort_values(
        ["target", "period", "crps", "mae_median", "model"]
    ).reset_index(drop=True)

    scalar_summary_path = OUT_DIR / "scalar_id1_id2_id3_summary_metrics.csv"
    scalar_results_df.to_csv(scalar_summary_path, index=False)

    scalar_pivot = scalar_results_df.pivot_table(
        index=["target", "model"],
        columns="period",
        values=[
            "mae_median",
            "rmse_median",
            "bias_median",
            "mae_mean",
            "rmse_mean",
            "bias_mean",
            "crps",
            "coverage_80",
            "width_80",
            "coverage_60",
            "width_60",
            "pinball_q10",
            "pinball_q50",
            "pinball_q90",
        ],
    )

    scalar_pivot_path = OUT_DIR / "scalar_id1_id2_id3_summary_metrics_pivot.csv"
    scalar_pivot.to_csv(scalar_pivot_path)

    scalar_all_only = (
        scalar_results_df[scalar_results_df["period"] == "all"]
        .sort_values(["target", "crps", "mae_median", "model"])
        .reset_index(drop=True)
    )

    scalar_all_only_path = OUT_DIR / "scalar_id1_id2_id3_summary_metrics_all_only.csv"
    scalar_all_only.to_csv(scalar_all_only_path, index=False)

    # ================================================================
    # Calibration: standard PIT (full set) + proper tail calibration
    # Replaces the stress-mask-conditioned PIT — masking the tail had the
    # same outcome-selection problem as the sloppy tail scores.
    # ================================================================
    print("\nComputing PIT + tail-calibration metrics...")
    pit_dir = OUT_DIR / "pit_calibration"
    pit_dir.mkdir(exist_ok=True)

    y_train = ensure_y_shape(np.load(Y_TRAIN_FILE))  # tail-cal thresholds (no look-ahead)

    # ---- (1) standard PIT on the FULL set (proper bulk calibration) ----
    pit_summary_rows, pit_horizon_rows, pit_dict_all = [], [], {}
    for model_name, pred in loaded_preds.items():
        pit = compute_pit_values(pred, y_true)  # full set, no mask
        pit_dict_all[model_name] = pit
        m = pit_calibration_metrics(pit)
        pit_summary_rows.append({
            "model": model_name, "n_obs": int(len(y_true)),
            "pit_mean": m["pit_mean"], "pit_std": m["pit_std"],
            "ks_stat": m["ks_stat"], "ks_pval": m["ks_pval"],
            "mace": m["mace"], "ace": m["ace"],
        })
        by_h = pit_metrics_by_horizon(pit)
        for h in range(pred.shape[1]):
            pit_horizon_rows.append({
                "model": model_name, "horizon": h + 1,
                "pit_mean": float(by_h["mean"][h]),
                "pit_std": float(by_h["std"][h]),
                "ks_stat": float(by_h["ks_stat"][h]),
            })

    plot_pit_diagnostics(pit_dict_all, out_dir=pit_dir / "all", mask_label="all",
                         n_bins=PIT_N_BINS, quantile_levels=PIT_QUANTILE_LEVELS)

    pd.DataFrame(pit_summary_rows).sort_values(["mace", "model"]).to_csv(
        pit_dir / "pit_summary_metrics.csv", index=False)
    pd.DataFrame(pit_horizon_rows).to_csv(
        pit_dir / "pit_horizon_metrics.csv", index=False)

    # ---- per-horizon PIT histograms (plots + CSV bin data) ----
    plot_pit_histograms_by_horizon(
        pit_dict_all, out_dir=pit_dir / "all", mask_label="all", n_bins=PIT_N_BINS
    )
    pit_hist_h_df = pit_histograms_by_horizon(pit_dict_all, n_bins=PIT_N_BINS)
    pit_hist_h_df.to_csv(pit_dir / "pit_histograms_by_horizon.csv", index=False)

    # ---- (2) proper tail calibration (replaces upper/lower/both masks) ----
    tc_h = tail_cal_per_horizon(loaded_preds, y_true, y_train,
                                levels_upper=(0.90, 0.95, 0.99),
                                levels_lower=(0.10, 0.05, 0.01),
                                n_rand=10)
    tc_h.to_csv(pit_dir / "tail_calibration_per_horizon.csv", index=False)

    curve_dir = pit_dir / "tail_cal_curves"
    curve_dir.mkdir(exist_ok=True)

    for tail, qs in (("upper", (0.90, 0.95, 0.99)), ("lower", (0.10, 0.05, 0.01))):
        curves = pd.concat(
            [tail_cal_curves_pooled(loaded_preds, y_true, y_train, tail, q, n_rand=20)
             for q in qs],
            ignore_index=True,
        )
        curves.to_csv(curve_dir / f"tail_cal_curves_{tail}.csv", index=False)
        plot_tail_cal_curves(curves, ratio="R_com", q_levels=qs,
                             out_path=curve_dir / f"reliability_com_{tail}.png",
                             title=f"Tail calibration (combined) — {tail} tail")
        plot_tail_cal_curves(curves, ratio="R_sev", q_levels=qs,
                             out_path=curve_dir / f"reliability_sev_{tail}.png",
                             title=f"Tail calibration (severity) — {tail} tail")
        plot_occ_by_horizon(tc_h, tail=tail, q=qs[1],
                            out_path=curve_dir / f"occ_by_horizon_{tail}_q{qs[1]}.png")

    # pooled over horizons = the headline calibration-on-extremes table
    tc_pooled = (tc_h.groupby(["model", "tail", "q"])[["occ", "sev_sup", "com_sup"]]
                 .mean().reset_index()
                 .sort_values(["tail", "q", "com_sup"]))
    tc_pooled.to_csv(pit_dir / "tail_calibration_pooled.csv", index=False)

    # sup-distance diagram (combined ratio) vs threshold, one line per model, per tail
    for tail in ("upper", "lower"):
        d = tc_pooled[tc_pooled["tail"] == tail]
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_name, g in d.groupby("model"):
            g = g.sort_values("q")
            ax.plot(g["q"], g["com_sup"], marker="o", ms=4, label=model_name)
        ax.axhline(0.0, ls=":", color="k", lw=1)
        ax.set_xlabel("threshold quantile level q");
        ax.set_ylabel("Combined-ratio sup-distance")
        ax.set_title(f"Tail calibration — {tail} tail");
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(pit_dir / f"tail_calibration_{tail}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"\nCalibration outputs saved to: {pit_dir.resolve()}")
    print("\n===== STANDARD PIT (full set) =====")
    print(pd.DataFrame(pit_summary_rows)[["model", "mace", "ace", "ks_stat", "pit_mean", "pit_std"]]
          .round(4).to_string(index=False))
    print("\n===== TAIL CALIBRATION (pooled over horizons) =====")
    print(tc_pooled.round(4).to_string(index=False))

    # ---- marginal distribution overlay (per stress regime) ----
    marginal_dir = OUT_DIR / "marginal_distribution_overlay"
    for label, mask in PERIODS:
        plot_marginal_distribution_overlay(
            loaded_preds, y_true, out_dir=marginal_dir,
            mask=mask, mask_label=label,
        )
    print(f"\nMarginal distribution overlays saved to: {marginal_dir.resolve()}")

    # ---- dispersion ratio diagnostic (per stress regime) ----
    print("\nComputing dispersion ratio (model std / empirical std) by regime...")
    dispersion_rows = []
    for label, mask in PERIODS:
        df_disp = dispersion_ratio_by_horizon(loaded_preds, y_true, mask)
        df_disp["regime"] = label
        dispersion_rows.append(df_disp)

    dispersion_df = pd.concat(dispersion_rows, ignore_index=True)
    dispersion_df.to_csv(OUT_DIR / "dispersion_ratio_by_horizon.csv", index=False)

    dispersion_pivot = (
        dispersion_df.groupby(["model", "regime"])["dispersion_ratio"]
        .mean()
        .unstack("regime")
        .sort_index()
    )
    dispersion_pivot.to_csv(OUT_DIR / "dispersion_ratio_pivot.csv")

    print("\n===== DISPERSION RATIO (model std / empirical std), averaged over subperiods =====")
    print(dispersion_pivot.round(3).to_string())

    # ----- save outputs -----
    wt_order = ["unweighted", "upper_tail", "lower_tail", "both_tails"]
    tw_df = pd.DataFrame(tw_rows)
    tw_df["weighting"] = pd.Categorical(tw_df["weighting"], wt_order, ordered=True)
    tw_df = tw_df.sort_values(["weighting", "energy_score"]).reset_index(drop=True)

    trading_df = pd.DataFrame(trading_rows)
    per_horizon_df = pd.DataFrame(per_horizon_rows)
    naive_df = pd.DataFrame(naive_rows)

    tw_df.to_csv(OUT_DIR / "distributional_scores_tw.csv", index=False)
    trading_df.to_csv(OUT_DIR / "trading_by_regime.csv", index=False)
    per_horizon_df.to_csv(OUT_DIR / "per_horizon_crps_tw.csv", index=False)
    naive_df.to_csv(OUT_DIR / "naive_baselines.csv", index=False)

    tw_df.pivot_table(index="model", columns="weighting",
                      values=["energy_score", "crps_mean"], observed=False
                      ).to_csv(OUT_DIR / "distributional_scores_tw_pivot.csv")
    trading_df.pivot_table(index="model", columns="regime",
                           values=["rtp_majority_vote", "profit_majority_vote"]
                           ).to_csv(OUT_DIR / "trading_by_regime_pivot.csv")

    print("\n========== THRESHOLD-WEIGHTED DISTRIBUTIONAL SCORES (full set) ==========")
    print(tw_df[["weighting", "model", "energy_score", "crps_mean"]].round(4).to_string(index=False))
    print("\n========== TRADING METRICS BY STRESS REGIME (descriptive) ==========")
    print(trading_df.round(4).to_string(index=False))
    print(f"\nSaved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()