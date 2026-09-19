from pathlib import Path
Y_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
CGM_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression")
ENG_EXP_DIR_1 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10")
ENG_EXP_DIR_2 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10")
CGM_HYBRID_DIR = Path(r"C:\Users\montenegrof\PycharmProjects\epf_cgm\DATA_DB\hybrid_outputs")
ENG_HTS_DIR_E16 = Path(r"C:\Users\montenegrof\PycharmProjects\epf_cgm\DATA_DB\engression_outputs_hts\engression_hts__embed16__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E200_ENS10")
ENG_EXP_DIR_2_HETERO_ALL_256 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
ENG_EXP_DIR_2_COMPLETE = Path(r"C:\Users\montenegrof\PycharmProjects\epf_cgm\engression_outputs_experiments\full__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10")
ENG_EXP_DIR_1_HETERO_ALL_256 = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256")
N_EVAL_SAMPLES = 1000
Y_TEST_PATH = ENG_EXP_DIR_1_HETERO_ALL_256 / "y_test.npy"
MODEL_FILES = {
    "engression_experiment_1": ENG_EXP_DIR_1 / "pred.npy",
    "engression_experiment_2": ENG_EXP_DIR_2 / "pred.npy",
    "engression_experiment_2_hetero_all_256": ENG_EXP_DIR_2_HETERO_ALL_256 / "pred.npy",
    "engression_experiment_1_hetero_all_256": ENG_EXP_DIR_1_HETERO_ALL_256 / "pred.npy",
    "cgm_esloss": CGM_DIR / "pred_cgm_esloss.npy",
    "lasso_bootstrap": CGM_DIR / "lasso_bootstrap.npy",
}

import numpy as np
from eval_proper_old import (
    energy_score_per_obs, crps_per_obs, variogram_per_obs,
    dm_matrix, dm_vs_reference, dm_latex, rtp_baselines)

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


preds  = {name: np.load(p) for name, p in MODEL_FILES.items()}
y_true = np.load(Y_TEST_PATH)
preds  = {k: maybe_subsample_samples(v, 200, seed=123) for k, v in preds.items()}

es_losses = {k: energy_score_per_obs(v, y_true) for k, v in preds.items()}
stat, pval = dm_matrix(es_losses, lag=24)
print(dm_latex(stat, pval, "DM statistics on the energy score, old window.", "Tab:DMRecent"))

print(dm_vs_reference(es_losses, reference="lasso_bootstrap", lag=24))
print(rtp_baselines(y_true, build_stress_masks(y_true)))