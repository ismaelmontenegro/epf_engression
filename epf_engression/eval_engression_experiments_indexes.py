import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass
class EvalConfig:
    # Point this to one run directory created by engression_experiments_indexes.py.
    # New thesis-comparison runs use 10-step time-weighted proxy targets.
    run_dir: str = r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_idx3_experiments\compact__Yid1__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E200_ENS10_HETERO_True_HS_256"

    # Used only to reconstruct the naive same-contract last VWAP baseline.
    # If omitted, the script tries to read data_dir from config.json in run_dir.
    data_dir: Optional[str] = None

    # Metric settings
    # Default only used if target_columns.json is missing.
    # Works for both joint IDX3 runs and scalar runs such as ["id3_p"].
    target_cols: Tuple[str, ...] = ("id3_p", "id2_p", "id1_p")
    n_score_samples: int = 1000
    score_seed: int = 123

    # Energy Score pairwise term approximation.
    # Exact ES would allocate roughly n_obs * n_samples^2 distances, which is huge.
    # 256-1024 pair draws is usually enough for stable model comparison.
    n_energy_pair_draws: int = 512

    # Coverage levels for marginal intervals
    coverage_levels: Tuple[float, ...] = (0.5, 0.6, 0.8, 0.9)

    # Output
    out_subdir: str = "evaluation_idx3"


CFG = EvalConfig()

EXPECTED_TARGET_DEFINITION = "10_step_time_weighted_proxy"


# ============================================================
# Basic helpers
# ============================================================

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_splits(n_obs: int, cfg_dict: dict):
    n_days = int(cfg_dict["n_days"])
    test_days = int(cfg_dict["test_days"])
    n_hours = int(cfg_dict["n_hours"])

    trainval_len = (n_days - test_days - 7) * n_hours
    val_size = int(0.2 * trainval_len)
    train_size = trainval_len - val_size

    idx_train = np.arange(0, train_size)
    idx_val = np.arange(train_size, trainval_len)
    idx_test = np.arange(trainval_len, n_obs)

    return idx_train, idx_val, idx_test


def read_hourly_id_file(folder: Path, prefix: str, n_keep: int, n_days: int, n_hours: int, start_date: datetime):
    arr = np.zeros((n_days, n_hours, 2 + n_keep), dtype=np.float64)

    for n_hour in range(n_hours):
        fname = folder / f"{prefix}_{n_hour:02d}"
        with open(fname) as f:
            lines = f.readlines()

        date_hour = [str(line.split(",")[0].strip()) for line in lines[1:]]
        data = np.array([[float(e) for e in line.strip().split(",")[1:]] for line in lines[1:]], dtype=np.float64)
        kept = data[:, :n_keep]

        date_str = [
            datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
            for date_string in date_hour
        ]
        date_diff = [
            (datetime.strptime(date_string, "%Y%m%d") - start_date).days
            for date_string in date_str
        ]

        arr[:, n_hour, 0] = np.array(date_diff) + 1
        arr[:, n_hour, 1] = n_hour
        arr[:, n_hour, 2:] = kept

    return arr


def reconstruct_last_contract_vwap_test(
    run_dir: Path,
    config: dict,
    n_test_expected: int,
    n_targets: int,
) -> np.ndarray:
    """
    Reconstructs the naive baseline: last observed same-contract VWAP, i.e. last_p.

    Returns
    -------
    naive : np.ndarray, shape (n_test, n_targets)
        Same last_p repeated for all evaluated targets.
    """
    data_dir = Path(CFG.data_dir) if CFG.data_dir is not None else Path(config["data_dir"])
    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    n_days = int(config["n_days"])
    n_hours = int(config["n_hours"])
    start_index = int(config["start_index"])

    pricesintra = read_hourly_id_file(
        folder=data_dir / "ID_DATA",
        prefix="prices_hourly",
        n_keep=13,
        n_days=n_days,
        n_hours=n_hours,
        start_date=start_date,
    )
    pricesintra_flat = pricesintra.reshape(-1, pricesintra.shape[-1])

    # pricesintra_flat columns: day, hour, id_1, ..., id_12, last_p
    last_p_flat = pricesintra_flat[:, 14].astype(np.float64)
    last_p_trimmed = last_p_flat[start_index:]

    _, _, idx_test = make_splits(len(last_p_trimmed), config)
    last_p_test = last_p_trimmed[idx_test]

    if len(last_p_test) != n_test_expected:
        raise ValueError(
            f"Naive baseline length mismatch: got {len(last_p_test)}, expected {n_test_expected}. "
            f"Check run config and data files."
        )

    return np.repeat(last_p_test[:, None], n_targets, axis=1)


def reconstruct_test_timestamps(config: dict, n_test_expected: int) -> pd.DatetimeIndex:
    start_date = datetime.strptime(config["start_date"], "%Y-%m-%d")
    n_days = int(config["n_days"])
    n_hours = int(config["n_hours"])
    start_index = int(config["start_index"])

    all_times = []
    for d in range(n_days):
        for h in range(n_hours):
            all_times.append(start_date + timedelta(days=d, hours=h))

    all_times = pd.DatetimeIndex(all_times)
    trimmed = all_times[start_index:]
    _, _, idx_test = make_splits(len(trimmed), config)
    test_times = trimmed[idx_test]

    if len(test_times) != n_test_expected:
        raise ValueError(
            f"Timestamp length mismatch: got {len(test_times)}, expected {n_test_expected}."
        )

    return test_times


# ============================================================
# Metrics
# ============================================================

def maybe_subsample_samples(pred: np.ndarray, n_score_samples: int, seed: int) -> np.ndarray:
    """
    pred shape: (n_obs, n_targets, n_samples)
    """
    n_samples = pred.shape[2]
    pred = pred.astype(np.float32, copy=False)

    if n_samples <= n_score_samples:
        return pred

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_samples, size=n_score_samples, replace=False)
    return pred[:, :, idx]


def crps_ensemble_1d(y: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Empirical CRPS for one target.

    y:       shape (n_obs,)
    samples: shape (n_obs, n_samples)

    Returns CRPS per observation, shape (n_obs,).
    """
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)

    s_sorted = np.sort(samples, axis=1)
    m = s_sorted.shape[1]
    weights = (2 * np.arange(1, m + 1) - m - 1).astype(np.float64)
    # mean pairwise absolute difference = 2 / m^2 * sum_i (2i - m - 1) x_(i)
    mean_pairwise = (2.0 / (m ** 2)) * np.sum(s_sorted * weights[None, :], axis=1)
    term2 = 0.5 * mean_pairwise

    return term1 - term2


def energy_score(y: np.ndarray, pred: np.ndarray, seed: int = 123, n_pair_draws: Optional[int] = 512) -> np.ndarray:
    """
    Empirical multivariate Energy Score per observation.

    y:    shape (n_obs, d)
    pred: shape (n_obs, d, m)

    Returns ES per observation, shape (n_obs,).

    Important:
    The exact pairwise term is O(n_obs * m^2) and can require tens of GB.
    Therefore this function uses a Monte Carlo approximation by default.
    """
    y = y.astype(np.float32, copy=False)
    samples = np.transpose(pred.astype(np.float32, copy=False), (0, 2, 1))  # (n_obs, m, d)
    _, m, _ = samples.shape

    term1 = np.mean(np.linalg.norm(samples - y[:, None, :], axis=2), axis=1)

    if n_pair_draws is None:
        # Exact mode, only safe for small m / small n_obs.
        diffs = samples[:, :, None, :] - samples[:, None, :, :]
        pairwise = np.linalg.norm(diffs, axis=3)
        term2 = 0.5 * np.mean(pairwise, axis=(1, 2))
    else:
        rng = np.random.default_rng(seed)
        i = rng.integers(0, m, size=n_pair_draws)
        j = rng.integers(0, m, size=n_pair_draws)
        pairwise = np.linalg.norm(samples[:, i, :] - samples[:, j, :], axis=2)
        term2 = 0.5 * np.mean(pairwise, axis=1)

    return term1 - term2


def point_energy_score(y: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Energy Score of a deterministic forecast equals Euclidean error.
    """
    return np.linalg.norm(point - y, axis=1)


def interval_coverage_and_width(y: np.ndarray, samples: np.ndarray, level: float) -> Tuple[float, float]:
    alpha = 1.0 - level
    lo = np.quantile(samples, alpha / 2.0, axis=1)
    hi = np.quantile(samples, 1.0 - alpha / 2.0, axis=1)
    coverage = np.mean((y >= lo) & (y <= hi))
    width = np.mean(hi - lo)
    return float(coverage), float(width)


def pinball_loss(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    e = y - q
    return np.maximum(tau * e, (tau - 1.0) * e)


def summarize_model(y: np.ndarray, pred: np.ndarray, target_cols: List[str], coverage_levels: Tuple[float, ...]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    y:    (n_obs, 3)
    pred: (n_obs, 3, n_samples)
    """
    rows = []
    obs_rows = []

    pred_mean = np.mean(pred, axis=2)
    pred_median = np.median(pred, axis=2)

    es_obs = energy_score(y, pred, seed=CFG.score_seed, n_pair_draws=CFG.n_energy_pair_draws)
    obs_rows.append(pd.DataFrame({"metric": "energy_score", "value": es_obs}))

    rows.append({
        "model": "engression_idx3",
        "target": "joint",
        "metric": "energy_score",
        "value": float(np.mean(es_obs)),
    })

    for j, target in enumerate(target_cols):
        samples_j = pred[:, j, :]
        y_j = y[:, j]

        crps_obs = crps_ensemble_1d(y_j, samples_j)
        mae_mean_obs = np.abs(pred_mean[:, j] - y_j)
        mae_median_obs = np.abs(pred_median[:, j] - y_j)
        rmse_mean = np.sqrt(np.mean((pred_mean[:, j] - y_j) ** 2))

        rows.extend([
            {"model": "engression_idx3", "target": target, "metric": "crps", "value": float(np.mean(crps_obs))},
            {"model": "engression_idx3", "target": target, "metric": "mae_mean", "value": float(np.mean(mae_mean_obs))},
            {"model": "engression_idx3", "target": target, "metric": "mae_median", "value": float(np.mean(mae_median_obs))},
            {"model": "engression_idx3", "target": target, "metric": "rmse_mean", "value": float(rmse_mean)},
            {"model": "engression_idx3", "target": target, "metric": "bias_mean", "value": float(np.mean(pred_mean[:, j] - y_j))},
            {"model": "engression_idx3", "target": target, "metric": "bias_median", "value": float(np.mean(pred_median[:, j] - y_j))},
            {"model": "engression_idx3", "target": target, "metric": "pinball_q10", "value": float(np.mean(pinball_loss(y_j, np.quantile(samples_j, 0.1, axis=1), 0.1)))},
            {"model": "engression_idx3", "target": target, "metric": "pinball_q50", "value": float(np.mean(pinball_loss(y_j, np.quantile(samples_j, 0.5, axis=1), 0.5)))},
            {"model": "engression_idx3", "target": target, "metric": "pinball_q90", "value": float(np.mean(pinball_loss(y_j, np.quantile(samples_j, 0.9, axis=1), 0.9)))},
        ])

        for level in coverage_levels:
            cov, width = interval_coverage_and_width(y_j, samples_j, level)
            rows.append({"model": "engression_idx3", "target": target, "metric": f"coverage_{int(level * 100)}", "value": cov})
            rows.append({"model": "engression_idx3", "target": target, "metric": f"width_{int(level * 100)}", "value": width})

        obs_rows.append(pd.DataFrame({
            "target": target,
            "crps": crps_obs,
            "mae_mean": mae_mean_obs,
            "mae_median": mae_median_obs,
        }))

    return pd.DataFrame(rows), pd.concat(obs_rows, axis=0, ignore_index=True)


def summarize_naive(y: np.ndarray, naive: np.ndarray, target_cols: List[str]) -> pd.DataFrame:
    rows = []

    es_obs = point_energy_score(y, naive)
    rows.append({
        "model": "naive_last_contract_vwap",
        "target": "joint",
        "metric": "energy_score",
        "value": float(np.mean(es_obs)),
    })

    for j, target in enumerate(target_cols):
        err = naive[:, j] - y[:, j]
        abs_err = np.abs(err)
        rows.extend([
            {"model": "naive_last_contract_vwap", "target": target, "metric": "crps", "value": float(np.mean(abs_err))},
            {"model": "naive_last_contract_vwap", "target": target, "metric": "mae_mean", "value": float(np.mean(abs_err))},
            {"model": "naive_last_contract_vwap", "target": target, "metric": "mae_median", "value": float(np.mean(abs_err))},
            {"model": "naive_last_contract_vwap", "target": target, "metric": "rmse_mean", "value": float(np.sqrt(np.mean(err ** 2)))},
            {"model": "naive_last_contract_vwap", "target": target, "metric": "bias_mean", "value": float(np.mean(err))},
            {"model": "naive_last_contract_vwap", "target": target, "metric": "bias_median", "value": float(np.mean(err))},
        ])

    return pd.DataFrame(rows)


def build_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot_table(
        index=["target", "metric"],
        columns="model",
        values="value",
        aggfunc="first",
    ).reset_index()

    if "engression_idx3" in pivot.columns and "naive_last_contract_vwap" in pivot.columns:
        pivot["improvement_vs_naive_pct"] = (
            (pivot["naive_last_contract_vwap"] - pivot["engression_idx3"])
            / pivot["naive_last_contract_vwap"]
            * 100.0
        )

    return pivot


# ============================================================
# Main
# ============================================================

def main():
    run_dir = Path(CFG.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    config = load_json(run_dir / "config.json")
    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    norm_stats_path = run_dir / "normalization_stats.json"
    norm_stats = load_json(norm_stats_path) if norm_stats_path.exists() else {}
    target_definition = norm_stats.get("target_definition", {})
    target_definition_type = target_definition.get("type")

    if target_definition_type != EXPECTED_TARGET_DEFINITION:
        print(
            "WARNING: this run does not advertise the expected 10-step "
            f"time-weighted proxy target definition. Found: {target_definition_type!r}. "
            "Old scalar runs may have used 12-bucket volume-weighted VWAP targets."
        )

    target_cols_path = run_dir / "target_columns.json"
    if target_cols_path.exists():
        target_cols = load_json(target_cols_path)
    else:
        target_cols = list(CFG.target_cols)

    pred = np.load(run_dir / "pred.npy")
    y_test = np.load(run_dir / "y_test.npy")

    if pred.ndim != 3:
        raise ValueError(f"Expected pred.npy shape (n_test, n_targets, n_samples), got {pred.shape}")
    if y_test.ndim != 2:
        raise ValueError(f"Expected y_test.npy shape (n_test, n_targets), got {y_test.shape}")
    if pred.shape[0] != y_test.shape[0] or pred.shape[1] != y_test.shape[1]:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, y_test {y_test.shape}")

    pred_score = maybe_subsample_samples(pred, CFG.n_score_samples, CFG.score_seed)

    if len(target_cols) != y_test.shape[1]:
        raise ValueError(
            f"target_columns.json has {len(target_cols)} targets {target_cols}, "
            f"but y_test.npy has {y_test.shape[1]} columns."
        )

    naive = reconstruct_last_contract_vwap_test(
        run_dir,
        config,
        n_test_expected=y_test.shape[0],
        n_targets=y_test.shape[1],
    )
    test_times = reconstruct_test_timestamps(config, n_test_expected=y_test.shape[0])

    out_dir = run_dir / CFG.out_subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    model_summary, obs_metrics = summarize_model(
        y=y_test,
        pred=pred_score,
        target_cols=target_cols,
        coverage_levels=CFG.coverage_levels,
    )
    naive_summary = summarize_naive(y=y_test, naive=naive, target_cols=target_cols)

    summary = pd.concat([model_summary, naive_summary], ignore_index=True)
    comparison = build_comparison_table(summary)

    # Save detailed prediction diagnostics for downstream plotting/debugging.
    pred_mean = np.mean(pred_score, axis=2)
    pred_median = np.median(pred_score, axis=2)

    diagnostics = pd.DataFrame({"timestamp": test_times})
    for j, target in enumerate(target_cols):
        diagnostics[f"y_{target}"] = y_test[:, j]
        diagnostics[f"pred_mean_{target}"] = pred_mean[:, j]
        diagnostics[f"pred_median_{target}"] = pred_median[:, j]
        diagnostics[f"naive_last_contract_vwap_{target}"] = naive[:, j]
        diagnostics[f"abs_err_median_{target}"] = np.abs(pred_median[:, j] - y_test[:, j])
        diagnostics[f"abs_err_naive_{target}"] = np.abs(naive[:, j] - y_test[:, j])

        for q in [0.05, 0.1, 0.25, 0.75, 0.9, 0.95]:
            diagnostics[f"q{int(q * 100):02d}_{target}"] = np.quantile(pred_score[:, j, :], q, axis=1)

    summary.to_csv(out_dir / "summary_metrics_long.csv", index=False)
    comparison.to_csv(out_dir / "comparison_vs_naive.csv", index=False)
    diagnostics.to_csv(out_dir / "diagnostics_by_observation.csv", index=False)

    metadata = {
        "run_dir": str(run_dir.resolve()),
        "target_cols": target_cols,
        "pred_shape_original": list(pred.shape),
        "pred_shape_scored": list(pred_score.shape),
        "y_test_shape": list(y_test.shape),
        "n_energy_pair_draws": CFG.n_energy_pair_draws,
        "naive_definition": "last observed same-contract VWAP, i.e. last_p, repeated for all evaluated targets",
        "expected_target_definition": EXPECTED_TARGET_DEFINITION,
        "target_definition": target_definition,
        "manifest": manifest,
    }
    with open(out_dir / "evaluation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("=" * 90)
    print("Index evaluation complete")
    print("=" * 90)
    print("Run dir:", run_dir.resolve())
    print("Output dir:", out_dir.resolve())
    print("Pred shape scored:", pred_score.shape)
    print("Y shape:", y_test.shape)
    print(f"Naive baseline: last observed same-contract VWAP repeated for {target_cols}")
    print("\nCore comparison:")

    core_metrics = [
        "energy_score",
        "crps",
        "mae_median",
        "mae_mean",
        "rmse_mean",
        "bias_median",
    ]
    core = comparison[comparison["metric"].isin(core_metrics)].copy()

    # Nice ordering
    target_order = {"joint": 0}
    target_order.update({target: i + 1 for i, target in enumerate(target_cols)})
    metric_order = {m: i for i, m in enumerate(core_metrics)}
    core["target_order"] = core["target"].map(target_order).fillna(99)
    core["metric_order"] = core["metric"].map(metric_order).fillna(99)
    core = core.sort_values(["target_order", "metric_order"]).drop(columns=["target_order", "metric_order"])

    with pd.option_context("display.max_rows", 100, "display.max_columns", 20, "display.width", 180):
        print(core.to_string(index=False))

    print("\nSaved files:")
    print(" -", out_dir / "summary_metrics_long.csv")
    print(" -", out_dir / "comparison_vs_naive.csv")
    print(" -", out_dir / "diagnostics_by_observation.csv")
    print(" -", out_dir / "evaluation_metadata.json")
    print("=" * 90)


if __name__ == "__main__":
    main()
