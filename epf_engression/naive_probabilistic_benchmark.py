from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# Base directory containing ID_DATA/prices_hourly_XX
CGM_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression")

# Pick an experiment directory containing: y_train.npy, y_test.npy
Y_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments\compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E200_ENS10_HETERO_False_HS_256")

Y_TRAIN_FILE = Y_DIR / "y_train.npy"
Y_TEST_FILE = Y_DIR / "y_test.npy"

# Output
OUT_DIR = CGM_DIR / "naive_probabilistic_evaluation_rolling_same_hour_old"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# Bootstrap settings
# ============================================================

# Match your Lasso bootstrap
BOOTSTRAP_WINDOW_DAYS = 30
# Number of trajectories. Your Lasso uses 10,000.
N_TRAJ = 10000
SEED = 123
# First 7 days removed by preprocessing
START_INDEX_OBS = 7 * 24

# ============================================================
# Scalar ID1 / ID2 / ID3 definitions
# These match the ACTUAL indexing in your evaluation script.
# ============================================================

SCALAR_SUBPERIOD_MINUTES = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 10], dtype=float)

SCALAR_TARGET_SPECS = {
    "id1_proxy_time_weighted": np.array([0, 1, 2, 3], dtype=int),
    "id2_proxy_time_weighted": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
    "id3_proxy_time_weighted": np.arange(10, dtype=int),
}


# ============================================================
# Shape helpers
# ============================================================

def ensure_y_shape(arr: np.ndarray) -> np.ndarray:
    """Force y into shape: (n_obs, 10)"""
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] == 10:
        return arr
    if arr.ndim == 3 and arr.shape[1] == 10 and arr.shape[2] == 1:
        return arr[:, :, 0]
    if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 10:
        return arr[:, 0, :]
    raise ValueError(f"Cannot infer y shape from {arr.shape}")


# ============================================================
# Read same-contract last_p
# ============================================================

def _resolve_hourly_file(folder: Path, stem: str, hour: int) -> Path:
    candidates = [folder / f"{stem}_{hour:02d}", folder / f"{stem}_{hour:02d}.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {stem} for hour {hour:02d}.\n"
        + "\n".join(str(p) for p in candidates)
    )


def read_last_p_full_from_raw_id_data(data_dir: Path) -> np.ndarray:
    """
    Reconstruct same-contract last available VWAP.
    Raw price columns after timestamp: id_1, ..., id_12, last_p
    Therefore numeric column index 12 is last_p.

    Returns
    -------
    shape: (n_days * 24,)
    ordered: day1 h0, ..., day1 h23, day2 h0, ...
    """
    folder = data_dir / "ID_DATA"
    per_hour_last_p = []
    for hour in range(24):
        fpath = _resolve_hourly_file(folder=folder, stem="prices_hourly", hour=hour)
        vals = []
        with open(fpath, "r") as f:
            # skip header
            next(f)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                nums = [float(x) for x in parts[1:]]
                if len(nums) < 13:
                    raise ValueError(f"{fpath} has fewer than " f"13 numeric columns.")
                # id_1,...,id_12,last_p
                vals.append(nums[12])
        per_hour_last_p.append(np.asarray(vals, dtype=float))

    n_days_set = {len(x) for x in per_hour_last_p}
    if len(n_days_set) != 1:
        raise ValueError(
            "Hourly price files contain inconsistent "
            f"numbers of days: {n_days_set}"
        )

    n_days = len(per_hour_last_p[0])
    last_p_panel = np.zeros((n_days, 24), dtype=float)
    for hour in range(24):
        last_p_panel[:, hour] = per_hour_last_p[hour]
    return last_p_panel.reshape(-1)


# ============================================================
# Align last_p with y_train + y_test
# ============================================================

def build_aligned_last_p(data_dir: Path, y_train: np.ndarray, y_test: np.ndarray,
                         start_index_obs: int = START_INDEX_OBS):
    """
    Align raw last_p with y_train and y_test.
    Assumes: raw data -> remove first 7 days -> train/test split
    and that train + test form the final contiguous block.
    """
    last_p_full = read_last_p_full_from_raw_id_data(data_dir)
    last_p_trimmed = last_p_full[start_index_obs:]

    n_train = len(y_train)
    n_test = len(y_test)
    n_required = n_train + n_test

    if len(last_p_trimmed) < n_required:
        raise ValueError(
            f"Not enough last_p observations.\n"
            f"Available after trimming: "
            f"{len(last_p_trimmed)}\n"
            f"Required: {n_required}"
        )

    last_p_used = last_p_trimmed[-n_required:]
    last_p_train = last_p_used[:n_train]
    last_p_test = last_p_used[n_train:]
    return (last_p_train, last_p_test)


# ============================================================
# Rolling same-hour residual bootstrap
# ============================================================

def build_naive_rolling_same_hour_bootstrap(
    y_train: np.ndarray,
    y_test: np.ndarray,
    last_p_train: np.ndarray,
    last_p_test: np.ndarray,
    window_days: int = 240,
    n_traj: int = 10000,
    seed: int = 123,
) -> np.ndarray:
    """
    Build probabilistic naive forecast analogous to your
    Lasso residual bootstrap.

    For each TEST day and delivery hour:
        1. naive point forecast = current same-contract last_p
        2. construct historical naive residuals:
               error = realized 10-D path - last_p
        3. select previous `window_days` residuals
           FOR THE SAME DELIVERY HOUR
        4. bootstrap complete 10-D residual vectors
        5. add them to current last_p

    IMPORTANT
    ---------
    As the test period progresses, realized errors from earlier
    test days enter the rolling residual window.
    For test day d, only days STRICTLY BEFORE d are used.
    This matches the sequential logic of your Lasso bootstrap
    and does not use the current/future test realization.

    Returns
    -------
    test_pred: shape (n_test_obs, 10, n_traj)
    """
    y_train = ensure_y_shape(y_train)
    y_test = ensure_y_shape(y_test)
    last_p_train = np.asarray(last_p_train, dtype=float)
    last_p_test = np.asarray(last_p_test, dtype=float)

    # --------------------------------------------------------
    # Basic consistency checks
    # --------------------------------------------------------
    if len(y_train) % 24 != 0:
        raise ValueError(f"y_train length {len(y_train)} " f"is not divisible by 24.")
    if len(y_test) % 24 != 0:
        raise ValueError(f"y_test length {len(y_test)} " f"is not divisible by 24.")
    if len(last_p_train) != len(y_train):
        raise ValueError("last_p_train length does not match y_train.")
    if len(last_p_test) != len(y_test):
        raise ValueError("last_p_test length does not match y_test.")

    n_train_days = len(y_train) // 24
    n_test_days = len(y_test) // 24

    if n_train_days < window_days:
        raise ValueError(
            f"Need at least {window_days} training days, "
            f"but only have {n_train_days}."
        )

    # --------------------------------------------------------
    # Reshape into (day, hour, path_step) and last_p -> (day, hour)
    # --------------------------------------------------------
    y_train_days = y_train.reshape(n_train_days, 24, 10)
    y_test_days = y_test.reshape(n_test_days, 24, 10)
    last_p_train_days = last_p_train.reshape(n_train_days, 24)
    last_p_test_days = last_p_test.reshape(n_test_days, 24)

    # --------------------------------------------------------
    # Combine train + test REALIZATIONS.
    # This allows yesterday's realized test residual to enter
    # today's residual pool, exactly as in your Lasso code.
    # --------------------------------------------------------
    y_all = np.concatenate([y_train_days, y_test_days], axis=0)
    last_p_all = np.concatenate([last_p_train_days, last_p_test_days], axis=0)

    # --------------------------------------------------------
    # Naive residual vectors: actual path - last_p
    # Shape: (total_days, 24, 10)
    # --------------------------------------------------------
    residual_all = y_all - last_p_all[:, :, None]

    # --------------------------------------------------------
    # Output: (test_day, hour, path_step, trajectory)
    # --------------------------------------------------------
    test_pred = np.zeros((n_test_days, 24, 10, n_traj), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # ========================================================
    # Rolling bootstrap
    # ========================================================
    for test_day in range(n_test_days):
        # Current day in combined train+test indexing
        current_day = n_train_days + test_day
        # Previous W days only
        window_start = current_day - window_days
        window_end = current_day

        if window_start < 0:
            raise RuntimeError("Bootstrap window reaches before " "available history.")

        for hour in range(24):
            # Naive point forecast for CURRENT test observation
            naive_point = last_p_test_days[test_day, hour]

            # Previous W SAME-HOUR residual vectors
            # Shape initially: (window_days, 10)
            error_pool = residual_all[window_start:window_end, hour, :]

            # Bootstrap historical DAYS
            # Exactly analogous to:
            #   a = np.random.choice(np.arange(240), size=N_traj, replace=True)
            sampled_days = rng.choice(window_days, size=n_traj, replace=True)

            # Preserve the ENTIRE 10-D error vector.
            # error_pool[sampled_days]: (N_traj, 10) -> transpose: (10, N_traj)
            error_bootstrap = error_pool[sampled_days, :].T

            # Naive probabilistic forecast
            # No centering. No scaling. No bias correction.
            # Shape: (10, N_traj)
            forecast_samples = naive_point + error_bootstrap

            test_pred[test_day, hour, :, :] = forecast_samples

    # --------------------------------------------------------
    # Flatten test day/hour: (n_test_days, 24, 10, N) -> (n_test_obs, 10, N)
    # --------------------------------------------------------
    test_pred = test_pred.reshape(n_test_days * 24, 10, n_traj)
    return test_pred


# ============================================================
# Scalar collapse
# ============================================================

def get_scalar_target_weights(idxs: np.ndarray) -> np.ndarray:
    weights = SCALAR_SUBPERIOD_MINUTES[idxs].astype(float)
    return weights / weights.sum()


def collapse_y_to_scalar_target(y: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    y = ensure_y_shape(y)
    weights = get_scalar_target_weights(idxs)
    return np.average(y[:, idxs], axis=1, weights=weights)


def collapse_pred_to_scalar_target(pred: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    pred: (n_obs, 10, n_traj)
    output: (n_obs, n_traj)
    """
    weights = get_scalar_target_weights(idxs)
    return np.average(pred[:, idxs, :], axis=1, weights=weights)


# ============================================================
# CRPS
# ============================================================

def crps_from_samples_1d(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)
    _, m = samples.shape

    # E|X-y|
    term1 = np.mean(np.abs(samples - y[:, None]), axis=1)

    # 0.5 E|X-X'|
    xs = np.sort(samples, axis=1)
    coeff = (2 * np.arange(1, m + 1) - m - 1).astype(float)
    term2 = np.sum(xs * coeff[None, :], axis=1) / (m ** 2)

    return term1 - term2


# ============================================================
# Pinball
# ============================================================

def pinball_loss(samples: np.ndarray, y: np.ndarray, q: float) -> float:
    q_pred = np.quantile(samples, q, axis=1)
    err = y - q_pred
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.mean(loss))


# ============================================================
# Evaluation
# ============================================================

def evaluate_scalar_prob_forecast(samples: np.ndarray, y: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=float)
    y = np.asarray(y, dtype=float)

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


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 85)
    print("NAIVE ROLLING SAME-HOUR RESIDUAL BOOTSTRAP")
    print("=" * 85)

    def read_raw_panel(data_dir: Path, cols) -> np.ndarray:
        """(n_days, 24, len(cols)) from prices_hourly_XX; numeric cols: id_1..id_12 = 0..11, last_p = 12."""
        folder = data_dir / "ID_DATA"
        per_hour = []
        for hour in range(24):
            fpath = _resolve_hourly_file(folder=folder, stem="prices_hourly", hour=hour)
            with open(fpath) as f:
                rows = [[float(e) for e in line.strip().split(",")[1:]] for line in f.readlines()[1:]]
            per_hour.append(np.asarray(rows)[:, cols])
        return np.stack(per_hour, axis=1)

    path_full = read_raw_panel(CGM_DIR, list(range(2, 12)))[7:]  # id_3..id_12, drop 7 burn-in days
    last_p_full = read_raw_panel(CGM_DIR, [12])[7:, :, 0]

    y_test = ensure_y_shape(np.load(Y_TEST_FILE))
    n_test_days = len(y_test) // 24
    assert np.allclose(path_full[-n_test_days:].reshape(-1, 10), y_test, atol=1e-3), "raw/test misaligned"

    # contiguous history: everything before the test block acts as "train" for the pool
    y_hist = path_full[:-n_test_days].reshape(-1, 10)
    lp_hist = last_p_full[:-n_test_days].reshape(-1)
    lp_test = last_p_full[-n_test_days:].reshape(-1)

    naive_pred = build_naive_rolling_same_hour_bootstrap(
        y_train=y_hist, y_test=y_test, last_p_train=lp_hist, last_p_test=lp_test,
        window_days=BOOTSTRAP_WINDOW_DAYS, n_traj=N_TRAJ, seed=SEED)

    print(f"\nProbabilistic naive shape: " f"{naive_pred.shape}")

    # ========================================================
    # Save full probabilistic path
    # ========================================================
    pred_path = OUT_DIR / (
        "naive_same_contract_last_p_"
        f"rolling_{BOOTSTRAP_WINDOW_DAYS}d_"
        "same_hour_bootstrap.npy"
    )
    np.save(pred_path, naive_pred.astype(np.float32))
    print(f"\nSaved full path ensemble:\n" f"{pred_path}")

    # ========================================================
    # Scalar ID1 / ID2 / ID3 evaluation
    # ========================================================
    rows = []

    for (target_name, idxs) in SCALAR_TARGET_SPECS.items():
        print("\n" + "-" * 85)
        print(target_name)
        print("-" * 85)

        # Actual scalar target
        y_scalar = collapse_y_to_scalar_target(y_test, idxs)

        # Forecast scalar ensemble
        scalar_samples = collapse_pred_to_scalar_target(naive_pred, idxs)

        # Metrics
        metrics = evaluate_scalar_prob_forecast(samples=scalar_samples, y=y_scalar)

        # Original deterministic last_p MAE
        deterministic_error = lp_test - y_scalar
        deterministic_mae = float(np.mean(np.abs(deterministic_error)))

        # Print
        print(f"Deterministic last_p MAE : " f"{deterministic_mae:.6f}")
        print(f"Bootstrap median MAE      : " f"{metrics['mae_median']:.6f}")
        print(f"Bootstrap mean MAE        : " f"{metrics['mae_mean']:.6f}")
        print(f"CRPS                      : " f"{metrics['crps']:.6f}")
        print(f"80% coverage              : " f"{metrics['coverage_80']:.4f}")
        print(f"80% width                 : " f"{metrics['width_80']:.4f}")
        print(f"60% coverage              : " f"{metrics['coverage_60']:.4f}")
        print(f"60% width                 : " f"{metrics['width_60']:.4f}")

        # Save scalar forecast samples
        scalar_path = OUT_DIR / (
            f"{target_name}_"
            f"rolling_{BOOTSTRAP_WINDOW_DAYS}d_"
            "same_hour_bootstrap.npy"
        )
        np.save(scalar_path, scalar_samples.astype(np.float32))

        # Result row
        rows.append({
            "target": target_name,
            "model": (
                "naive_same_contract_last_p_"
                f"rolling_{BOOTSTRAP_WINDOW_DAYS}d_"
                "same_hour_bootstrap"
            ),
            "bootstrap_window_days": BOOTSTRAP_WINDOW_DAYS,
            "n_traj": N_TRAJ,
            "n_obs": len(y_scalar),
            "deterministic_last_p_mae": deterministic_mae,
            **metrics,
        })

    # ========================================================
    # Save metrics
    # ========================================================
    results_df = pd.DataFrame(rows)
    results_path = OUT_DIR / (
        "naive_probabilistic_scalar_metrics_"
        f"rolling_{BOOTSTRAP_WINDOW_DAYS}d_"
        "same_hour.csv"
    )
    results_df.to_csv(results_path, index=False)

    # ========================================================
    # Final table
    # ========================================================
    print("\n")
    print("=" * 115)
    print("FINAL RESULTS")
    print("=" * 115)

    display_cols = [
        "target",
        "deterministic_last_p_mae",
        "mae_median", "mae_mean",
        "crps",
        "coverage_80", "width_80",
        "coverage_60", "width_60",
        "pinball_q10", "pinball_q50", "pinball_q90",
    ]

    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print(f"\nMetrics saved to:\n" f"{results_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()