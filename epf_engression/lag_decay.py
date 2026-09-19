"""
Lag-decay diagnostic for results.tex, Step 3 - extended across all 10 ID-path
horizons (id_3 ... id_12)

Outputs:
    lag_decay_all_horizons.csv                  - correlation(lag_i, target)
                                                   for each of the 10 target
                                                   columns, lag_i in 0..164
    lag_decay_all_horizons.png                   - 2x5 grid, one panel/horizon
    lag_decay_near_vs_periodic_by_horizon.png    - summary comparison plot
    lag_decay_summary.csv                        - per-horizon near-term vs
                                                   periodic signal strength
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engression_experiments_old import (
    ExperimentConfig,
    build_base_tables,
    build_raw_feature_dataframe,
    normalize_dataframe_clean,
)

CFG = ExperimentConfig()

TARGET_COLS = [f"id_{i}" for i in range(3, 13)]  # h1 (id_3) ... h10 (id_12)
N_LAGS = 165
DAILY_PERIOD = 24  # hour-products per day


def compute_lag_decay(ts_values: np.ndarray, ts_cols: list, target_col: str, cfg) -> np.ndarray:
    target_idx = ts_cols.index(target_col)
    target_series = ts_values[:, target_idx]
    n = ts_values.shape[0]
    index_all = np.arange(n)

    correlations = np.full(N_LAGS, np.nan)
    for lag_i in range(N_LAGS):
        src_idx = index_all - cfg.lead - lag_i
        valid = src_idx >= 0
        correlations[lag_i] = np.corrcoef(
            target_series[valid], ts_values[src_idx[valid], target_idx]
        )[0, 1]
    return correlations


def summarize_decay(corr: np.ndarray):
    """Split each horizon's decay curve into a 'near-term' block (before the
    first daily echo) and the strength of the periodic (multiples-of-24)
    component, so horizons can be compared on both axes."""
    near_term_mask = np.arange(N_LAGS) < DAILY_PERIOD
    near_term_mean_abs = np.abs(corr[near_term_mask]).mean()

    # Sample |correlation| in a small window around each daily multiple
    # (24, 48, 72, ...), robust to the local peak not landing exactly on it.
    daily_multiples = np.arange(DAILY_PERIOD, N_LAGS, DAILY_PERIOD)
    periodic_vals = []
    for m in daily_multiples:
        window = corr[max(0, m - 2): min(N_LAGS, m + 3)]
        periodic_vals.append(np.max(np.abs(window)))
    periodic_mean_abs = float(np.mean(periodic_vals))

    return float(near_term_mean_abs), periodic_mean_abs


def main():
    pred_combine_df = build_base_tables(CFG)
    raw_feature_df = build_raw_feature_dataframe(pred_combine_df)
    pred_combine_norm_df, data_id_mu, data_id_sigma, norm_stats = normalize_dataframe_clean(
        raw_feature_df, CFG
    )

    # Same ts_cols exclusion logic as build_cgm_like_arrays in the training
    # script, so lag positions here line up with the flattened input exactly.
    time_cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy"]
    excluded_cols = set(time_cols + ["weekday"])
    ts_cols = [c for c in pred_combine_norm_df.columns if c not in excluded_cols]
    ts_values = pred_combine_norm_df[ts_cols].values.astype(np.float32)

    all_results = {"lag": np.arange(N_LAGS)}
    summary_rows = []

    for h_i, col in enumerate(TARGET_COLS, start=1):
        corr = compute_lag_decay(ts_values, ts_cols, col, CFG)
        all_results[col] = corr
        near_term, periodic = summarize_decay(corr)
        summary_rows.append({
            "horizon": h_i,
            "target_col": col,
            "near_term_mean_abs_corr": near_term,
            "periodic_mean_abs_corr": periodic,
            "periodic_minus_near_term": periodic - near_term,
        })
        print(f"{col} (h{h_i}): near-term |corr|={near_term:.3f}, "
              f"periodic |corr|={periodic:.3f}")

    decay_df = pd.DataFrame(all_results)
    decay_df.to_csv("lag_decay_all_horizons.csv", index=False)
    print("\nSaved lag_decay_all_horizons.csv")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("lag_decay_summary.csv", index=False)
    print("Saved lag_decay_summary.csv")
    print("\n", summary_df.to_string(index=False))

    # --- Small-multiple grid: one panel per horizon -------------------------
    fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True, sharey=True)
    for h_i, col in enumerate(TARGET_COLS, start=1):
        ax = axes.flat[h_i - 1]
        ax.plot(decay_df["lag"], decay_df[col], lw=1.0)
        for m in range(DAILY_PERIOD, N_LAGS, DAILY_PERIOD):
            ax.axvline(m, color="grey", lw=0.5, ls="--", alpha=0.5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"h{h_i} ({col})", fontsize=10)
        if h_i > 5:
            ax.set_xlabel("Lag (hour-products)")
        if h_i in (1, 6):
            ax.set_ylabel("Correlation")

    fig.suptitle("Autocorrelation decay across the 165-lag input window, by horizon", y=1.02)
    fig.tight_layout()
    fig.savefig("lag_decay_all_horizons.png", dpi=200, bbox_inches="tight")
    print("Saved lag_decay_all_horizons.png")

    # --- Summary plot: near-term vs periodic strength, by horizon -----------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(summary_df["horizon"], summary_df["near_term_mean_abs_corr"],
              marker="o", label="Near-term (lag < 24)")
    ax2.plot(summary_df["horizon"], summary_df["periodic_mean_abs_corr"],
              marker="o", label="Periodic (daily echoes)")
    ax2.set_xlabel("Horizon (h1 = id_3 ... h10 = id_12)")
    ax2.set_ylabel("Mean |correlation|")
    ax2.set_title("Near-term vs periodic signal strength, by horizon")
    ax2.legend()
    ax2.set_xticks(summary_df["horizon"])
    fig2.tight_layout()
    fig2.savefig("lag_decay_near_vs_periodic_by_horizon.png", dpi=200)
    print("Saved lag_decay_near_vs_periodic_by_horizon.png")


if __name__ == "__main__":
    main()