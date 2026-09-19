"""
Unconditional target variance by horizon (h1 = id_3 ... h10 = id_12), computed
for BOTH target transforms side by side on the metDesk / recent-window pipeline:

    resid_last : y = id_h - last_p   (the locked model's target)
    raw        : y = id_h            (the untransformed target; also what CGM uses)

Output
------
    target_variance_by_horizon_raw_vs_resid.csv
    target_variance_by_horizon_raw_vs_resid.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engression_experiments_old import (
    ExperimentConfig,
    build_base_tables,
    build_raw_feature_dataframe,
    normalize_dataframe_clean,
    build_cgm_like_arrays,
    build_feature_dataframe,
    build_anchor,
    transform_target,
    make_splits,
)

CFG = ExperimentConfig()

TARGET_MODES = ["resid_last", "raw"]
TARGET_COLS = [f"id_{i}" for i in range(3, 13)]  # h1 (id_3) ... h10 (id_12)


def variance_by_horizon(target: np.ndarray, idx_trainval, idx_test, mode: str) -> pd.DataFrame:
    rows = []
    for h_i, col in enumerate(TARGET_COLS, start=1):
        rows.append({
            "target_mode": mode,
            "horizon": h_i,
            "target_col": col,
            "var_all": float(target[:, h_i - 1].var()),
            "var_in_sample": float(target[idx_trainval, h_i - 1].var()),
            "var_out_of_sample": float(target[idx_test, h_i - 1].var()),
        })
    return pd.DataFrame(rows)


def main():
    # --- shared pipeline, built once ---------------------------------------- #
    pred_combine_df = build_base_tables(CFG)
    raw_feature_df = build_raw_feature_dataframe(pred_combine_df)
    pred_combine_norm_df, data_id_mu, data_id_sigma, norm_stats = normalize_dataframe_clean(
        raw_feature_df, CFG
    )
    input_ts, input_std, input_all, input_weekday, output_norm = build_cgm_like_arrays(
        pred_combine_norm_df, CFG
    )
    feature_df = build_feature_dataframe(
        pred_combine_norm_df, input_all, input_std, input_weekday, output_norm, CFG
    )

    idx_train, idx_val, idx_test = make_splits(len(output_norm), CFG)
    idx_trainval = np.concatenate([idx_train, idx_val])

    # --- one variance profile per target transform -------------------------- #
    frames = []
    for mode in TARGET_MODES:
        anchor = build_anchor(feature_df, mode)              # None for "raw", last_p for "resid_last"
        target = transform_target(output_norm, anchor, mode)  # (n, 10)
        df = variance_by_horizon(target, idx_trainval, idx_test, mode)
        frames.append(df)

        v = df["var_all"].values
        collapse = v[0] / v[-1] if v[-1] > 0 else np.inf
        mono = bool(np.all(np.diff(v) <= 0))
        print(f"\n[{mode}] var_all by horizon (h1->h10):")
        for _, r in df.iterrows():
            print(f"  h{int(r.horizon):>2} ({r.target_col}): "
                  f"var_all={r.var_all:.4f}  IS={r.var_in_sample:.4f}  OOS={r.var_out_of_sample:.4f}")
        print(f"  monotonically decreasing h1->h10? {mono}")
        print(f"  collapse factor var[h1]/var[h10] = {collapse:.1f}x")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv("target_variance_by_horizon_raw_vs_resid.csv", index=False)
    print("\nSaved target_variance_by_horizon_raw_vs_resid.csv")

    # --- plot: var_all vs horizon, both transforms -------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in TARGET_MODES:
        sub = combined[combined["target_mode"] == mode]
        ax.plot(sub["horizon"], sub["var_all"], marker="o", label=f"{mode} target")
    ax.axvspan(8.5, 10.5, color="orange", alpha=0.08)  # near-origin region h9-h10
    ax.set_xlabel("Horizon (h1 = id_3 ... h10 = id_12)")
    ax.set_ylabel("unconditional variance (normalized units)")
    ax.set_title("Target variance by horizon: resid_last vs raw")
    ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.legend()
    fig.tight_layout()
    fig.savefig("target_variance_by_horizon_raw_vs_resid.png", dpi=200)
    print("Saved target_variance_by_horizon_raw_vs_resid.png")


if __name__ == "__main__":
    main()