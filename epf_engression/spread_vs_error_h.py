"""
Spread-skill (predicted-spread vs realized-error) check by horizon, across the
generative models presented in the thesis: CGM and the main Engression variants
(heteroscedastic resid_last = locked model, and the untransformed raw-target
Engression). The full-vs-compact input comparison is deliberately left out here,
since that is covered in the input-representation subsection.

Output
------
    spread_vs_error_by_horizon_models_old.csv
    spread_vs_error_by_horizon_models_old.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths. Fill / confirm these against the runs you actually have.
# --------------------------------------------------------------------------- #
ENGR_OUT = Path(
    r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression\engression_outputs_experiments"
)
CGM_DIR = Path(r"C:\Users\Ismas\PycharmProjects\epf_engression\epf_engression")


LOCKED_RUN = (
    "compact__compact_v1__resid_last__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256"
)

# Shared realized targets: any Engression run's y_test.npy holds the true
# id_3..id_12 test prices (de-normalized, target-transform inverted before
# saving, so identical regardless of target_mode).
SHARED_Y_TEST = ENGR_OUT / LOCKED_RUN / "y_test.npy"


def engr(run_name: str) -> dict:
    """An Engression run: pred.npy + y_test.npy inside its run dir."""
    d = ENGR_OUT / run_name
    return {"pred": d / "pred.npy", "y_test": d / "y_test.npy"}


RUNS = {
    # Locked model: heteroscedastic Engression, resid_last target.
    "Engression (hetero, resid_last)": engr(LOCKED_RUN),
    # Untransformed target, the raw sibling of the locked run.
    "Engression (hetero, raw)": engr(
        "compact__compact_v1__raw__L2_H128_N32_LR0.0001_BS1024_E2000_ENS10_HETERO_True_HS_256"
    ),
    # CGM (Chen et al.): its own predictions, but the shared Engression targets.
    "CGM": {
        "pred": CGM_DIR / "pred_cgm_esloss.npy",
        "y_test": SHARED_Y_TEST,
    },
}

TARGET_COLS = [f"id_{i}" for i in range(3, 13)]  # h1 (id_3) ... h10 (id_12)


def load_run(spec: dict):
    """Load one run's samples and realized targets, with shape checks."""
    pred = np.load(spec["pred"])      # expected (n_test, 10, n_samples)
    y_test = np.load(spec["y_test"])  # expected (n_test, 10)

    # A saved array carrying the CGM best_step in front would be (n, 11, ns);
    # trim it defensively so only the 10 price horizons are used.
    if pred.ndim == 3 and pred.shape[1] == 11:
        pred = pred[:, 1:, :]
    if y_test.ndim == 2 and y_test.shape[1] == 11:
        y_test = y_test[:, 1:]

    if pred.ndim != 3 or pred.shape[1] != 10:
        raise ValueError(f"pred should be (n,10,ns), got {pred.shape} for {spec['pred']}")
    if y_test.ndim != 2 or y_test.shape[1] != 10:
        raise ValueError(f"y_test should be (n,10), got {y_test.shape} for {spec['y_test']}")
    if pred.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"n_test mismatch: pred {pred.shape[0]} vs y_test {y_test.shape[0]} "
            f"({spec['pred']}). pred and y_test must come from the same run/data build."
        )
    return pred, y_test


def spread_vs_error(pred: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    n_test, n_horizons, n_samples = pred.shape
    rows = []
    for h_i in range(n_horizons):
        ens_mean = pred[:, h_i, :].mean(axis=1)               # (n_test,)
        predicted_var = pred[:, h_i, :].var(axis=1, ddof=1)   # (n_test,)
        squared_error = (ens_mean - y_test[:, h_i]) ** 2       # (n_test,)

        avg_predicted_var = float(predicted_var.mean())
        avg_squared_error = float(squared_error.mean())
        ratio = avg_predicted_var / avg_squared_error if avg_squared_error > 0 else np.nan

        rows.append({
            "horizon": h_i + 1,
            "target_col": TARGET_COLS[h_i],
            "avg_predicted_var": avg_predicted_var,
            "avg_squared_error": avg_squared_error,
            "predicted_std": np.sqrt(avg_predicted_var),
            "realized_rmse": np.sqrt(avg_squared_error),
            "ratio_var_to_error": ratio,
        })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for label, spec in RUNS.items():
        if not Path(spec["pred"]).exists():
            print(f"WARNING: pred missing, skipping '{label}': {spec['pred']}")
            continue
        if not Path(spec["y_test"]).exists():
            print(f"WARNING: y_test missing, skipping '{label}': {spec['y_test']}")
            continue
        pred, y_test = load_run(spec)
        print(f"\n{label}: pred {pred.shape}, y_test {y_test.shape}")
        df = spread_vs_error(pred, y_test)
        df.insert(0, "model", label)
        all_results.append(df)
        print(df[["horizon", "target_col", "predicted_std",
                  "realized_rmse", "ratio_var_to_error"]].to_string(index=False))
        h10 = df.loc[df["horizon"] == 10, "ratio_var_to_error"].values[0]
        print(f"  -> h10 (id_12) spread-skill ratio = {h10:.3f}"
              f"  ({'OVERdispersed' if h10 > 1 else 'underdispersed'})")

    if not all_results:
        print("\nNo runs found - fill in the RUNS paths above.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv("spread_vs_error_by_horizon_models_old.csv", index=False)
    print("\nSaved spread_vs_error_by_horizon_models_old.csv")

    # --- Plots -------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    for label in RUNS:
        sub = combined[combined["model"] == label]
        if len(sub):
            ax.plot(sub["horizon"], sub["ratio_var_to_error"], marker="o", label=label)
    ax.axhline(1.0, color="grey", lw=1, ls="--", label="calibrated (ratio = 1)")
    ax.axvspan(8.5, 10.5, color="orange", alpha=0.08)  # near-origin region h9-h10
    ax.set_xlabel("Horizon (h1 = id_3 ... h10 = id_12)")
    ax.set_ylabel("avg predicted var / avg squared error")
    ax.set_title("Spread-skill ratio by horizon\n(> 1 = overdispersed)")
    ax.set_yscale("log")
    ax.set_xticks(range(1, 11))
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for label in RUNS:
        sub = combined[combined["model"] == label]
        if len(sub):
            line, = ax2.plot(sub["horizon"], sub["predicted_std"], marker="o", ls="-",
                             label=f"{label} - predicted std")
            ax2.plot(sub["horizon"], sub["realized_rmse"], marker="x", ls="--",
                     color=line.get_color(), label=f"{label} - realized RMSE")
    ax2.set_xlabel("Horizon (h1 = id_3 ... h10 = id_12)")
    ax2.set_ylabel("EUR/MWh (per model - levels not cross-comparable)")
    ax2.set_title("Predicted spread vs realized error, by horizon")
    ax2.set_yscale("log")
    ax2.set_xticks(range(1, 11))
    ax2.legend(fontsize=6)

    fig.tight_layout()
    fig.savefig("spread_vs_error_by_horizon_models_old.png", dpi=200)
    print("Saved spread_vs_error_by_horizon_models_old.png")


if __name__ == "__main__":
    main()