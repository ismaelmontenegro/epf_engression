import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from engression_module import engression


# ============================================================
# Configuration
# ============================================================

@dataclass
class ExperimentConfig:
    data_dir: str = r"C:/Users/Ismas/PycharmProjects/epf_engression/epf_engression"
    out_subdir: str = "engression_outputs_idx3_experiments"
    start_date: str = "2017-06-14"

    n_days: int = 837
    n_hours: int = 24
    test_days: int = 200
    lead: int = 4
    start_index: int = 168
    add_id_others: bool = True

    # Dedicated scalar target experiment.
    # Targets are 10-step proxies derived from the same path columns used by
    # the path-then-collapse evaluation: [id_3, id_4, ..., id_12].
    target_name: str = "id1_p"  # "id3_p", "id2_p", "id1_p"
    target_mode: str = "resid_last"          # "raw", "resid_last", "resid_da", "resid_id3"
    benchmark_mode: str = "compact"   # "compact" or "full"
    feature_set: str = "compact_v1"   # "compact_v1", "compact_v2", "compact_v3", "compact_v4", "all_clean_no_weekday"

    num_layers: int = 2
    hidden_dim: int = 128
    noise_dim: int = 32
    lr: float = 1e-4
    num_epochs: int = 200
    batch_size: int = 1024
    n_samples_test: int = 1000
    n_ensemble: int = 10
    standardize_inside_engression: bool = False
    save_normalized_outputs: bool = True
    seed_base: int = 1000

    hetero_noise: bool = True
    scale_hidden_dim: int = 256


CFG = ExperimentConfig()
DATA_DIR = Path(CFG.data_dir)
OUT_DIR = DATA_DIR / CFG.out_subdir
OUT_DIR.mkdir(exist_ok=True, parents=True)

START_DATE = datetime.strptime(CFG.start_date, "%Y-%m-%d")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_COLS = [CFG.target_name]

INDEX_PATH_COLS = [f"id_{i}" for i in range(3, 13)]
INDEX_PROXY_MINUTES = np.array([15, 15, 15, 15, 15, 15, 15, 15, 15, 10], dtype=np.float64)
INDEX_PROXY_SPECS = {
    "id1_p": np.array([0, 1, 2, 3], dtype=int),
    "id2_p": np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int),
    "id3_p": np.arange(10, dtype=int),
}


# ============================================================
# Data loading helpers
# ============================================================

def read_hourly_id_file(folder: Path, prefix: str, n_keep: int, n_days: int, n_hours: int):
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
            (datetime.strptime(date_string, "%Y%m%d") - START_DATE).days
            for date_string in date_str
        ]

        arr[:, n_hour, 0] = np.array(date_diff) + 1
        arr[:, n_hour, 1] = n_hour
        arr[:, n_hour, 2:] = kept

    return arr


def read_exog_series(filepath: Path):
    with open(filepath) as f:
        lines = f.readlines()
    return np.array([float(line.strip()) for line in lines], dtype=np.float64)


def time_weighted_proxy(prices_10_step: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    """
    Collapse the 10-step path target to the same scalar proxy used in the
    path-then-collapse evaluation.
    """
    weights = INDEX_PROXY_MINUTES[idxs]
    weights = weights / weights.sum()
    return np.average(prices_10_step[:, idxs], axis=1, weights=weights)


# ============================================================
# Base table construction
# ============================================================

def build_base_tables(cfg: ExperimentConfig) -> pd.DataFrame:
    pricesintra = read_hourly_id_file(
        folder=Path(cfg.data_dir) / "ID_DATA",
        prefix="prices_hourly",
        n_keep=13,
        n_days=cfg.n_days,
        n_hours=cfg.n_hours,
    )
    pricesintra_flat = pricesintra.reshape(-1, pricesintra.shape[-1])

    # Scalar index targets use the exact 10-step path proxy convention:
    # prices_10_step columns are [id_3, id_4, ..., id_12].
    prices_10_step = pricesintra_flat[:, 4:14]
    id3_proxy = time_weighted_proxy(prices_10_step, INDEX_PROXY_SPECS["id3_p"])
    id2_proxy = time_weighted_proxy(prices_10_step, INDEX_PROXY_SPECS["id2_p"])
    id1_proxy = time_weighted_proxy(prices_10_step, INDEX_PROXY_SPECS["id1_p"])

    id_pred = np.zeros((cfg.n_days * cfg.n_hours, 18), dtype=np.float64)
    id_pred[:, :2] = pricesintra_flat[:, :2]
    id_pred[:, 2] = id3_proxy
    id_pred[:, 3] = id2_proxy
    id_pred[:, 4] = id1_proxy
    id_pred[:, 5:] = pricesintra_flat[:, 2:15]

    id_pred_df = pd.DataFrame(
        id_pred,
        columns=[
            "day", "hour",
            "id3_p", "id2_p", "id1_p",
            "id_1", "id_2", "id_3", "id_4", "id_5", "id_6",
            "id_7", "id_8", "id_9", "id_10", "id_11", "id_12",
            "last_p",
        ],
    )

    # 9 columns: day, hour, DA, wind pred/real, load pred/real
    exog_pred = np.zeros((cfg.n_days * cfg.n_hours, 7), dtype=np.float64)

    with open(Path(cfg.data_dir) / "EXOG_DATA" / "Day_Ahead_Epex.csv") as f:
        lines = f.readlines()

    date_str = [str(line.split(";")[0].strip()) for line in lines]
    hour = [int(line.split(";")[1].strip()) for line in lines]
    da_price = [float(line.split(";")[2].strip()) for line in lines]
    date_diff = [
        (datetime.strptime(date_string, "%Y%m%d") - START_DATE).days
        for date_string in date_str
    ]

    exog_pred[:, 0] = np.array(date_diff)
    exog_pred[:, 1] = np.array(hour) - 1
    exog_pred[:, 2] = np.array(da_price)

    woff = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_wind_offshore.csv")
    won = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_wind_onshore.csv")
    woffreal = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_wind_offshore_real.csv")
    wonreal = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_wind_onshore_real.csv")

    #pv = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_pv.csv")
    #pvreal = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_pv_real.csv")

    load_da = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_load_da.csv")
    load_real = read_exog_series(Path(cfg.data_dir) / "EXOG_DATA" / "final_load_real.csv")

    exog_pred[:, 3] = won + woff
    exog_pred[:, 4] = wonreal + woffreal
    #exog_pred[:, 5] = pv
    #exog_pred[:, 6] = pvreal
    exog_pred[:, 5] = load_da
    exog_pred[:, 6] = load_real

    exog_pred_df = pd.DataFrame(
        exog_pred,
        columns=[
            "day", "hour", "da_p",
            "w_pred", "w_real",
            #"pv_pred", "pv_real",
            "l_pred", "l_real",
        ],
    )

    return pd.merge(exog_pred_df, id_pred_df, how="outer", on=["day", "hour"])


# ============================================================
# Splits
# ============================================================

def get_split_sizes(cfg: ExperimentConfig) -> Tuple[int, int, int]:
    usable_days = cfg.n_days - (cfg.start_index // cfg.n_hours)
    trainval_len = (cfg.n_days - cfg.test_days - 7) * cfg.n_hours
    val_size = int(0.2 * trainval_len)
    train_size = trainval_len - val_size
    test_size = usable_days * cfg.n_hours - trainval_len
    return train_size, val_size, test_size


def make_splits(n_obs: int, cfg: ExperimentConfig):
    trainval_len = (cfg.n_days - cfg.test_days - 7) * cfg.n_hours
    val_size = int(0.2 * trainval_len)
    train_size = trainval_len - val_size

    idx_train = np.arange(0, train_size)
    idx_val = np.arange(train_size, trainval_len)
    idx_test = np.arange(trainval_len, n_obs)

    return idx_train, idx_val, idx_test


# ============================================================
# Feature engineering and normalization
# ============================================================

def build_raw_feature_dataframe(pred_combine_df: pd.DataFrame) -> pd.DataFrame:
    df = pred_combine_df.copy()

    result_date = [START_DATE + timedelta(days=int(diff)) for diff in df["day"].values]
    df.insert(0, "date", result_date)
    df.insert(1, "weekday", np.array([d.weekday() + 1 for d in result_date]).astype(int))
    df.insert(2, "weekofyear", np.array([d.strftime("%V") for d in result_date]).astype(int))
    df.insert(3, "dayofyear", np.array([d.strftime("%j") for d in result_date]).astype(int))

    sin_hod = np.sin((df["hour"].values / 24) * 2 * np.pi)
    cos_hod = np.cos((df["hour"].values / 24) * 2 * np.pi)
    sin_doy = np.sin((df["dayofyear"].values / 365) * 2 * np.pi)
    cos_doy = np.cos((df["dayofyear"].values / 365) * 2 * np.pi)

    df.insert(1, "cos_doy", cos_doy)
    df.insert(1, "sin_doy", sin_doy)
    df.insert(1, "cos_hod", cos_hod)
    df.insert(1, "sin_hod", sin_hod)

    return df


def normalize_dataframe_clean(raw_df: pd.DataFrame, cfg: ExperimentConfig):
    df = raw_df.copy()

    train_size, _, _ = get_split_sizes(cfg)
    train_start = cfg.start_index
    train_stop = cfg.start_index + train_size

    exog_cols = [
        "da_p",
        "w_pred", "w_real",
        #"pv_pred", "pv_real",
        "l_pred", "l_real",
    ]

    id_cols = [
        "id3_p", "id2_p", "id1_p",
        "id_1", "id_2", "id_3", "id_4", "id_5", "id_6",
        "id_7", "id_8", "id_9", "id_10", "id_11", "id_12",
        "last_p",
    ]

    norm_stats = {
        "exog": {},
        "id_joint": {},
        "target_cols": TARGET_COLS,
        "target_definition": {
            "type": "10_step_time_weighted_proxy",
            "source_cols": INDEX_PATH_COLS,
            "minutes": INDEX_PROXY_MINUTES.tolist(),
            "specs": {k: v.tolist() for k, v in INDEX_PROXY_SPECS.items()},
        },
    }

    for col in exog_cols:
        train_vals = df.iloc[train_start:train_stop][col].values.astype(float)
        mu = train_vals.mean()
        sigma = train_vals.std()
        if sigma == 0:
            sigma = 1.0

        df[col] = (df[col].values.astype(float) - mu) / sigma
        norm_stats["exog"][col] = {"mu": float(mu), "sigma": float(sigma)}

    train_id = df.iloc[train_start:train_stop][id_cols].values.astype(float)
    data_id_mu = train_id.mean()
    data_id_sigma = train_id.std()
    if data_id_sigma == 0:
        data_id_sigma = 1.0

    df[id_cols] = (df[id_cols].values.astype(float) - data_id_mu) / data_id_sigma
    norm_stats["id_joint"] = {
        "mu": float(data_id_mu),
        "sigma": float(data_id_sigma),
        "cols": id_cols,
    }

    df["id_std"] = np.std(
        df[
            [
                "id_3", "id_4", "id_5", "id_6", "id_7", "id_8",
                "id_9", "id_10", "id_11", "id_12", "last_p",
            ]
        ].values,
        axis=1,
    )

    df = df.drop(columns=["day", "date", "hour", "weekofyear", "dayofyear"])
    return df, data_id_mu, data_id_sigma, norm_stats


def build_cgm_like_arrays(pred_combine_norm_df: pd.DataFrame, cfg: ExperimentConfig):
    data_len = pred_combine_norm_df.shape[0]
    index_all = np.arange(data_len)

    time_cols = ["sin_hod", "cos_hod", "sin_doy", "cos_doy"]
    excluded_cols = set(time_cols + ["weekday"])

    ts_cols = [c for c in pred_combine_norm_df.columns if c not in excluded_cols]
    ts_values = pred_combine_norm_df[ts_cols].values.astype(np.float32)

    n_ts_features = len(ts_cols)
    input_ts = np.zeros((data_len, n_ts_features, 165), dtype=np.float32)

    for i in range(165):
        input_ts[:, :, i] = ts_values[index_all - cfg.lead - i, :]

    id_std_idx = ts_cols.index("id_std")
    input_std = input_ts[:, id_std_idx, :]

    input_weekday = pred_combine_norm_df["weekday"].values.astype(int)
    output_norm = pred_combine_norm_df[TARGET_COLS].values.astype(np.float32)

    input_ts = input_ts[cfg.start_index:, :, :]
    input_std = input_std[cfg.start_index:, :]
    input_weekday = input_weekday[cfg.start_index:]
    output_norm = output_norm[cfg.start_index:, :]

    input_all = None

    return input_ts, input_std, input_all, input_weekday, output_norm


def build_feature_dataframe(
    pred_combine_norm_df: pd.DataFrame,
    input_all,
    input_std: np.ndarray,
    input_weekday: np.ndarray,
    output_norm: np.ndarray,
    cfg: ExperimentConfig,
):
    n = output_norm.shape[0]
    source = pred_combine_norm_df.reset_index(drop=True)
    base_trim = source.iloc[cfg.start_index:].reset_index(drop=True)

    idx = np.arange(cfg.start_index, cfg.start_index + n)

    def lag(col: str, k: int) -> np.ndarray:
        return source[col].values[idx - k]

    def hist_matrix(col: str, start_lag: int, length: int) -> np.ndarray:
        return np.column_stack([lag(col, start_lag + j) for j in range(length)])

    feat = pd.DataFrame(index=np.arange(n))

    # Time
    feat["sin_hod"] = base_trim["sin_hod"].values
    feat["cos_hod"] = base_trim["cos_hod"].values
    feat["sin_doy"] = base_trim["sin_doy"].values
    feat["cos_doy"] = base_trim["cos_doy"].values
    feat["weekday"] = input_weekday

    # Forecast-origin variables
    feat["da_p"] = base_trim["da_p"].values
    feat["w_pred"] = base_trim["w_pred"].values
    #feat["pv_pred"] = base_trim["pv_pred"].values
    feat["l_pred"] = base_trim["l_pred"].values

    # Actuals only from latest safely available point
    feat["w_real_lag4"] = lag("w_real", cfg.lead)
    #feat["pv_real_lag4"] = lag("pv_real", cfg.lead)
    feat["l_real_lag4"] = lag("l_real", cfg.lead)

    # Current actual columns are kept only for feature-set experiments.
    # Be careful: depending on data availability assumptions, these may be leaky.
    feat["w_real"] = base_trim["w_real"].values
    #feat["pv_real"] = base_trim["pv_real"].values
    feat["l_real"] = base_trim["l_real"].values

    # Recent intraday state
    feat["id3_p_lag4"] = lag("id3_p", cfg.lead)
    feat["last_p"] = base_trim["last_p"].values
    feat["id_std"] = lag("id_std", cfg.lead)

    # Price-state features
    feat["spread_last_da"] = feat["last_p"] - feat["da_p"]
    feat["spread_last_id3_lag4"] = feat["last_p"] - feat["id3_p_lag4"]
    feat["neg_last"] = (feat["last_p"] < 0).astype(float)

    # Forecast errors, in standardized feature units
    feat["w_error_lag4"] = feat["w_real_lag4"] - feat["w_pred"]
    #feat["pv_error_lag4"] = feat["pv_real_lag4"] - feat["pv_pred"]
    feat["l_error_lag4"] = feat["l_real_lag4"] - feat["l_pred"]

    feat["abs_w_error_lag4"] = np.abs(feat["w_error_lag4"])
    #feat["abs_pv_error_lag4"] = np.abs(feat["pv_error_lag4"])
    feat["abs_l_error_lag4"] = np.abs(feat["l_error_lag4"])

    # Residual-load-style derived features, including wind and PV consistently
    feat["res_load_pred"] = feat["l_pred"] - feat["w_pred"]
    feat["res_load_real"] = feat["l_real"] - feat["w_real"]
    feat["res_load_real_lag4"] = (
        feat["l_real_lag4"] - feat["w_real_lag4"]
    )
    feat["res_load_error_lag4"] = feat["res_load_real_lag4"] - feat["res_load_pred"]

    # Lagged forecast/state variables
    feat["da_p_lag1"] = lag("da_p", 1)
    feat["w_pred_lag1"] = lag("w_pred", 1)
    #feat["pv_pred_lag1"] = lag("pv_pred", 1)
    feat["l_pred_lag1"] = lag("l_pred", 1)
    feat["last_p_lag1"] = lag("last_p", 1)

    feat["res_load_pred_lag1"] = (
        feat["l_pred_lag1"] - feat["w_pred_lag1"]
    )
    feat["res_load_ramp_1"] = feat["res_load_pred"] - feat["res_load_pred_lag1"]
    feat["last_move_1"] = feat["last_p"] - feat["last_p_lag1"]

    # Historical id_std summaries
    id_std_hist_24 = hist_matrix("id_std", cfg.lead, 24)

    feat["id_std_hist_last"] = lag("id_std", cfg.lead)
    feat["id_std_hist_lag24"] = lag("id_std", cfg.lead + 24)
    feat["id_std_hist_lag48"] = lag("id_std", cfg.lead + 48)
    feat["id_std_hist_mean_24"] = id_std_hist_24.mean(axis=1)
    feat["id_std_hist_std_24"] = id_std_hist_24.std(axis=1)

    if cfg.add_id_others:
        feat["h2_id_11"] = lag("id_11", 2)
        feat["h2_id_12"] = lag("id_12", 2)
        feat["h2_last_p"] = lag("last_p", 2)
        feat["h2_id_std"] = lag("id_std", 2)

        for colname in ["id_7", "id_8", "id_9", "id_10", "id_11", "id_12"]:
            feat[f"h3_{colname}"] = lag(colname, 3)

        feat["h3_last_p"] = lag("last_p", 3)
        feat["h3_id_std"] = lag("id_std", 3)

    return feat


def build_feature_sets(feature_df: pd.DataFrame):
    all_cols = list(feature_df.columns)
    no_weekday = [c for c in all_cols if c != "weekday"]

    base_v1 = [
        "sin_hod", "cos_hod", "sin_doy", "cos_doy",
        "da_p",
        "res_load_pred", "res_load_real",
        "id3_p_lag4",
        "last_p",
        "id_std",
    ]

    compact_v1 = base_v1 + [
        "spread_last_da",
        "spread_last_id3_lag4",
        "neg_last",
        "id_std_hist_last",
        "id_std_hist_mean_24",
        "id_std_hist_std_24",
    ]

    compact_v2 = compact_v1 + [
        "w_error_lag4",
        #"pv_error_lag4",
        "l_error_lag4",
        "abs_w_error_lag4",
        #"abs_pv_error_lag4",
        "abs_l_error_lag4",
        "res_load_real_lag4",
        "res_load_error_lag4",
        "res_load_pred_lag1",
        "res_load_ramp_1",
        "last_p_lag1",
        "last_move_1",
    ]

    compact_v3 = compact_v2 + [
        "da_p_lag1",
        "w_pred_lag1",
        #"pv_pred_lag1",
        "l_pred_lag1",
        "id_std_hist_lag24",
        "id_std_hist_lag48",
    ]

    compact_v4 = compact_v3.copy()

    extra = [
        c for c in [
            "h2_id_11", "h2_id_12", "h2_last_p", "h2_id_std",
            "h3_id_9", "h3_id_10", "h3_id_11", "h3_id_12",
            "h3_last_p", "h3_id_std",
        ]
        if c in feature_df.columns
    ]
    compact_v4 += extra

    return {
        "compact_v1": compact_v1,
        "compact_v2": compact_v2,
        "compact_v3": compact_v3,
        "compact_v4": compact_v4,
        "all_clean_no_weekday": no_weekday,
    }


# ============================================================
# Target transform helpers
# ============================================================

def build_anchor(feature_df: pd.DataFrame, target_mode: str) -> Optional[np.ndarray]:
    if target_mode == "raw":
        return None
    if target_mode == "resid_last":
        return feature_df["last_p"].values.astype(np.float32)
    if target_mode == "resid_da":
        return feature_df["da_p"].values.astype(np.float32)
    if target_mode == "resid_id3":
        return feature_df["id3_p_lag4"].values.astype(np.float32)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def transform_target(output_norm: np.ndarray, anchor: Optional[np.ndarray], target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return output_norm.astype(np.float32)
    return (output_norm - anchor[:, None]).astype(np.float32)


def invert_target_transform(preds_norm_target_space: np.ndarray, anchor: Optional[np.ndarray], target_mode: str) -> np.ndarray:
    """
    preds_norm_target_space has shape (n_obs, n_targets, n_samples).
    anchor has shape (n_obs,).
    """
    if target_mode == "raw":
        return preds_norm_target_space
    return preds_norm_target_space + anchor[:, None, None]


def make_engression_matrices(feature_df: pd.DataFrame, feature_cols: List[str], output_norm: np.ndarray, target_mode: str):
    weekday_one_hot = np.eye(7)[feature_df["weekday"].values.astype(int) - 1]
    x_main = feature_df[feature_cols].values.astype(np.float32)
    x = np.concatenate([x_main, weekday_one_hot.astype(np.float32)], axis=1)

    anchor = build_anchor(feature_df, target_mode)
    y = transform_target(output_norm, anchor, target_mode)

    return x.astype(np.float32), y.astype(np.float32), anchor


def make_full_engression_matrices(
    input_ts: np.ndarray,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    output_norm: np.ndarray,
    target_mode: str,
):
    n = output_norm.shape[0]
    weekday_one_hot = np.eye(7)[feature_df["weekday"].values.astype(int) - 1]
    x_ts_flat = input_ts.transpose(0, 2, 1).reshape(n, -1).astype(np.float32)
    x_main = feature_df[feature_cols].values.astype(np.float32)
    x = np.concatenate([x_ts_flat, x_main, weekday_one_hot.astype(np.float32)], axis=1)

    anchor = build_anchor(feature_df, target_mode)
    y = transform_target(output_norm, anchor, target_mode)

    return x.astype(np.float32), y.astype(np.float32), anchor


# ============================================================
# Model fitting and sampling
# ============================================================

def fit_single_engression(x_train: np.ndarray, y_train: np.ndarray, cfg: ExperimentConfig):
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    return engression(
        x=x_train_t,
        y=y_train_t,
        classification=False,
        num_layer=cfg.num_layers,
        hidden_dim=cfg.hidden_dim,
        noise_dim=cfg.noise_dim,
        lr=cfg.lr,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        device=DEVICE,
        standardize=cfg.standardize_inside_engression,
        hetero_noise= cfg.hetero_noise,
        scale_hidden_dim=cfg.noise_dim,
        verbose=True,
    )


@torch.no_grad()
def sample_ensemble(fits, x_test: np.ndarray, sample_size_each: int):
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    sample_list = []

    for fit in fits:
        s = fit.sample(x_test_t, sample_size=sample_size_each, expand_dim=True)
        sample_list.append(s.detach().cpu().numpy())

    return np.concatenate(sample_list, axis=2)


# ============================================================
# Saving helpers
# ============================================================

def build_run_name(cfg: ExperimentConfig) -> str:
    target_label = cfg.target_name.replace("_p", "")
    return (
        f"{cfg.benchmark_mode}"
        f"__Y{target_label}"
        f"__{cfg.feature_set}"
        f"__{cfg.target_mode}"
        f"__L{cfg.num_layers}"
        f"_H{cfg.hidden_dim}"
        f"_N{cfg.noise_dim}"
        f"_LR{cfg.lr}"
        f"_BS{cfg.batch_size}"
        f"_E{cfg.num_epochs}"
        f"_ENS{cfg.n_ensemble}"
        f"_HETERO_{cfg.hetero_noise}"
        f"_HS_{cfg.scale_hidden_dim}"
    )


def save_config_and_stats(run_dir: Path, cfg: ExperimentConfig, feature_cols: List[str], norm_stats: dict):
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(run_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(run_dir / "target_columns.json", "w", encoding="utf-8") as f:
        json.dump(TARGET_COLS, f, indent=2)

    with open(run_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Running Engression IDX3 experiment")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Mode: {CFG.benchmark_mode}")
    print(f"Feature set: {CFG.feature_set}")
    print(f"Target columns: {TARGET_COLS}")
    print(f"Target mode: {CFG.target_mode}")

    pred_combine_df = build_base_tables(CFG)
    raw_feature_df = build_raw_feature_dataframe(pred_combine_df)
    pred_combine_norm_df, data_id_mu, data_id_sigma, norm_stats = normalize_dataframe_clean(
        raw_feature_df, CFG
    )

    input_ts, input_std, input_all, input_weekday, output_norm = build_cgm_like_arrays(
        pred_combine_norm_df, CFG
    )

    feature_df = build_feature_dataframe(
        pred_combine_norm_df,
        input_all,
        input_std,
        input_weekday,
        output_norm,
        CFG,
    )

    feature_sets = build_feature_sets(feature_df)
    if CFG.feature_set not in feature_sets:
        raise ValueError(f"Unknown feature_set: {CFG.feature_set}. Available: {list(feature_sets.keys())}")

    feature_cols = feature_sets[CFG.feature_set]

    if CFG.benchmark_mode == "compact":
        x, y, anchor = make_engression_matrices(feature_df, feature_cols, output_norm, CFG.target_mode)
    elif CFG.benchmark_mode == "full":
        x, y, anchor = make_full_engression_matrices(input_ts, feature_df, feature_cols, output_norm, CFG.target_mode)
    else:
        raise ValueError("benchmark_mode must be 'compact' or 'full'")

    print("Final X shape:", x.shape)
    print("Final Y shape:", y.shape)
    print(f"Using {len(feature_cols)} clean feature columns + weekday one-hot")

    idx_train, idx_val, idx_test = make_splits(len(x), CFG)

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]
    anchor_test = None if anchor is None else anchor[idx_test]

    print("Train:", x_train.shape, y_train.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    fits = []
    for k in range(CFG.n_ensemble):
        print(f"\n========== Engression run {k + 1}/{CFG.n_ensemble} ==========")
        torch.manual_seed(CFG.seed_base + k)
        np.random.seed(CFG.seed_base + k)
        fits.append(fit_single_engression(x_train, y_train, CFG))

    sample_size_each = max(1, CFG.n_samples_test // CFG.n_ensemble)
    preds_norm_target_space = sample_ensemble(fits, x_test, sample_size_each)
    preds_norm = invert_target_transform(preds_norm_target_space, anchor_test, CFG.target_mode)

    preds = data_id_sigma * preds_norm + data_id_mu

    y_test_raw_norm = invert_target_transform(y_test[:, :, None], anchor_test, CFG.target_mode)[:, :, 0]
    y_test_denorm = data_id_sigma * y_test_raw_norm + data_id_mu

    run_name = build_run_name(CFG)
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    np.save(run_dir / "pred.npy", preds)
    np.save(run_dir / "y_test.npy", y_test_denorm)

    if CFG.save_normalized_outputs:
        np.save(run_dir / "pred_norm.npy", preds_norm)
        np.save(run_dir / "y_test_norm.npy", y_test_raw_norm)

    if anchor_test is not None:
        np.save(run_dir / "anchor_test.npy", anchor_test)

    save_config_and_stats(run_dir, CFG, feature_cols, norm_stats)

    manifest = {
        "run_name": run_name,
        "target_columns": TARGET_COLS,
        "pred_path": str((run_dir / "pred.npy").resolve()),
        "y_test_path": str((run_dir / "y_test.npy").resolve()),
        "n_test_obs": int(len(idx_test)),
        "n_targets": int(preds.shape[1]),
        "n_samples_test": int(preds.shape[2]),
        "device": DEVICE,
    }

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("Saved outputs to:")
    print(run_dir.resolve())
    print("Prediction shape:", preds.shape, "= (n_test, 3 targets, n_samples)")
    print("Y test shape:", y_test_denorm.shape, "= (n_test, 3 targets)")
    print("=" * 80)


if __name__ == "__main__":
    main()
