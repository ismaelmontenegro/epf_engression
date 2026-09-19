
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from engression_module import engression


@dataclass
class ExperimentConfig:
    data_dir: str = r"C:/Users/Ismas/PycharmProjects/epf_engression/epf_engression"
    out_subdir: str = "engression_outputs_experiments"
    start_date: str = "2017-06-14"

    n_days: int = 837
    n_hours: int = 24
    test_days: int = 200
    lead: int = 4
    start_index: int = 168
    add_id_others: bool = True

    benchmark_mode: str = "compact"   # "compact" or "full"
    feature_set: str = "compact_v1"   # "compact_v1", "compact_v2", ...
    target_mode: str = "resid_last"          # "raw", "resid_last", "resid_da", "resid_id3"

    hetero_noise: bool = True
    scale_hidden_dim: int = 256


    num_layers: int = 2
    hidden_dim: int = 128
    noise_dim: int = 32
    lr: float = 1e-4
    num_epochs: int = 2000
    batch_size: int = 1024
    n_samples_test: int = 1000
    n_ensemble: int = 10
    standardize_inside_engression: bool = False
    save_normalized_outputs: bool = True
    seed_base: int = 1000


CFG = ExperimentConfig()
DATA_DIR = Path(CFG.data_dir)

OUT_DIR = DATA_DIR / CFG.out_subdir
OUT_DIR.mkdir(exist_ok=True, parents=True)

START_DATE = datetime.strptime(CFG.start_date, "%Y-%m-%d")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_hourly_id_file(folder: Path, prefix: str, n_keep: int, n_days: int, n_hours: int):
    arr = np.zeros((n_days, n_hours, 2 + n_keep))
    for n_hour in range(n_hours):
        fname = folder / f"{prefix}_{n_hour:02d}"
        with open(fname) as f:
            lines = f.readlines()
        date_hour = [str(line.split(",")[0].strip()) for line in lines[1:]]
        data = np.array([[float(e) for e in line.strip().split(",")[1:]] for line in lines[1:]])
        kept = data[:, :n_keep]
        date_str = [datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y%m%d")
                    for date_string in date_hour]
        date_diff = [(datetime.strptime(date_string, "%Y%m%d") - START_DATE).days
                     for date_string in date_str]
        arr[:, n_hour, 0] = np.array(date_diff) + 1
        arr[:, n_hour, 1] = n_hour
        arr[:, n_hour, 2:] = kept
    return arr


def read_exog_series(filepath: Path):
    with open(filepath) as f:
        lines = f.readlines()
    return np.array([float(line.strip()) for line in lines])


def build_base_tables(cfg: ExperimentConfig) -> pd.DataFrame:
    volumesintra = read_hourly_id_file(
        folder=Path(cfg.data_dir) / "ID_DATA",
        prefix="volumes_hourly",
        n_keep=12,
        n_days=cfg.n_days,
        n_hours=cfg.n_hours,
    )
    volumesintra_flat = volumesintra.reshape(-1, volumesintra.shape[-1])

    pricesintra = read_hourly_id_file(
        folder=Path(cfg.data_dir) / "ID_DATA",
        prefix="prices_hourly",
        n_keep=13,
        n_days=cfg.n_days,
        n_hours=cfg.n_hours,
    )
    pricesintra_flat = pricesintra.reshape(-1, pricesintra.shape[-1])

    subprices = pricesintra_flat[:, 2:14]
    subvolumes = volumesintra_flat[:, 2:]

    id3_volume_sum = np.sum(subvolumes, axis=1)
    id3_pricevolume_sum = np.sum(subprices * subvolumes, axis=1)
    mean_subprice = np.mean(subprices, axis=1)

    zero_volume_index = np.where(id3_volume_sum == 0)[0]
    nonzero_mask = np.ones(subprices.shape[0], dtype=bool)
    nonzero_mask[zero_volume_index] = False

    id3_vwa_price = np.zeros(subprices.shape[0])
    id3_vwa_price[nonzero_mask] = (
        id3_pricevolume_sum[nonzero_mask] / id3_volume_sum[nonzero_mask]
    )
    id3_vwa_price[zero_volume_index] = mean_subprice[zero_volume_index]

    id_pred = np.zeros((cfg.n_days * cfg.n_hours, 16))
    id_pred[:, :2] = pricesintra_flat[:, :2]
    id_pred[:, 2] = id3_vwa_price
    id_pred[:, 3:] = pricesintra_flat[:, 2:15]

    id_pred_df = pd.DataFrame(
        id_pred,
        columns=[
            "day", "hour", "id3_p",
            "id_1", "id_2", "id_3", "id_4", "id_5", "id_6",
            "id_7", "id_8", "id_9", "id_10", "id_11", "id_12", "last_p"
        ],
    )

    exog_pred = np.zeros((cfg.n_days * cfg.n_hours, 7))
    with open(Path(cfg.data_dir) / "EXOG_DATA" / "Day_Ahead_Epex.csv") as f:
        lines = f.readlines()

    date_str = [str(line.split(";")[0].strip()) for line in lines]
    hour = [int(line.split(";")[1].strip()) for line in lines]
    da_price = [float(line.split(";")[2].strip()) for line in lines]
    date_diff = [(datetime.strptime(date_string, "%Y%m%d") - START_DATE).days
                 for date_string in date_str]

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
    exog_pred[:, 5] = load_da
    exog_pred[:, 6] = load_real

    exog_pred_df = pd.DataFrame(
        exog_pred,
        columns=["day", "hour", "da_p", "w_pred", "w_real", "l_pred", "l_real"],
    )

    return pd.merge(exog_pred_df, id_pred_df, how="outer", on=["day", "hour"])


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

    exog_cols = ["da_p", "w_pred", "w_real", "l_pred", "l_real"]
    id_cols = ["id3_p", "id_1", "id_2", "id_3", "id_4", "id_5", "id_6",
               "id_7", "id_8", "id_9", "id_10", "id_11", "id_12", "last_p"]

    norm_stats = {"exog": {}, "id_joint": {}}

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
    norm_stats["id_joint"] = {"mu": float(data_id_mu), "sigma": float(data_id_sigma)}

    df["id_std"] = np.std(df[["id_3", "id_4", "id_5", "id_6", "id_7", "id_8",
                              "id_9", "id_10", "id_11", "id_12", "last_p"]].values, axis=1)

    df = df.drop(columns=["day", "date", "hour", "weekofyear", "dayofyear"])
    return df, data_id_mu, data_id_sigma, norm_stats


def build_cgm_like_arrays(pred_combine_norm_df: pd.DataFrame, cfg: ExperimentConfig):
    data_len = pred_combine_norm_df.shape[0]
    index_all = np.arange(data_len)

    input_ts = np.zeros((data_len, 20, 165))
    for i in range(165):
        input_ts[:, :, i] = pred_combine_norm_df.iloc[(index_all - cfg.lead - i), 5:].values

    input_all = np.zeros((data_len, 52 if cfg.add_id_others else 40))
    input_all[:, :4] = pred_combine_norm_df.iloc[:, 1:5].values

    for i in range(cfg.lead):
        input_all[:, 4 + i] = pred_combine_norm_df.iloc[(index_all - i), 5].values
        input_all[:, 8 + i] = pred_combine_norm_df.iloc[(index_all - i), 6].values
        input_all[:, 12 + i] = pred_combine_norm_df.iloc[(index_all - i), 8].values
        input_all[:, 16 + i] = pred_combine_norm_df.iloc[(index_all - i), -2].values

    input_all[:, 20:40] = input_ts[:, :, 0]

    if cfg.add_id_others:
        input_all[:, 40:44] = pred_combine_norm_df.iloc[(index_all - 2), 19:23].values
        input_all[:, 44:52] = pred_combine_norm_df.iloc[(index_all - 3), 15:23].values

    input_weekday = pred_combine_norm_df.iloc[:, 0].values.astype(int)
    input_std = input_ts[:, -1, :]
    output_norm = pred_combine_norm_df.iloc[:, 13:23].values

    input_ts = input_ts[cfg.start_index:, :, :]
    input_all = input_all[cfg.start_index:, :]
    input_std = input_std[cfg.start_index:, :]
    input_weekday = input_weekday[cfg.start_index:]
    output_norm = output_norm[cfg.start_index:, :]
    return input_ts, input_std, input_all, input_weekday, output_norm


def build_feature_dataframe(
    pred_combine_norm_df, input_all, input_std, input_weekday, output_norm, cfg
):
    n = output_norm.shape[0]
    base_trim = pred_combine_norm_df.iloc[cfg.start_index:].reset_index(drop=True)

    feat = pd.DataFrame(index=np.arange(n))

    # time
    feat["sin_hod"] = base_trim["sin_hod"].values
    feat["cos_hod"] = base_trim["cos_hod"].values
    feat["sin_doy"] = base_trim["sin_doy"].values
    feat["cos_doy"] = base_trim["cos_doy"].values
    feat["weekday"] = input_weekday

    # forecast-origin variables — lag 0, all clean per original CGM
    feat["da_p"]   = base_trim["da_p"].values
    feat["w_pred"] = base_trim["w_pred"].values
    #feat["pv_pred"] = base_trim["pv_pred"].values
    feat["l_pred"] = base_trim["l_pred"].values
    feat["l_real"] = input_all[:, 20 + 4]          # PATCHED: lag 4 via input_ts[:,4,0]
    feat["w_real"] = input_all[:, 20 + 2]          # PATCHED: lag 4 via input_ts[:,2,0]
    feat["id3_p_lag4"] = input_all[:, 20 + 5]
    #feat["id3_p"] = base_trim["id3_p"].values
    feat["last_p"] = base_trim["last_p"].values
    feat["id_std"] = input_std[:, 0]               # PATCHED: lag 4, matches original input_std

    # derived features
    feat["spread_last_da"]  = feat["last_p"] - feat["da_p"]
    feat["spread_last_id3_lag4"] = feat["last_p"] - feat["id3_p_lag4"]
    feat["neg_last"]        = (feat["last_p"] < 0).astype(float)

    feat["w_error"]     = feat["w_real"] - feat["w_pred"]   # real lag4 minus pred lag0
    feat["l_error"]     = feat["l_real"] - feat["l_pred"]
    feat["abs_w_error"] = np.abs(feat["w_error"])
    feat["abs_l_error"] = np.abs(feat["l_error"])

    feat["res_load_pred"]       = feat["l_pred"] - feat["w_pred"]
    feat["res_load_real_lag4"]  = feat["l_real"] - feat["w_real"]          # PATCHED name
    feat["res_load_error_lag4"] = feat["res_load_real_lag4"] - feat["res_load_pred"]  # PATCHED name

    # lagged summaries from input_all (these were already clean)
    feat["da_p_lag1"]      = input_all[:, 5]
    feat["w_pred_lag1"]    = input_all[:, 9]
    feat["l_pred_lag1"]    = input_all[:, 13]
    feat["last_p_lag1"]    = input_all[:, 17]

    feat["res_load_pred_lag1"] = feat["l_pred_lag1"] - feat["w_pred_lag1"]
    feat["res_load_ramp_1"]    = feat["res_load_pred"] - feat["res_load_pred_lag1"]
    feat["last_move_1"]        = feat["last_p"] - feat["last_p_lag1"]

    # history summaries (already clean)
    feat["id_std_hist_last"]   = input_std[:, 0]
    feat["id_std_hist_lag24"]  = input_std[:, 24]
    feat["id_std_hist_lag48"]  = input_std[:, 48]
    feat["id_std_hist_mean_24"] = input_std[:, :24].mean(axis=1)
    feat["id_std_hist_std_24"]  = input_std[:, :24].std(axis=1)

    if cfg.add_id_others:
        for j, colname in enumerate(["h2_id_11", "h2_id_12", "h2_last_p", "h2_id_std"]):
            feat[colname] = input_all[:, 40 + j]
        for j, colname in enumerate([
            "h3_id_7", "h3_id_8", "h3_id_9", "h3_id_10",
            "h3_id_11", "h3_id_12", "h3_last_p", "h3_id_std"
        ]):
            feat[colname] = input_all[:, 44 + j]

    return feat


def build_feature_sets(feature_df):
    all_cols  = list(feature_df.columns)
    no_weekday = [c for c in all_cols if c != "weekday"]

    base_v1 = [
        "sin_hod", "cos_hod", "sin_doy", "cos_doy",
        "da_p", "res_load_pred", "res_load_real_lag4",
        "id3_p_lag4", "last_p", "id_std",
        "res_load_pred_lag1", "res_load_ramp_1",
        "last_p_lag1", "last_move_1",
    ]

    compact_v1 = base_v1 + [
        "spread_last_da", "spread_last_id3_lag4", "neg_last",
        "id_std_hist_last", "id_std_hist_mean_24", "id_std_hist_std_24",
    ]

    compact_v2 = compact_v1 + [
        "res_load_pred", "res_load_real_lag4", "res_load_error_lag4",   # PATCHED names
        "res_load_pred_lag1", "res_load_ramp_1",
        "last_p_lag1", "last_move_1",
    ]

    compact_v3 = compact_v2 + [
        "da_p_lag1", "w_pred_lag1", "l_pred_lag1",
        "id_std_hist_lag24", "id_std_hist_lag48",
    ]

    compact_v4 = compact_v3.copy()
    extra = [c for c in [
        "h2_id_11", "h2_id_12", "h2_last_p", "h2_id_std",
        "h3_id_9", "h3_id_10", "h3_id_11", "h3_id_12", "h3_last_p", "h3_id_std"
    ] if c in feature_df.columns]
    compact_v4 += extra

    return {
        "compact_v1": compact_v1,
        "compact_v2": compact_v2,
        "compact_v3": compact_v3,
        "compact_v4": compact_v4,
        "all_clean_no_weekday": no_weekday,
    }


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


def invert_target_transform(preds_norm: np.ndarray, anchor: Optional[np.ndarray], target_mode: str) -> np.ndarray:
    if target_mode == "raw":
        return preds_norm
    return preds_norm + anchor[:, None, None]


def make_engression_matrices(feature_df, feature_cols, output_norm, target_mode):
    weekday_one_hot = np.eye(7)[feature_df["weekday"].values.astype(int) - 1]
    x_main = feature_df[feature_cols].values.astype(np.float32)
    x = np.concatenate([x_main, weekday_one_hot.astype(np.float32)], axis=1)
    anchor = build_anchor(feature_df, target_mode)
    y = transform_target(output_norm, anchor, target_mode)
    return x.astype(np.float32), y.astype(np.float32), anchor


def make_full_engression_matrices(input_ts, feature_df, feature_cols, output_norm, target_mode):
    n = output_norm.shape[0]
    weekday_one_hot = np.eye(7)[feature_df["weekday"].values.astype(int) - 1]
    x_ts_flat = input_ts.transpose(0, 2, 1).reshape(n, -1).astype(np.float32)
    x_main = feature_df[feature_cols].values.astype(np.float32)
    x = np.concatenate([x_ts_flat, x_main, weekday_one_hot.astype(np.float32)], axis=1)
    anchor = build_anchor(feature_df, target_mode)
    y = transform_target(output_norm, anchor, target_mode)
    return x.astype(np.float32), y.astype(np.float32), anchor


def fit_single_engression(x_train, y_train, cfg: ExperimentConfig):
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
        verbose=True,
        hetero_noise=cfg.hetero_noise,
        scale_cond_idx=None,
        scale_hidden_dim=cfg.scale_hidden_dim,
        upper_twes_lambda= 0.0,
        upper_twes_quantile= 0.9,
        hetero_output=False,
        output_scale_hidden_dim=256
    )


@torch.no_grad()
def sample_ensemble(fits, x_test, sample_size_each: int):
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    sample_list = []
    for fit in fits:
        s = fit.sample(x_test_t, sample_size=sample_size_each, expand_dim=True)
        sample_list.append(s.detach().cpu().numpy())
    return np.concatenate(sample_list, axis=2)


def build_run_name(cfg: ExperimentConfig) -> str:
    return (
        f"{cfg.benchmark_mode}"
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
    with open(run_dir / "normalization_stats.json", "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)


def main():
    print("=" * 80)
    print("Running clean Engression experiment")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Mode: {CFG.benchmark_mode}")
    print(f"Feature set: {CFG.feature_set}")
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
        pred_combine_norm_df, input_all, input_std, input_weekday, output_norm, CFG
    )
    feature_sets = build_feature_sets(feature_df)
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
    anchor_train = None if anchor is None else anchor[idx_train]

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

    y_train_raw_denorm = invert_target_transform(y_train[:,:,None], anchor_train, CFG.target_mode)[:, :, 0]
    y_train_denorm = data_id_sigma * y_train_raw_denorm + data_id_mu

    run_name = build_run_name(CFG)
    run_dir = OUT_DIR / run_name
    run_dir.mkdir(exist_ok=True, parents=True)

    np.save(run_dir / "pred.npy", preds)
    np.save(run_dir / "y_test.npy", y_test_denorm)
    np.save(run_dir / "y_train.npy", y_train_denorm)
    if CFG.save_normalized_outputs:
        np.save(run_dir / "pred_norm.npy", preds_norm)
        np.save(run_dir / "y_test_norm.npy", y_test_raw_norm)
    if anchor_test is not None:
        np.save(run_dir / "anchor_test.npy", anchor_test)

    save_config_and_stats(run_dir, CFG, feature_cols, norm_stats)

    manifest = {
        "run_name": run_name,
        "pred_path": str((run_dir / "pred.npy").resolve()),
        "y_test_path": str((run_dir / "y_test.npy").resolve()),
        "n_test_obs": int(len(idx_test)),
        "n_samples_test": int(preds.shape[2]),
        "device": DEVICE,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 80)
    print("Saved outputs to:")
    print(run_dir.resolve())
    print("=" * 80)


if __name__ == "__main__":
    main()