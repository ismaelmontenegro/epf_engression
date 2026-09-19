import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# adjust this import to your local engression package structure
# e.g. from engression import engression
from engression_module import engression

# =========================
# Config
# =========================
DATA_DIR = Path("C:/Users/Ismas/PycharmProjects/epf_engression/epf_engression")
START_DATE = datetime(2017, 6, 14)

N_DAYS = 837
N_HOURS = 24
TEST_DAYS = 200
LEAD = 4
START_INDEX = 168  # 24 * 7

ADD_ID_OTHERS = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# choose from: "compact", "full"
BENCHMARK_MODE = "full"

# engression hyperparams
NUM_LAYERS = 2
HIDDEN_DIM = 128
NOISE_DIM = 32
LR = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 1024
N_SAMPLES_TEST = 1000
N_ENSEMBLE = 10


# =========================
# File loading helpers
# =========================
def read_hourly_id_file(folder: Path, prefix: str, n_keep: int):

    arr = np.zeros((N_DAYS, N_HOURS, 2 + n_keep))

    for n_hour in range(N_HOURS):
        fname = folder / f"{prefix}_{n_hour:02d}"
        with open(fname) as f:
            lines = f.readlines()

        date_hour = [str(line.split(",")[0].strip()) for line in lines[1:]]
        data = np.array([[float(e) for e in line.strip().split(",")[1:]] for line in lines[1:]])

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
    return np.array([float(line.strip()) for line in lines])


# =========================
# Step 1: reproduce CGM raw tables
# =========================
def build_base_tables(data_dir: Path):
    # ---------- intraday volumes ----------
    volumesintra = read_hourly_id_file(
        folder=data_dir / "ID_DATA",
        prefix="volumes_hourly",
        n_keep=12,
    )
    volumesintra_flat = volumesintra.reshape(-1, volumesintra.shape[-1])

    # ---------- intraday prices ----------
    pricesintra = read_hourly_id_file(
        folder=data_dir / "ID_DATA",
        prefix="prices_hourly",
        n_keep=13,
    )
    pricesintra_flat = pricesintra.reshape(-1, pricesintra.shape[-1])

    # ---------- reconstruct ID3 ----------
    subprices = pricesintra_flat[:, 2:14]   # 12 subprices
    subvolumes = volumesintra_flat[:, 2:]   # 12 subvolumes

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

    # target-relevant raw intraday prices
    id_pred = np.zeros((N_DAYS * N_HOURS, 16))
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

    # ---------- exogenous variables ----------
    exog_pred = np.zeros((N_DAYS * N_HOURS, 7))

    with open(data_dir / "EXOG_DATA" / "Day_Ahead_Epex.csv") as f:
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

    woff = read_exog_series(data_dir / "EXOG_DATA" / "final_wind_offshore.csv")
    won = read_exog_series(data_dir / "EXOG_DATA" / "final_wind_onshore.csv")
    woffreal = read_exog_series(data_dir / "EXOG_DATA" / "final_wind_offshore_real.csv")
    wonreal = read_exog_series(data_dir / "EXOG_DATA" / "final_wind_onshore_real.csv")
    load_da = read_exog_series(data_dir / "EXOG_DATA" / "final_load_da.csv")
    load_real = read_exog_series(data_dir / "EXOG_DATA" / "final_load_real.csv")

    exog_pred[:, 3] = won + woff
    exog_pred[:, 4] = wonreal + woffreal
    exog_pred[:, 5] = load_da
    exog_pred[:, 6] = load_real

    exog_pred_df = pd.DataFrame(
        exog_pred,
        columns=["day", "hour", "da_p", "w_pred", "w_real", "l_pred", "l_real"],
    )

    pred_combine_df = pd.merge(exog_pred_df, id_pred_df, how="outer", on=["day", "hour"])
    return pred_combine_df


# =========================
# Step 2: reproduce CGM normalization
# =========================
def build_normalized_dataframe(pred_combine_df: pd.DataFrame):
    pred_combine_arr = pred_combine_df.values.copy()

    # --- train-only normalization window (excludes val AND test) ---
    # mirror the chronological split used later in make_splits()
    trainval_len = (N_DAYS - TEST_DAYS - 7) * N_HOURS  # 15120
    val_size = int(0.2 * trainval_len)  # 3024
    train_size = trainval_len - val_size  # 12096

    # first START_INDEX rows get trimmed later, so the train block in this
    # (untrimmed) frame is [START_INDEX, START_INDEX + train_size) = [168, 12264)
    norm_start = START_INDEX
    norm_stop = START_INDEX + train_size

    pred_combine_norm = pred_combine_arr.copy()

    # exogenous predictors, columns 2..6, on train only
    for i in range(2, 7):
        data_i = pred_combine_arr[:, i]
        data_cal = pred_combine_arr[norm_start:norm_stop, i]
        mu = data_cal.mean()
        sigma = data_cal.std()
        pred_combine_norm[:, i] = (data_i - mu) / sigma

    # ID block jointly, on train only
    data_id_cal = pred_combine_arr[norm_start:norm_stop, 7:]  # was [:15288, 8:]
    data_id_mu = data_id_cal.mean()
    data_id_sigma = data_id_cal.std()
    pred_combine_norm[:, 7:] = (pred_combine_arr[:, 7:] - data_id_mu) / data_id_sigma

    pred_combine_norm_df = pd.DataFrame(pred_combine_norm, columns=pred_combine_df.columns)

    result_date = [
        START_DATE + timedelta(days=int(diff))
        for diff in pred_combine_norm_df["day"].values
    ]

    pred_combine_norm_df.insert(0, "date", result_date)
    pred_combine_norm_df.insert(1, "weekday", np.array([d.weekday() + 1 for d in result_date]).astype(int))
    pred_combine_norm_df.insert(2, "weekofyear", np.array([d.strftime("%V") for d in result_date]).astype(int))
    pred_combine_norm_df.insert(3, "dayofyear", np.array([d.strftime("%j") for d in result_date]).astype(int))

    pred_combine_norm_df = pred_combine_norm_df.drop(columns=["day", "date"])

    sin_hod = np.sin((pred_combine_norm_df["hour"].values / 24) * 2 * np.pi)
    cos_hod = np.cos((pred_combine_norm_df["hour"].values / 24) * 2 * np.pi)
    sin_doy = np.sin((pred_combine_norm_df["dayofyear"].values / 365) * 2 * np.pi)
    cos_doy = np.cos((pred_combine_norm_df["dayofyear"].values / 365) * 2 * np.pi)

    pred_combine_norm_df.insert(1, "cos_doy", cos_doy)
    pred_combine_norm_df.insert(1, "sin_doy", sin_doy)
    pred_combine_norm_df.insert(1, "cos_hod", cos_hod)
    pred_combine_norm_df.insert(1, "sin_hod", sin_hod)

    pred_combine_norm_df = pred_combine_norm_df.drop(columns=["hour", "weekofyear", "dayofyear"])

    # std over id_3..last_p block exactly as in CGM script
    pred_combine_norm_df["id_std"] = np.std(pred_combine_norm_df.iloc[:, 13:].values, axis=1)

    return pred_combine_norm_df, data_id_mu, data_id_sigma


# =========================
# Step 3: reproduce CGM tensors
# =========================
def build_cgm_like_arrays(pred_combine_norm_df: pd.DataFrame, add_id_others: bool = True):

    data_len = pred_combine_norm_df.shape[0]
    index_all = np.arange(data_len)

    # -------- Input 1: long lag block --------
    input_ts = np.zeros((data_len, 20, 165))
    for i in range(165):
        input_ts[:, :, i] = pred_combine_norm_df.iloc[(index_all - LEAD - i), 5:].values

    # -------- Input 3: recent summary block --------
    input_all = np.zeros((data_len, 52 if add_id_others else 40))
    input_all[:, :4] = pred_combine_norm_df.iloc[:, 1:5].values  # sin/cos terms

    for i in range(LEAD):
        input_all[:, 4 + i] = pred_combine_norm_df.iloc[(index_all - i), 5].values    # da_p
        input_all[:, 8 + i] = pred_combine_norm_df.iloc[(index_all - i), 6].values    # w_pred
        input_all[:, 12 + i] = pred_combine_norm_df.iloc[(index_all - i), 8].values   # l_pred
        input_all[:, 16 + i] = pred_combine_norm_df.iloc[(index_all - i), -2].values  # last_p

    input_all[:, 20:40] = input_ts[:, :, 0]

    if add_id_others:
        input_all[:, 40:44] = pred_combine_norm_df.iloc[(index_all - 2), 19:23].values
        input_all[:, 44:52] = pred_combine_norm_df.iloc[(index_all - 3), 15:23].values

    # -------- weekday --------
    input_weekday = pred_combine_norm_df.iloc[:, 0].values.astype(int)

    # -------- Input 2: id_std history --------
    input_std = input_ts[:, -1, :]

    # -------- target --------
    output_norm = pred_combine_norm_df.iloc[:, 13:23].values  # id_3 .. id_12

    # trim first 168 rows
    input_ts = input_ts[START_INDEX:, :, :]
    input_all = input_all[START_INDEX:, :]
    input_std = input_std[START_INDEX:, :]
    input_weekday = input_weekday[START_INDEX:]
    output_norm = output_norm[START_INDEX:, :]

    return input_ts, input_std, input_all, input_weekday, output_norm


# =========================
# Step 4: build engression matrices
# =========================
def make_engression_matrices(
    input_ts: np.ndarray,
    input_std: np.ndarray,
    input_all: np.ndarray,
    input_weekday: np.ndarray,
    output_norm: np.ndarray,
    mode: str = "compact",
):
    """
    mode='compact':
        x = [input_all, input_std, weekday_one_hot]
    mode='full':
        x = [flatten(input_ts), input_all, weekday_one_hot]
    """
    n = output_norm.shape[0]
    weekday_one_hot = np.eye(7)[input_weekday - 1]

    if mode == "compact":
        x = np.concatenate([input_all, input_std, weekday_one_hot], axis=1)
    elif mode == "full":
        # lag-major flattening: (n, 20, 165) -> (n, 165, 20) -> (n, 3300)
        x_ts_flat = input_ts.transpose(0, 2, 1).reshape(n, -1)
        x = np.concatenate([x_ts_flat, input_all, weekday_one_hot], axis=1)
    else:
        raise ValueError("mode must be 'compact' or 'full'")

    y = output_norm.copy()  # shape (n, 10)
    return x.astype(np.float32), y.astype(np.float32)


# =========================
# Step 5: chronological split
# =========================
def make_splits(n_obs: int):
    """
    Same test logic as the CGM script:
      train_end = (837 - 200 - 7) * 24
    We also create an explicit chronological validation split from the train block.
    """
    train_end = (N_DAYS - TEST_DAYS - 7) * N_HOURS  # same as original script
    assert train_end < n_obs

    val_size = int(0.2 * train_end)
    tr_end = train_end - val_size

    idx_train = np.arange(0, tr_end)
    idx_val = np.arange(tr_end, train_end)
    idx_test = np.arange(train_end, n_obs)

    return idx_train, idx_val, idx_test


# =========================
# Step 6: fitting helper
# =========================
def fit_single_engression(x_train, y_train):
    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)

    fit = engression(
        x=x_train_t,
        y=y_train_t,
        classification=False,
        num_layer=NUM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        noise_dim=NOISE_DIM,
        lr=LR,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        standardize=False,   #data already normalized like CGM
        verbose=True,
    )
    return fit


@torch.no_grad()
def sample_ensemble(fits, x_test, sample_size_each: int):
    x_test_t = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    sample_list = []

    for fit in fits:
        s = fit.sample(x_test_t, sample_size=sample_size_each, expand_dim=True)
        s = s.detach().cpu().numpy()  # (n_test, 10, sample_size_each)
        sample_list.append(s)

    return np.concatenate(sample_list, axis=2)


# =========================
# Main
# =========================
def main():
    print(f"Running engression benchmark in '{BENCHMARK_MODE}' mode on {DEVICE}")

    # 1) rebuild same raw merged table as CGM
    pred_combine_df = build_base_tables(DATA_DIR)

    # 2) apply same normalization logic
    pred_combine_norm_df, data_id_mu, data_id_sigma = build_normalized_dataframe(pred_combine_df)

    # 3) rebuild same intermediate CGM arrays
    input_ts, input_std, input_all, input_weekday, output_norm = build_cgm_like_arrays(
        pred_combine_norm_df,
        add_id_others=ADD_ID_OTHERS,
    )

    # 4) map to engression-style X, Y
    x, y = make_engression_matrices(
        input_ts=input_ts,
        input_std=input_std,
        input_all=input_all,
        input_weekday=input_weekday,
        output_norm=output_norm,
        mode=BENCHMARK_MODE,
    )

    print("X shape:", x.shape)
    print("Y shape:", y.shape)

    # 5) chronological split
    idx_train, idx_val, idx_test = make_splits(len(x))

    x_train = x[idx_train]
    y_train = y[idx_train]

    x_val = x[idx_val]
    y_val = y[idx_val]

    x_test = x[idx_test]
    y_test = y[idx_test]

    print("Train:", x_train.shape, y_train.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", x_test.shape, y_test.shape)

    # 6) fit multiple engression runs for stability
    fits = []
    for k in range(N_ENSEMBLE):
        print(f"\n========== Engression run {k + 1}/{N_ENSEMBLE} ==========")
        torch.manual_seed(1000 + k)
        np.random.seed(1000 + k)
        fit = fit_single_engression(x_train, y_train)
        fits.append(fit)

    # 7) sample path forecasts on test set
    sample_size_each = max(1, N_SAMPLES_TEST // N_ENSEMBLE)
    preds_norm = sample_ensemble(fits, x_test, sample_size_each=sample_size_each)

    # 8) de-normalize exactly like CGM
    preds = data_id_sigma * preds_norm + data_id_mu
    y_test_denorm = data_id_sigma * y_test + data_id_mu

    print("Predictions shape:", preds.shape)  # (n_test, 10, total_samples)

    # 9) save
    out_dir = DATA_DIR / "engression_outputs"
    out_dir.mkdir(exist_ok=True)

    np.save(out_dir / f"pred_engression_{BENCHMARK_MODE}_e200.npy", preds)
    np.save(out_dir / f"y_test_{BENCHMARK_MODE}_e200.npy", y_test_denorm)

    # optional: save normalized too
    np.save(out_dir / f"pred_engression_{BENCHMARK_MODE}_norm_e200.npy", preds_norm)
    np.save(out_dir / f"y_test_{BENCHMARK_MODE}_norm_e200.npy", y_test)

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()