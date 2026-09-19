import time
from multiprocessing.spawn import freeze_support

import numpy as np
import pandas as pd
from multiprocessing import Pool
from statsmodels.robust import mad
from sklearn.linear_model import LassoCV
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only the ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

tr = None
cal_len = 397

LEAD = 4  # predict h with info until h-LEAD

woff = []
won = []
woffreal = []
wonreal = []
load = []
loadreal = []

home_dir = 'C:/Users/Ismas/PycharmProjects/epf_engression/epf_engression/'

pricesda = np.zeros((837, 24))
with open(home_dir + 'EXOG_DATA/' + 'Day_Ahead_Epex.csv') as f:
    data = [float(line.split(';')[2].strip()) for line in f.readlines()]
    for product in range(24):
        pricesda[:, product] = data[product::24]

volumesintra = []
for product in range(24):
    with open(home_dir + 'ID_DATA/' + f'volumes_hourly_{("0" + str(product))[-2:]}') as f:
        data = np.array([[float(e) for e in line.strip().split(',')[1:]] for line in f.readlines()[1:]])
    volumesintra.append(data)

pricesintra = []
for product in range(24):
    with open(home_dir + 'ID_DATA/' + f'prices_hourly_{("0" + str(product))[-2:]}') as f:
        data = np.array([[float(e) for e in line.strip().split(',')[1:]] for line in f.readlines()[1:]])
    pricesintra.append(data)


def ID(day, product, lag):
    subprices = pricesintra[product][day, lag]
    subvolumes = volumesintra[product][day, lag]
    if not np.sum(subvolumes):
        return np.mean(subprices)
    return np.sum(subprices * subvolumes) / np.sum(subvolumes)


def ID3(day, product):
    return ID(day, product, list(range(12)))


naive = np.array([[ID(d, p, [12]) for p in range(24)] for d in range(837)])

forecast_days = 837 - cal_len

with open(home_dir + 'EXOG_DATA/' + 'final_wind_offshore.csv') as f:
    data = f.readlines()
    for line in data:
        woff.append(float(line.strip()))

woff = np.array([woff[h::24] for h in range(24)]).T

with open(home_dir + 'EXOG_DATA/' + 'final_wind_onshore.csv') as f:
    data = f.readlines()
    for line in data:
        won.append(float(line.strip()))

won = np.array([won[h::24] for h in range(24)]).T

with open(home_dir + 'EXOG_DATA/' + 'final_wind_offshore_real.csv') as f:
    data = f.readlines()
    for line in data:
        woffreal.append(float(line.strip()))

woffreal = np.array([woffreal[h::24] for h in range(24)]).T

with open(home_dir + 'EXOG_DATA/' + 'final_wind_onshore_real.csv') as f:
    data = f.readlines()
    for line in data:
        wonreal.append(float(line.strip()))

wonreal = np.array([wonreal[h::24] for h in range(24)]).T

wsum = won + woff
wsumreal = wonreal + woffreal

with open(home_dir + 'EXOG_DATA/' + 'final_load_da.csv') as f:
    data = f.readlines()
    for line in data:
        load.append(float(line.strip()))

load = np.array([load[h::24] for h in range(24)]).T

with open(home_dir + 'EXOG_DATA/' + 'final_load_real.csv') as f:
    data = f.readlines()
    for line in data:
        loadreal.append(float(line.strip()))

loadreal = np.array([loadreal[h::24] for h in range(24)]).T


TRAIN_LO, TRAIN_HI = 7, 511            # days 7..510, matching NN models
N_TRAIN = TRAIN_HI - TRAIN_LO          # 504

def build_X_matrix(hour, days):
    """Feature matrix for a list of target days, shape (len(days), 101)."""
    days = list(days)
    n = len(days)
    X = np.zeros((n, 101))
    for c in range(21):
        last = (0, hour-LEAD-c) if hour >= LEAD+c else (1, 24+hour-LEAD-c)
        X[:, c] = [ID3(D - last[0], last[1]) for D in days]
    for c in range(25):
        last = (0, hour-c) if hour >= c else (1, 24+hour-c)
        X[:, 21+c] = [pricesda[D - last[0], last[1]] for D in days]
    for c in range(25):
        last = (0, hour-c) if hour >= c else (1, 24+hour-c)
        X[:, 46+c] = [wsum[D - last[0], last[1]] for D in days]
    last = (0, hour-LEAD) if hour >= LEAD else (1, 24+hour-LEAD)
    X[:, 71] = [wsumreal[D - last[0], last[1]] for D in days]
    X[:, 72] = [wsumreal[D - 1, hour] for D in days]
    for c in range(25):
        last = (0, hour-c) if hour >= c else (1, 24+hour-c)
        X[:, 73+c] = [load[D - last[0], last[1]] for D in days]
    last = (0, hour-LEAD) if hour >= LEAD else (1, 24+hour-LEAD)
    X[:, 98] = [loadreal[D - last[0], last[1]] for D in days]
    X[:, 99] = [loadreal[D - 1, hour] for D in days]
    X[:, 100] = [ID(D, hour, [12]) for D in days]
    return X


def main():
    train_days  = list(range(TRAIN_LO, TRAIN_HI))                   # 504 fixed train days
    target_days = [d + cal_len for d in range(forecast_days)]        # all 440 target days

    all_predictions = np.zeros((forecast_days, 24, 12))

    for hour in range(24):
        print(f"Hour {hour}")
        X_train   = build_X_matrix(hour, train_days)    # (504, 101)  — built once per hour
        X_targets = build_X_matrix(hour, target_days)   # (440, 101)

        a = np.median(X_train, 0); b = mad(X_train); b[b == 0] = 1.0
        X_train_n   = np.arcsinh((X_train   - a) / b)
        X_targets_n = np.arcsinh((X_targets - a) / b)

        for sub in range(12):
            Y_train = np.array([ID(D, hour, [sub]) for D in train_days], dtype=float)
            aY = np.median(Y_train); bY = mad(Y_train) or 1.0
            Y_train_n = np.arcsinh((Y_train - aY) / bY)

            model = LassoCV(cv=3, alphas=50, eps=1e-6, max_iter=2000)
            model.fit(X_train_n, Y_train_n)
            err = Y_train_n - model.predict(X_train_n)    # (504,) in-sample smearing residuals

            # vectorized smearing across all 440 target days at once
            preds_n = model.predict(X_targets_n)           # (440,)
            preds   = np.mean(
                np.sinh(preds_n[:, None] + err[None, :]) * bY + aY,
                axis=1
            )                                              # (440,)
            all_predictions[:, hour, sub] = preds

    # same output format as the original — the rest of main() is unchanged
    results = all_predictions.reshape(forecast_days * 24, 12)
    np.savetxt(home_dir + 'LASSO_12traj_stationary.csv', results, delimiter=',')

    # --- everything from lasso_traj = pd.read_csv(...) onward stays identical ---

    # Convert results to DataFrame and save
    lasso_traj = pd.read_csv(home_dir + 'LASSO_12traj_stationary.csv', header=None)
    lasso_10t = lasso_traj.iloc[:, 2:].copy()

    id_pred_df = pd.read_feather(home_dir + 'lasso_y.feather')
    id_true = id_pred_df.drop(['last_p', 'id3_p', 'id_1', 'id_2'], axis=1).copy()

    id_true = id_true.iloc[9528:, :].copy()  # 397*24
    id_true.reset_index(drop=True, inplace=True)

    lasso_pred = pd.concat([id_true.iloc[:, :2], lasso_10t], axis=1)

    # t1 means 3h before delivery...
    lasso_pred.columns = ['day', 'hour', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']
    id_true.columns = ['day', 'hour', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1']

    lasso_t = pd.melt(lasso_pred, id_vars=['day', 'hour'], var_name='traj', value_name='pred')
    id_t = pd.melt(id_true, id_vars=['day', 'hour'], var_name='traj', value_name='true')
    print(id_t.dtypes)

    lasso_t['traj'] = lasso_t['traj'].astype(int)
    lasso_t['day'] = lasso_t['day'].astype(int)
    lasso_t['hour'] = lasso_t['hour'].astype(int)
    print(lasso_t.dtypes)

    id_t['traj'] = id_t['traj'].astype(int)
    id_t['day'] = id_t['day'].astype(int)
    id_t['hour'] = id_t['hour'].astype(int)
    print(id_t.dtypes)

    lasso_xy = pd.concat([lasso_t, id_t[['true']]], axis=1)
    lasso_xy.dtypes

    lasso_xy.to_feather(home_dir + 'lasso_stationary_xy.feather')


if __name__ == "__main__":
    freeze_support()
    main()