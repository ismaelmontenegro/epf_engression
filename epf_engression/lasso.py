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

LEAD = 4 # predict h with info until h-LEAD

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


def forecast(dayhour):
    # for LASSO
    #print(dayhour)
    day, hour = dayhour
    Nfeatures = 101
    X = np.zeros((cal_len - 1, Nfeatures))
    # 0..20:     ID3 of product h-LEAD..h-24 (for LEAD=4; h-26 for LEAD=6)
    # 21..45:    DA price for h..h-24
    # 45..70:    wind forecast for h..h-24
    # 71,72:     wind real for h-LEAD and h-24
    # 73..97:    load forecast for h..h-24
    # 98,99:     load real for h-LEAD and h-24
    # 100:       naive (last 15 min before h-LEAD)
    Xfut = np.zeros((1, Nfeatures))
    
    # cols 0..20, ID3 prices
    for c in range(21):
        last = (0, hour - LEAD - c) if hour >= LEAD + c else (1, 24 + hour - LEAD - c)
        X[:, c] = [ID3(d, last[1]) for d in range(day + 1 - last[0], day + cal_len - last[0])]
        Xfut[0, c] = ID3(day + cal_len - last[0], last[1])
    
    # cols 21..45, DA prices
    for c in range(25):
        last = (0, hour - c) if hour >= c else (1, 24 + hour - c)
        X[:, 21+c] = [pricesda[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
        Xfut[0, 21+c] = pricesda[day + cal_len - last[0], last[1]]
    
    # cols 46..70, wind forecasts
    for c in range(25):
        last = (0, hour - c) if hour >= c else (1, 24 + hour - c)
        X[:, 46+c] = [wsum[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
        Xfut[0, 46+c] = wsum[day + cal_len - last[0], last[1]]
    
    # cols 71,72 wind real for h-LEAD and h-24
    last = (0, hour - LEAD) if hour >= LEAD else (1, 24 + hour - LEAD)
    X[:, 71] = [wsumreal[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
    Xfut[0, 71] = wsumreal[day + cal_len - last[0], last[1]]
    last = (1, hour)
    X[:, 72] = [wsumreal[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
    Xfut[0, 72] = wsumreal[day + cal_len - last[0], last[1]]
    
    # cols 73..97 load forecasts
    for c in range(25):
        last = (0, hour - c) if hour >= c else (1, 24 + hour - c)
        X[:, 73+c] = [load[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
        Xfut[0, 73+c] = load[day + cal_len - last[0], last[1]]
    
    # cols 98,99 load real for h-LEAD and h-24
    last = (0, hour - LEAD) if hour >= LEAD else (1, 24 + hour - LEAD)
    X[:, 98] = [loadreal[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
    Xfut[0, 98] = loadreal[day + cal_len - last[0], last[1]]
    last = (1, hour)
    X[:, 99] = [loadreal[d, last[1]] for d in range(day + 1 - last[0], day + cal_len - last[0])]
    Xfut[0, 99] = loadreal[day + cal_len - last[0], last[1]]
    
    # col 100 naive
    naive = [ID(d, hour, [12]) for d in range(day + 1, day + cal_len)]
    naivef = ID(day + cal_len, hour, [12])
    X[:, 100] = naive
    Xfut[0, 100] = naivef

    # X done, now normalize and transform; Y prepared ad-hoc in the loop (12 different Ys)
    a = np.median(X, 0)
    b = mad(X)
    # Median + MAD should work fine, as no solar is included
    X = (X - a) / b
    Xfut = (Xfut - a) / b
    X = np.arcsinh(X)
    Xfut = np.arcsinh(Xfut)
    # Y normalization also ad-hoc
    # template for Y
    Y = np.zeros((cal_len-1,))
    predictions = np.zeros((12,))
    for sub in range(12):
        # Prepare Y given the subperiod, where sub=0 is 15min closest to delivery
        temp = [ID(d, hour, [sub]) for d in range(day + 1, day + cal_len)]
        Yh = Y.copy()
        Yh[:] = temp
        aY = np.median(Yh)
        bY = mad(Yh)
        Yh = (Yh - aY) / bY
        Yh = np.arcsinh(Yh)
        # betas = np.linalg.lstsq(X, Yh, rcond=None)[0]
        # err = Yh - np.dot(X, betas)
        # predictions[sub] = np.mean(np.sinh(np.dot(Xfut, betas) + err) * bY + aY)
        model = LassoCV(cv=3, alphas=50, eps=1e-6, max_iter=2000)
        model.fit(X, Yh)
        err = Yh - model.predict(X)
        predictions[sub] = np.mean(np.sinh(model.predict(Xfut)[0] + err) * bY + aY)

    return predictions

def main():
    tuples = [(d, h) for d in range(forecast_days) for h in range(24)]

    start = time.time()
    # result = forecast(tuples[0])
    with Pool() as pool:
        print(pool)
        results = np.array(pool.map(forecast, tuples))
    np.savetxt(home_dir + 'LASSO_12traj_new.csv', results, delimiter=',')

    end = time.time()
    duration = end - start
    print(results)
    print('Total Computation Time:', duration)

    # Convert results to DataFrame and save
    lasso_traj = pd.read_csv(home_dir+'LASSO_12traj_new.csv', header=None)
    lasso_10t = lasso_traj.iloc[:, 2:].copy()

    id_pred_df = pd.read_feather(home_dir+'lasso_y.feather')
    id_true = id_pred_df.drop(['last_p','id3_p','id_1','id_2'], axis=1).copy()

    id_true = id_true.iloc[9528:, :].copy() # 397*24
    id_true.reset_index(drop=True, inplace=True)

    lasso_pred = pd.concat([id_true.iloc[:, :2], lasso_10t], axis=1)

    # t1 means 3h before delivery...
    lasso_pred.columns = ['day','hour','10','9','8','7','6','5','4','3','2','1']
    id_true.columns = ['day','hour','10','9','8','7','6','5','4','3','2','1']

    lasso_t = pd.melt(lasso_pred, id_vars=['day','hour'], var_name='traj', value_name='pred')
    id_t = pd.melt(id_true, id_vars=['day','hour'], var_name='traj', value_name='true')
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

    lasso_xy.to_feather(home_dir+'lasso_xy.feather')

if __name__ == "__main__":
    freeze_support()
    main()