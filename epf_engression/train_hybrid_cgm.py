import os
import random

# Configure deterministic TensorFlow operations before importing TensorFlow.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from latent_perturbation_cgm_models import cgm_engression_hybrid

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

data_dir = './'

loss_index_w = 0.0  # 0 if use only energy score as loss.
add_id_others = True

print('Model with 3 input parts, new model structure, add custom loss.')
print(f'\nloss = {1 - loss_index_w} * (ES/2) + {loss_index_w} * custom loss.')

# --------------------------- Read data ---------------------------------------
start_date = datetime(2017, 6, 14)

# intraday market, volumes data
volumesintra = np.zeros((837, 24, 14))
for n_hour in range(24):
    with open(data_dir + 'ID_DATA/' + f'volumes_hourly_{("0" + str(n_hour))[-2:]}') as f:
        lines = f.readlines()
        date_hour = [str(line.split(',')[0].strip()) for line in lines[1:]]
        data = np.array([[float(e) for e in line.strip().split(',')[1:]] for line in lines[1:]])
    date_3hour = data[:, :12]
    date_str = [datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%d')
                for date_string in date_hour]
    date_diff = [(datetime.strptime(date_string, '%Y%m%d') - start_date).days
                 for date_string in date_str]
    hour = np.zeros(837) + n_hour

    volumesintra[:, n_hour, 0] = np.array(date_diff) + 1  # date
    volumesintra[:, n_hour, 1] = hour
    volumesintra[:, n_hour, 2:] = date_3hour

volumesintra_flat = volumesintra.reshape(-1, volumesintra.shape[-1])

# intraday market, prices data, 24 arrays for each hour, shape (837 days, 128 time steps)
pricesintra = np.zeros((837, 24, 15))
for n_hour in range(24):
    with open(data_dir + 'ID_DATA/' + f'prices_hourly_{("0" + str(n_hour))[-2:]}') as f:
        lines = f.readlines()
        date_hour = [str(line.split(',')[0].strip()) for line in lines[1:]]
        data = np.array([[float(e) for e in line.strip().split(',')[1:]] for line in lines[1:]])
    date_3hour = data[:, :13]
    date_str = [datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%d')
                for date_string in date_hour]
    date_diff = [(datetime.strptime(date_string, '%Y%m%d') - start_date).days
                 for date_string in date_str]
    hour = np.zeros(837) + n_hour

    pricesintra[:, n_hour, 0] = np.array(date_diff) + 1  # date
    pricesintra[:, n_hour, 1] = hour
    pricesintra[:, n_hour, 2:] = date_3hour

pricesintra_flat = pricesintra.reshape(-1, pricesintra.shape[-1])

subprices = pricesintra_flat[:, 2:14]
subvolumes = volumesintra_flat[:, 2:]

id3_volume_sum = np.sum(subvolumes, axis=1)
id3_pricevolume_sum = np.sum((subprices * subvolumes), axis=1)
mean_subprice = np.mean(subprices, axis=1)

zero_volume_index = np.where(id3_volume_sum == 0)
nonzero_volume_index = ~np.isin(np.array(list(range(837 * 24))), zero_volume_index)

id3_vwa_price = np.zeros(837 * 24)
id3_vwa_price[nonzero_volume_index] = id3_pricevolume_sum[nonzero_volume_index] / id3_volume_sum[nonzero_volume_index]
id3_vwa_price[zero_volume_index] = mean_subprice[zero_volume_index]

id_price_true = subprices[:, 2:]
last_vwa_price = pricesintra_flat[:, 14]

id_pred = np.zeros((837 * 24, 16))  # (837*24, 14)
id_pred[:, :2] = pricesintra_flat[:, :2]
id_pred[:, 2] = id3_vwa_price  # ID3 price
id_pred[:, 3:] = pricesintra_flat[:, 2:15]  # including Last VWA ID price

exog_pred = np.zeros((837 * 24, 7))

# day ahead price, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'Day_Ahead_Epex.csv') as f:
    lines = f.readlines()
    date_str = [str(line.split(';')[0].strip()) for line in lines]
    hour = [int(line.split(';')[1].strip()) for line in lines]  # 1~24
    data = [float(line.split(';')[2].strip()) for line in lines]
    date_diff = [(datetime.strptime(date_string, '%Y%m%d') - start_date).days
                 for date_string in date_str]

exog_pred[:, 0] = np.array(date_diff)  # date
exog_pred[:, 1] = np.array(hour) - 1  # hour
exog_pred[:, 2] = np.array(data)  # day ahead price

# wind generation, offshore day-ahead forecasts, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'final_wind_offshore.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]
woff = np.array(data)

# wind generation, onshore day-ahead forecasts, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'final_wind_onshore.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]
won = np.array(data)

# wind generation, offshore real values/observations, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'final_wind_offshore_real.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]
woffreal = np.array(data)

# wind generation, onshore real values/observations, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'final_wind_onshore_real.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]
wonreal = np.array(data)

exog_pred[:, 3] = won + woff  # wind sum forecast
exog_pred[:, 4] = wonreal + woffreal  # wind sum real

# eletricity load (consumption) forecasts, shape (837 days, 24 hours)
with open(data_dir + 'EXOG_DATA/' + 'final_load_da.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]

exog_pred[:, 5] = np.array(data)  # load forecast

# eletricity load (consumption) real values/observations, shape (837 days, 24 hours)
loadreal = np.zeros((837, 24))
with open(data_dir + 'EXOG_DATA/' + 'final_load_real.csv') as f:
    lines = f.readlines()
    data = [float(line.strip()) for line in lines]

exog_pred[:, 6] = np.array(data)  # load real

id_pred_df = pd.DataFrame(id_pred)
id_pred_df.columns = ['day', 'hour', 'id3_p', 'id_1', 'id_2', 'id_3', 'id_4', 'id_5',
                      'id_6', 'id_7', 'id_8', 'id_9', 'id_10', 'id_11', 'id_12', 'last_p']
id_pred_df.to_feather(data_dir + 'lasso_y.feather')

exog_pred_df = pd.DataFrame(exog_pred)
exog_pred_df.columns = ['day', 'hour', 'da_p', 'w_pred', 'w_real', 'l_pred', 'l_real']
pred_combine_df = pd.merge(exog_pred_df, id_pred_df, how="outer", on=["day", "hour"])

dayhour_df = pred_combine_df[['day', 'hour', 'id_3', 'id_4',
                              'id_5', 'id_6', 'id_7', 'id_8',
                              'id_9', 'id_10', 'id_11', 'id_12']]

pred_combine_arr = pred_combine_df.values

# normalization of data
cal_len_norm = 837 - 200

pred_combine_norm = pred_combine_arr.copy()
for i in range(2, 7):
    data_i = pred_combine_arr[:, i]
    data_cal = pred_combine_arr[:cal_len_norm * 24, i]
    mu = data_cal.mean()
    sigma = data_cal.std()
    data_norm = (data_i - mu) / sigma
    pred_combine_norm[:, i] = data_norm

# normalization of observation data
data_id_cal = pred_combine_arr[:cal_len_norm * 24, 8:]
data_id_mu = data_id_cal.mean()
data_id_sigma = data_id_cal.std()
del data_cal, data_i, data_norm, data_id_cal

pred_combine_norm[:, 7:] = (pred_combine_arr[:, 7:] - data_id_mu) / data_id_sigma

pred_combine_norm_df = pd.DataFrame(pred_combine_norm)
pred_combine_norm_df.columns = pred_combine_df.columns

result_date = [start_date + timedelta(days=diff) for diff in pred_combine_norm_df['day'].values]
pred_combine_norm_df.insert(0, 'date', result_date)

result_weekday = np.array([date.weekday() + 1 for date in result_date]).astype('int')
pred_combine_norm_df.insert(1, 'weekday', result_weekday)

result_weekofyear = np.array([date.strftime('%V') for date in result_date]).astype('int')
pred_combine_norm_df.insert(2, 'weekofyear', result_weekofyear)

result_dayofyear = np.array([date.strftime('%j') for date in result_date]).astype('int')
pred_combine_norm_df.insert(3, 'dayofyear', result_dayofyear)

pred_combine_norm_df = pred_combine_norm_df.drop('day', axis=1)
pred_combine_norm_df = pred_combine_norm_df.drop('date', axis=1)

sin_hod = np.sin((pred_combine_norm_df['hour'].values / 24) * 2 * np.pi)
cos_hod = np.cos((pred_combine_norm_df['hour'].values / 24) * 2 * np.pi)
sin_woy = np.sin((pred_combine_norm_df['weekofyear'].values / 52) * 2 * np.pi)
cos_woy = np.cos((pred_combine_norm_df['weekofyear'].values / 52) * 2 * np.pi)
sin_doy = np.sin((pred_combine_norm_df['dayofyear'].values / 365) * 2 * np.pi)
cos_doy = np.cos((pred_combine_norm_df['dayofyear'].values / 365) * 2 * np.pi)

pred_combine_norm_df.insert(1, 'cos_doy', cos_doy)
pred_combine_norm_df.insert(1, 'sin_doy', sin_doy)
pred_combine_norm_df.insert(1, 'cos_hod', cos_hod)
pred_combine_norm_df.insert(1, 'sin_hod', sin_hod)
del cos_doy, sin_doy, cos_woy, sin_woy, cos_hod, sin_hod

pred_combine_norm_df = pred_combine_norm_df.drop('hour', axis=1)
pred_combine_norm_df = pred_combine_norm_df.drop('weekofyear', axis=1)
pred_combine_norm_df = pred_combine_norm_df.drop('dayofyear', axis=1)

pred_combine_norm_df['id_std'] = np.std(pred_combine_norm_df.iloc[:, 13:].values, axis=1)

print(pred_combine_norm_df.columns)
# 'weekday', 'sin_hod', 'cos_hod', 'sin_doy', 'cos_doy',
# 'da_p', 'w_pred', 'w_real', 'l_pred', 'l_real', 'id3_p',
# 'id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8',
# 'id_9', 'id_10', 'id_11', 'id_12', 'last_p', 'id_std'
pred_combine_norm_df.dtypes
aaa = pred_combine_norm_df.describe().transpose()

del data, date_3hour, date_diff, date_hour, date_str, hour, lines
del id3_pricevolume_sum, id3_volume_sum, id3_vwa_price
del exog_pred, exog_pred_df, id_pred, id_pred_df, last_vwa_price
del volumesintra, volumesintra_flat, subprices, subvolumes
del pricesintra, pricesintra_flat, result_date
del pred_combine_arr, pred_combine_df, pred_combine_norm

# 3 parts of inputs
LEAD = 4
data_len = pred_combine_norm_df.shape[0]
index_all = np.array(list(range(data_len)))
### Input 1:
# time series forecasting: h-LEAD..h-48/h-168
# 'da_p', 'w_pred', 'w_real', 'l_pred', 'l_real', 'id3_p',
# 'id_1', 'id_2', 'id_3', 'id_4', 'id_5', 'id_6', 'id_7', 'id_8',
# 'id_9', 'id_10', 'id_11', 'id_12', 'last_p', 'id_std'
input_ts = np.zeros((data_len, 20, 165))

for i in range(165):
    input_ts[:, :, i] = pred_combine_norm_df.iloc[(index_all - LEAD - i), 5:].values

### Input 3:
# all predictors input: h-LEAD
# 'da_p', 'w_pred', 'l_pred', 'last_p',
if add_id_others:
    input_all = np.zeros((data_len, 52))
else:
    input_all = np.zeros((data_len, 40))

input_all[:, :4] = pred_combine_norm_df.iloc[:, 1:5].values

for i in range(LEAD):
    input_all[:, 4 + i] = pred_combine_norm_df.iloc[(index_all - i), 5].values
    input_all[:, 8 + i] = pred_combine_norm_df.iloc[(index_all - i), 6].values
    input_all[:, 12 + i] = pred_combine_norm_df.iloc[(index_all - i), 8].values
    input_all[:, 16 + i] = pred_combine_norm_df.iloc[(index_all - i), -2].values

input_all[:, 20:40] = input_ts[:, :, 0]

if add_id_others:
    input_all[:, 40:44] = pred_combine_norm_df.iloc[(index_all - 2), 19:23].values
    input_all[:, 44:52] = pred_combine_norm_df.iloc[(index_all - 3), 15:23].values

# weekday separate as embeddings
input_weekday = pred_combine_norm_df.iloc[:, 0].values

### Input 2:
# conditional noise delta
input_std = input_ts[:, -1, :]

### Target ID prices
output_norm = pred_combine_norm_df.iloc[:, 13:23].values

start_index = 168  # 24*7
input_ts = input_ts[start_index:, :, :]
input_all = input_all[start_index:, :]
input_std = input_std[start_index:, :]
input_weekday = input_weekday[start_index:]
output_norm = output_norm[start_index:, :]
output_norm = np.expand_dims(output_norm, axis=2)  # (n, 10, 1)

best_step = np.argmax(output_norm, axis=1)  # (n, 1)
best_step = np.expand_dims(best_step, axis=2)  # (n, 1, 1)

y_all = np.concatenate([best_step, output_norm], axis=1)  # (n, 11, 1)

# Model hyperparameters
Nfeatures = input_all.shape[1]
Npast = 165
DIM_LATENT = 100
N_SAMPLES_TRAIN = 200  # number of samples drawn during training
N_SAMPLES_TEST = 1000  # 200

VERBOSE = 2
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 1e-4  # 'decay' or value 1e-4

n_ens = 10

print('\n batch size', BATCH_SIZE, '\n latent dim', DIM_LATENT,
      '\n learning rate', LEARNING_RATE, '\n #(training generated samples)', N_SAMPLES_TRAIN,
      '\n #(test generated samples)', N_SAMPLES_TEST, '\n #(model run)', n_ens)

# Split training, validation, and test data
train_end = (837 - 200 - 7) * 24

x_train = [input_ts[:(train_end), :, :],
           input_std[:(train_end), :],
           input_all[:(train_end), :],
           input_weekday[:(train_end)]]
x_test = [input_ts[(train_end):, :, :],
          input_std[(train_end):, :],
          input_all[(train_end):, :],
          input_weekday[(train_end):]]

y_train = y_all[:(train_end), :, :]
y_test = y_all[(train_end):, :, :]

predictions_list = []
# Model training and predicting
for ens in range(n_ens):
    print('\n hybrid model run', ens)
    tf.keras.backend.clear_session()

    # Use a distinct but repeatable training seed for every ensemble member.
    tf.keras.utils.set_random_seed(SEED + ens)

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        restore_best_weights=True
    )

    # Initialize hybrid model
    hybrid_init = cgm_engression_hybrid(
        dim_out=10,
        dim_in_features=Nfeatures,
        dim_in_past=Npast,
        dim_latent=DIM_LATENT,
        n_samples_train=N_SAMPLES_TRAIN,
        loss_weight=loss_index_w
    )

    # Fit model
    hybrid_init.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=VERBOSE,
        callbacks=[callback],
        validation_split=0.2,
        sample_weight=None,
        learningrate=LEARNING_RATE
    )

    print('finish training')

    # Predict and append to list
    # Reset the seed so latent samples generated during prediction are repeatable.
    tf.keras.utils.set_random_seed(SEED + 10_000 + ens)
    predictions = hybrid_init.predict(x_test, N_SAMPLES_TEST)
    predictions_list.append(predictions)

    # optional: save each ensemble member
    #hybrid_init.model.save(data_dir + 'saved_models/model_cgm_engression_hybrid_run' + str(ens) + '.keras')

# Concatenate the arrays in the list along the first axis (axis=0)
predictions_norm = np.concatenate(predictions_list, axis=2)
ens_fcst = data_id_sigma * predictions_norm + data_id_mu

print('shape of saved forecasts:', ens_fcst.shape)
#np.save(data_dir + 'pred_cgm_customloss.npy', ens_fcst)
np.save(data_dir+'/hybrid_outputs/pred_hybrid_esloss_2.npy', ens_fcst) # if using only energy score as loss.
