import numpy as np
import pandas as pd

data_dir = './'
data = pd.read_feather(data_dir + 'lasso_stationary_xy.feather')

arr = data.values
arr_re = np.reshape(arr, (10, 440, 24, 5))
# axis 0: 10 -> 1
# axis 1: 397 -> 837
# axis 3: 0 -> 23
# axis 4: day, hour, traj, pred, obs


test_lasso = arr_re[:, 240:, :, 3]  # (10, 200, 24)

train_lasso = arr_re[:, :, :, 3]  # (10, 440, 24)
train_true = arr_re[:, :, :, 4]  # (10, 440, 24)

train_diff = train_true - train_lasso  # (10, 440, 24)

N_traj = 10000
# lasso bootstrap
test_pred = np.zeros((10, 200, 24, N_traj))

CAL_LO, CAL_HI = 114, 240   # forecast indices for target days 511..636 (OOS, fixed)

for day in range(200):
    for hour in range(24):
        pred  = test_lasso[:, day, hour]                 # target 637..836
        preds = np.expand_dims(pred, 1)
        error_vector   = train_diff[:, CAL_LO:CAL_HI, hour]              # was day:(day+240)
        a              = np.random.choice(error_vector.shape[1], size=N_traj, replace=True)
        error_bootstrap = error_vector[:, a]
        test_pred[:, day, hour, :] = preds + error_bootstrap

test_pred_output = np.moveaxis(test_pred, 0, 2)
test_pred_save = np.reshape(test_pred_output, (4800, 10, N_traj))
test_pred_save = test_pred_save.astype(np.float32)

np.save(data_dir + 'lasso_bootstrap_stationary.npy', test_pred_save)


