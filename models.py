"""takes preprocessed data and validates models
saves 1 file:
C:\\DMC_2018\\model_summaries\\[ERROR]_[ALGORITHM]_[TIMESTAMP].txt"
    -- file containing validation results

"""

from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from TimeSeriesSplitCustom import TimeSeriesSplitCustom

output_path = "C:\\DMC_2018\\model_summaries\\"
data_path = "C:\\DMC_2018\\preprocessed_data\\full.csv"

# read preprocessed data
print('started reading')
types = {
    'date': 'object',
    'pid': 'int64',
    'size': 'object',
    'units': 'int64',
    'color': 'category',
    'brand': 'object',
    'rrp': 'float64',
    'mainCategory': 'int64',
    'category': 'int64',
    'subCategory': 'int64',
    'stock': 'int64',
    'releaseDate': 'object',
    'weekday': 'int64',
    'day_of_month': 'int64'
}
full = pd.read_csv(data_path, sep='|', dtype=types)


# define error function
def calculate_error(series1, series2):
    return np.sqrt(np.sum(np.abs(series1 - series2)))


# create file to write performance stats to
algorithm = 'RF'

date = dt.now().date()
time = dt.now().time()

name = str(date).replace('-', '') + '_' + str(time)[:8].replace(':', '') + '_' + algorithm + '.txt'

file = open(output_path + name, "w")

print('started training')

# set up time series crossvalidation
split_dates = ('2017-11-01', '2017-12-01', '2018-01-01')
tscv = TimeSeriesSplitCustom(split_dates=split_dates)

file.write('dates used to split train/test: \n' + ', '.join(split_dates))

# select variables to train
train_vars = ('rrp', 'stock', 'weekday', 'day_of_month')
file.write('\n\nvariables used: \n' + ', '.join(train_vars))

all_zero_errors = []
model_errors = []
regr = RandomForestRegressor(n_estimators=3, max_depth=2, random_state=0)

for train_index, test_index in tscv.split(full):
    X_train, X_test = full.iloc[train_index, :], full.iloc[test_index, :]
    y_train, y_test = full.loc[train_index, 'units'], full.loc[test_index, 'units']

    # calculate all 0 benchmark error on test set
    all_zero_errors.append(calculate_error(np.zeros(len(y_test), dtype='int'), y_test))

    # train model
    regr.fit(X_train.loc[:, train_vars], y_train)
    test_preds = regr.predict(X_test.loc[:, train_vars])
    model_errors.append(calculate_error(test_preds, y_test))

print(all_zero_errors)
print(model_errors)

file.write("\n\nall zero benchmark: " + ', '.join((str(e) for e in all_zero_errors)))
file.write("\nalgorithm performance: " + ', '.join((str(e) for e in model_errors)))
file.write("\n\nmodel parameters: ")
params = regr.get_params()
for param in params:
    file.write("\n" + param + ': ' + str(params[param]))
file.write("\n\nlast run predictions: ")
for t in test_preds[:10000]:
    file.write("\n" + str(t))
file.close()
