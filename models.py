"""takes preprocessed data and validates models
saves 1 file:
C:\\DMC_2018\\model_summaries\\[ERROR]_[ALGORITHM]_[TIMESTAMP].txt"
    -- file containing validation results

"""

import itertools
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
    'day_of_month': 'int64',
    'brand_0': 'float64',
    'brand_1': 'float64',
    'brand_2': 'float64',
    'brand_3': 'float64',
    'brand_4': 'float64',
}

full = pd.read_csv(data_path, sep='|', dtype=types)

full.loc[full['units'] > 1, 'units'] = 1

# select train variables
train_vars = ('rrp',
              'stock',
              'weekday',
              'day_of_month',
              'brand_0',
              'brand_1',
              'brand_2',
              'brand_3',
              'brand_4',)


# define error function
def calculate_error(series1, series2):
    return np.sqrt(np.sum(np.abs(series1 - series2)))


# set up time series crossvalidation
split_dates = ('2017-11-01', '2017-12-01', '2018-01-01')
tscv = TimeSeriesSplitCustom(split_dates=split_dates)

print('started gridsearch')
param_grid = {
    # 'n_estimators': 100,
    'max_depth': (2, 4, 8)
}

a = [param_grid[k] for k in param_grid]

# get all combinations of parameters
values_comb = (list(itertools.product(*a)))
keys = list(param_grid.keys())
counter = 1
for values in values_comb:
    print("training: " + str(counter) + " of " + str(len(values_comb)))
    counter += 1
    parameters = {}
    for i in range(len(values)):
        parameters[keys[i]] = values[i]

    # build model
    # parameters = {
    #     'n_estimators': 3,
    #     'max_depth': 2
    # }

    regr = RandomForestClassifier(**parameters)
    regr.set_params(n_jobs=-1)
    regr.set_params(n_estimators=3)

    all_zero_errors = []
    model_errors = []
    for train_index, test_index in tscv.split(full):
        X_train, X_test = full.iloc[train_index, :], full.iloc[test_index, :]
        y_train, y_test = full.loc[train_index, 'units'], full.loc[test_index, 'units']

        # calculate all 0 benchmark error on test set
        all_zero_errors.append(calculate_error(np.zeros(len(y_test), dtype='int'), y_test))

        # fit model
        regr.fit(X_train.loc[:, train_vars], y_train)
        test_preds = regr.predict(X_test.loc[:, train_vars])
        model_errors.append(calculate_error(test_preds, y_test))

    # create file to write performance stats to
    algorithm = 'RF'
    date = dt.now().date()
    time = dt.now().time()
    name = str(np.round(np.mean(model_errors), 3)) + '_' \
           + str(date).replace('-', '') + '_' \
           + str(time)[:8].replace(':', '') + '_' \
           + algorithm + '.txt'

    file = open(output_path + name, "w")

    file.write('dates used to split train/test: \n' + ', '.join(split_dates))
    file.write('\n\nvariables used: ')
    for t in train_vars:
        file.write("\n" + str(t))
    file.write("\n\nall zero benchmark: " + ', '.join((str(e) for e in all_zero_errors)))
    file.write("\nalgorithm performance: " + ', '.join((str(e) for e in model_errors)))
    file.write("\n\nmodel parameters: ")
    params = regr.get_params()
    for param in params:
        file.write("\n" + param + ': ' + str(params[param]))
    file.write("\n\nlast run predictions (actual vs predicted): ")
    for t, t1 in zip(test_preds[:10000], full['units']):
        file.write("\n" + str(t1) + '\t\t' + str(t))
    file.close()
