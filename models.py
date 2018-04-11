"""takes preprocessed data and validates models
saves 1 file:
C:\\DMC_2018\\model_summaries\\[ERROR]_[ALGORITHM]_[TIMESTAMP].txt"
    -- file containing validation results

"""

import datetime as dt

import pandas as pd
import numpy as np

from TimeSeriesSplitCustom import TimeSeriesSplitCustom

output_path = "C:\\DMC_2018\\model_summaries\\"
data_path = "C:\\DMC_2018\\preprocessed_data\\full.csv"

# read preprocessed data
full = pd.read_csv(data_path, sep='|')

# set up time series crossvalidation
tscv2 = TimeSeriesSplitCustom(split_dates=['2017-12-01', '2017-11-01'])


# define error function
def calculate_error(series1, series2):
    pass


for train_index, test_index in tscv2.split(full):
    X_train, X_test = full.iloc[train_index, :], full.iloc[test_index, :]
    y_train, y_test = full.loc[train_index, 'units'], full.loc[test_index, 'units']

    # calcucalate all 0 benchmark error on test set

# save output to txt


error = 0.5342
algorithm = "RF"
now = ''.join(str(dt.date.today()).split('-'))
name = str(error) + '_' + algorithm + '_' + now + '.txt'
file = open(output_path + name, "w")
file.write("Oh DI danny \n")
file.write("Oh hi Lisa \n")
file.close()
