from datetime import datetime as dt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import LSTM
from keras.models import Sequential

from TimeSeriesSplitCustom import TimeSeriesSplitCustom
from functions import calculate_error
from functions import read_types

# from train_vars import train_vars
train_vars = ('size')

output_path = "C:\\DMC_2018\\model_summaries\\"
data_path = "C:\\DMC_2018\\preprocessed_data\\full.csv"

types = read_types()
full = pd.read_csv(data_path, sep='|', dtype=types)

# trim target variable for now
full.loc[full['units'] > 1, 'units'] = 1

# set up time series crossvalidation
split_dates = ('2017-11-01', '2017-12-01', '2018-01-01')
tscv = TimeSeriesSplitCustom(split_dates=split_dates)

# print('started gridsearch')
# param_grid = {
#     # 'n_estimators': 100,
#     'max_depth': (2, 4)
# }
#
# a = [param_grid[k] for k in param_grid]
#
# # get all combinations of parameters
# values_comb = (list(itertools.product(*a)))
# keys = list(param_grid.keys())
# counter = 1
# for values in values_comb:
# print("training: " + str(counter) + " of " + str(len(values_comb)))
# counter += 1
# parameters = {}
# for i in range(len(values)):
#     parameters[keys[i]] = values[i]

n_ids = full['size'].nunique()
le = LabelEncoder()
le.fit(full['size'])
full['size'] = le.transform(full['size'])

model = Sequential()
model.add(Embedding(input_dim=n_ids, output_dim=256, input_length=1))
# model.add(Flatten())
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

all_zero_errors = []
model_errors = []
for train_index, test_index in tscv.split(full):
    X_train, X_test = full.iloc[train_index, :], full.iloc[test_index, :]
    y_train, y_test = full.loc[train_index, 'units'], full.loc[test_index, 'units']

    # calculate all 0 benchmark error on test set
    all_zero_errors.append(calculate_error(np.zeros(len(y_test), dtype='int'), y_test))

    # fit model
    # Train the model, iterating on the data in batches of 32 samples
    model.fit(X_train.loc[:, train_vars], y_train, epochs=2, batch_size=16, verbose=0)

    test_preds = np.round(model.predict(X_test.loc[:, train_vars])[:, 0])
    model_errors.append(calculate_error(test_preds, y_test))

# create file to write performance stats to
algorithm = 'keras'
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

# save as JSON
json_string = model.to_json()
file.write(json_string)
file.write("\n\nlast run predictions (actual vs predicted): ")
for t, t1 in zip(test_preds[:10000], full['units']):
    file.write("\n" + str(t1) + '\t\t' + str(t))
file.close()
