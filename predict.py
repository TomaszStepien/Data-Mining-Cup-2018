"""trains final model on full train.csv,
produces predictions for final submission
saves 1 file:
C:\\DMC_2018\\submission.csv
"""

import pandas as pd

from functions import *

TRAIN_PATH = "C:\\DMC_2018\\preprocessed_data\\train.csv"
TEST_PATH = "C:\\DMC_2018\\preprocessed_data\\test.csv"

train = read_train(TRAIN_PATH)
test = read_test(TEST_PATH)

# train model

# make predictions
