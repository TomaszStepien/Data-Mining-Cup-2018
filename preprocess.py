""" takes raw data and makes it good for modelling
saves 2 files:
C:\\DMC_2018\\preprocessed_data\\train.csv -- with y, used to train models
C:\\DMC_2018\\preprocessed_data\\test.csv  -- without y, used to make final predictions
"""

import os
import pandas as pd

RAW_DATA_PATH = "C:\\DMC_2018\\raw_data"

print(os.listdir(RAW_DATA_PATH))

# train.to_csv("C:\\DMC_2018\\preprocessed_data\\train.csv")
# test.to_csv("C:\\DMC_2018\\preprocessed_data\\test.csv")
