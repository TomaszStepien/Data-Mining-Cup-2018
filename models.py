"""takes preprocessed data and validates models
saves 1 file:
C:\\DMC_2018\\model_summaries\\[ERROR]_[ALGORITHM]_[TIMESTAMP].txt"
    -- file containing validation results
"""


import pandas as pd
import datetime as dt
from functions import read_train

TRAIN_PATH = "C:\\DMC_2018\\preprocessed_data\\train.csv"
OUTPUT_PATH = "C:\\DMC_2018\\model_summaries\\"

# train = read_train(TRAIN_PATH)

# train model and validate

# save output to txt
error = 0.5342
algorithm = "RF"
now = ''.join(str(dt.date.today()).split('-'))

name = str(error) + '_' + algorithm + '_' + now + '.txt'

file = open(OUTPUT_PATH + name, "w")
file.write("Oh DI danny \n")
file.write("Oh hi Lisa \n")
file.close()
