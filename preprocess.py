""" takes raw data and makes it good for modelling
saves 2 files:
C:\\DMC_2018\\preprocessed_data\\train.csv -- with y, used to train models
C:\\DMC_2018\\preprocessed_data\\test.csv  -- without y, used to make final predictions
"""

import pandas as pd

PATH = "C:\\DMC_2018\\raw_data\\"

items = pd.read_csv(PATH + "items.csv", sep='|')
prices = pd.read_csv(PATH + "prices.csv", sep='|')
train = pd.read_csv(PATH + "train.csv", sep='|')

train['size'].fillna(value='unisize', inplace=True)
prices['size'].fillna(value='unisize', inplace=True)
items['size'].fillna(value='unisize', inplace=True)
items['subCategory'].fillna(0, inplace=True)
items['subCategory'] = items['subCategory'].astype('int64', inplace=True)

full = pd.merge(left=train, right=items, how='left', on=('pid', 'size'))
full = pd.merge(left=full, right=prices, how='left', on=('pid', 'size'))

full.to_csv('C:\\DMC_2018\\preprocessed_data\\full.csv', sep='|')

# train.to_csv("C:\\DMC_2018\\preprocessed_data\\train.csv")
# test.to_csv("C:\\DMC_2018\\preprocessed_data\\test.csv")
