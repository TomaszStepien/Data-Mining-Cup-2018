""" takes raw data and makes it good for modelling
saves 1 file:
C:\\DMC_2018\\preprocessed_data\\full.csv
"""

import pandas as pd

from raw_data_types import items_types
from raw_data_types import prices_types
from raw_data_types import train_types

from sklearn.feature_extraction import DictVectorizer, FeatureHasher

PATH = "C:\\DMC_2018\\raw_data\\"

print('started reading')

items = pd.read_csv(PATH + "items.csv", sep='|', dtype=items_types)
prices = pd.read_csv(PATH + "prices.csv", sep='|', dtype=prices_types)
train = pd.read_csv(PATH + "train.csv", sep='|', dtype=train_types)

# handle NaNs
train['size'].fillna(value='unisize', inplace=True)
prices['size'].fillna(value='unisize', inplace=True)
items['size'].fillna(value='unisize', inplace=True)
items['subCategory'].fillna(0, inplace=True)
items['subCategory'] = items['subCategory'].astype('int64', inplace=True)

# encode brand and color to category
# items['color'] = items['color'].astype('category')
# items['brand'] = items['brand'].astype('category')

# extend data so it also contains rows in which no sales occurred for a particular item
dates = train['date'].unique()
full = pd.DataFrame(columns=('date', 'pid', 'size'))

for i in range(dates.shape[0]):
    data = {'date': pd.Series((dates[i] for d in range(items.shape[0]))),
            'pid': items['pid'],
            'size': items['size']}

    date_item_size = pd.DataFrame(data)
    full = full.append(date_item_size)

# join sets
print('started joining')
full = pd.merge(left=full, right=train, how='left', on=('pid', 'size', 'date'))
full = pd.merge(left=full, right=items, how='left', on=('pid', 'size'))

# leave prices for now too speed things up
# full = pd.merge(left=full, right=prices, how='left', on=('pid', 'size'))

# change NaNs in units to 0s
full['units'].fillna(0, inplace=True)
full['units'] = full['units'].astype('int64')

# create variable weekday
full['weekday'] = pd.to_datetime(full['date'])
full['weekday'] = full['weekday'].dt.weekday  # 0 is monday, 6 is sunday

# create variable day_of_month
full['day_of_month'] = full['date'].str.split('-').str.get(2)


# hash categorical variables brand and color
def hash_column(df, colname, n_columns):
    h = FeatureHasher(n_features=n_columns, input_type='string')
    rename_dict = {i: colname + '_hash_' + str(i) for i in range(n_columns)}
    f = h.transform(df[colname].astype('object'))
    hashed = pd.DataFrame(f.toarray())
    hashed.rename(columns=rename_dict, inplace=True)
    df = pd.concat([df, hashed], axis=1)
    df = df.drop(colname, axis=1)
    return df


full = hash_column(full, 'color', len(full['color'].unique()) // 2)
full = hash_column(full, 'brand', len(full['brand'].unique()) // 2)


# one hot categories
def one_hot_column(df, colname):
    dummies = pd.get_dummies(df[colname], colname + '_onehot')
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


full = one_hot_column(full, 'mainCategory')
# full = one_hot_column(full, 'category')
# full = one_hot_column(full, 'subCategory')


# save to csv
print('started saving')

full.to_csv('C:\\DMC_2018\\preprocessed_data\\full.csv', sep='|', index=False)

print(full.info())

# save types to text file
file = open("C:\\DMC_2018\\preprocessed_data\\types.txt", 'w')
for t in list(full.columns.values):
    file.write(str(t) + ',' + str(full[t].dtype) + '\n')
