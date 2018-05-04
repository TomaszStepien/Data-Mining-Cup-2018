""" takes raw data and makes it good for modelling
saves s files:
C:\\DMC_2018\\preprocessed_data\\full.csv
C:\\DMC_2018\\preprocessed_data\\types.txt -- used to define types in pd.read_csv()
"""

import pandas as pd

from functions import hash_column
from functions import one_hot_column
from raw_data_types import items_types
from raw_data_types import prices_types
from raw_data_types import train_types

input_path = "C:\\DMC_2018\\raw_data\\"
output_path = "C:\\DMC_2018\\preprocessed_data\\"

print('reading data')
items = pd.read_csv(input_path + "items.csv", sep='|', dtype=items_types)
prices = pd.read_csv(input_path + "prices.csv", sep='|', dtype=prices_types)
train = pd.read_csv(input_path + "train.csv", sep='|', dtype=train_types)

# handle NaNs
print('handling NaNs')

# becase all sizez which are NA have unique PID and those PID have only one size(NA) we change them to 'unisize'
train['size'].fillna(value='unisize', inplace=True)
prices['size'].fillna(value='unisize', inplace=True)
items['size'].fillna(value='unisize', inplace=True)

items['subCategory'].fillna(0, inplace=True)
items['subCategory'] = items['subCategory'].astype('int64', inplace=True)

# PROBABLY REDUNTANT
# encode brand and color to category
# items['color'] = items['color'].astype('category')
# items['brand'] = items['brand'].astype('category')

# extend data so it also contains rows in which no sales occurred for a particular item
print('building final set')
dates = train['date'].unique()
full = pd.DataFrame(columns=('date', 'pid', 'size'))

for i in range(dates.shape[0]):
    data = {'date': pd.Series((dates[i] for d in range(items.shape[0]))),
            'pid': items['pid'],
            'size': items['size']}

    date_item_size = pd.DataFrame(data)
    full = full.append(date_item_size)

# join sets
full = pd.merge(left=full, right=train, how='left', on=('pid', 'size', 'date'))
full = pd.merge(left=full, right=items, how='left', on=('pid', 'size'))

# leave prices for now too speed things up
# full = pd.merge(left=full, right=prices, how='left', on=('pid', 'size'))

# create variable days_since release
full['days_since_release'] = (pd.to_datetime(full['date']) - pd.to_datetime(full['releaseDate'])).dt.days
full['days_since_release'] = full['days_since_release'].astype('int64')

# remove rows in which release_date is later than date.
# We do not want to confuse models with 0 sales for items that did not yet exist
# full = full.loc[full['date'] >= full['releaseDate'], :]
full = full.loc[full['days_since_release'] >= 0, :]
full = full.drop('releaseDate', axis=1)

# we have to reset index to avoid weid errors
full.reset_index(inplace=True, drop=True)

# change NaNs in units to 0s
full['units'].fillna(0, inplace=True)
full['units'] = full['units'].astype('int64')

# create variable weekday
# full['weekday'] = pd.to_datetime(full['date'])
# full['weekday'] = full['weekday'].dt.weekday  # 0 is monday, 6 is sunday

# # create variable day_of_month
# full['day_of_month'] = full['date'].str.split('-').str.get(2)

# create variable week of month
# full['week_of_month'] = pd.to_numeric(full['date'].str.split('-').str.get(2)) // 7

# hash categorical variables brand and color
print('hashing')
# full = hash_column(full, 'color', len(full['color'].unique()) // 2)
# full = hash_column(full, 'brand', len(full['brand'].unique()) // 2)

# one hot categories
# print('one hotting')
# full = one_hot_column(full, 'mainCategory')
# full = one_hot_column(full, 'weekday')
# full = one_hot_column(full, 'week_of_month')
# full = one_hot_column(full, 'category')
# full = one_hot_column(full, 'subCategory')

# add price variable from prices to full
rename_dict = {date: 'price' + date for date in prices.columns.values if date != 'pid' and date != 'size'}
prices.rename(columns=rename_dict, inplace=True)
prices = pd.wide_to_long(prices, stubnames='price', i=['pid', 'size'], j='date')
prices.reset_index(inplace=True)
full = pd.merge(left=full, right=prices, how='left', on=('pid', 'size', 'date'))

# add rrp - price
full['rrp_minus_price'] = full['rrp'] - full['price']

# add rrp/price
full['rrp_div_price'] = full['rrp'] / full['price']

# save to csv
print('saving')
full.to_csv(output_path + 'full.csv', sep='|', index=False)
print(full.info())

print(full.head(10))

# save types to text file
file = open(output_path + 'types.txt', 'w')
for t in list(full.columns.values):
    file.write(str(t) + ',' + str(full[t].dtype) + '\n')
