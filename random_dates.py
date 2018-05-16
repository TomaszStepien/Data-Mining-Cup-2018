"""assignes each item a random date"""

import pandas as pd
from datetime import datetime
import random

from raw_data_types import items_types

items = pd.read_csv("C:\\DMC_2018\\raw_data\\items.csv", sep='|', dtype=items_types)

items = items.loc[:, ['pid', 'size']]


def gen_ran_date():
    day = random.choice(tuple(range(1, 29)))
    return datetime(2018, 2, day).date()


dates = [gen_ran_date() for i in range(items.shape[0])]
dates = pd.Series(dates)
items.loc[:, 'soldOutDate'] = dates
items.to_csv("C:\\DMC_2018\\preprocessed_data\\School_SGH_Warsaw_1.csv", sep='|', index=False)

print(items.head())
