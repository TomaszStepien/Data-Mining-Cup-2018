"""plots stuff
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from functions import read_types

plot_color = colors.hex2color('#ff00ff')

data_path = "C:\\DMC_2018\\preprocessed_data\\full.csv"

types = {'pid': 'int64',
         'size': 'object',
         'units': 'int64',
         'date': 'object',
         'stock': 'int64'}

full = pd.read_csv(data_path,
                   sep='|',
                   dtype=types,
                   usecols=['pid', 'size', 'units', 'date', 'stock'])

item_ids = full['pid'].unique()

for item_id in item_ids:
    sub_full = full[full['pid'] == item_id]
    sizes = sub_full['size'].unique()
    for size in sizes:
        sub_sub_full = sub_full[sub_full['size'] == size]
        fig, ax = plt.subplots(figsize=(18, 10))
        s1 = sns.barplot(x=sub_sub_full['date'], y=sub_sub_full['units'], ax=ax, color=plot_color)
        ax.plot(sub_sub_full['date'], sub_sub_full['stock'])
        l1 = s1.set_xticklabels(labels=s1.get_xticklabels(), rotation=90)
        plt.yticks(np.arange(min(min(sub_sub_full['units']), min(sub_sub_full['stock'])),
                             max(max(sub_sub_full['units']), max(sub_sub_full['stock'])) + 1, 1.0))

        size = size.replace('/', '_')
        size = size.replace('\\', '_')
        size = size.replace('?', '_')
        size = size.replace(':', '_')
        size = size.replace('*', '_')
        size = size.replace('"', '_')
        size = size.replace('<', '_')
        size = size.replace('>', '_')
        size = size.replace('|', '_')

        name = 'plots\\' \
               + str(item_id) \
               + '_' \
               + size + '.png'

        plt.savefig(name)
        plt.close()
