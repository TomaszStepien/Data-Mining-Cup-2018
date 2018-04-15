import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

from models import read_types

output_path = "C:\\DMC_2018\\model_summaries\\"
data_path = "C:\\DMC_2018\\preprocessed_data\\full.csv"

types = read_types()
full = pd.read_csv(data_path, sep='|', dtype=types)

train_vars = ('rrp',
              'stock',
              'brand_hash_0',
              'brand_hash_1',
              'brand_hash_2',
              'brand_hash_3',
              'brand_hash_4',
              'brand_hash_5',
              'brand_hash_6',
              'brand_hash_7',
              'brand_hash_8',
              'brand_hash_9',
              'brand_hash_10',
              'brand_hash_11',
              'color_hash_0',
              'color_hash_1',
              'color_hash_2',
              'color_hash_3',
              'color_hash_4',
              'color_hash_5',
              'color_hash_6',
              'color_hash_7',
              'mainCategory_onehot_1',
              'mainCategory_onehot_9',
              'mainCategory_onehot_15',
              )
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=len(train_vars)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
model.fit(full.loc[:, train_vars], full.loc[:, 'units'], epochs=1, batch_size=32)
