""" contains functions

"""

import pandas as pd


def read_train(train_path):
    types = {}
    train = pd.read_csv(train_path,
                        dtype=types,
                        low_memory=True)
    return train


def read_test(test_path):
    types = {}
    test = pd.read_csv(test_path,
                       dtype=types,
                       low_memory=True)
    return test
