""" contains functions

"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher


def hash_column(df, colname, n_columns):
    """
    Implements scikit's hashing trick,
    Can only hash columns that are of type object
    :param df: pd.DataFrame
    :param colname: column to be hashed
    :param n_columns: number of output hashed columns
    :return: pd.DataFrame with a column replaced with n hashed columns
    """

    h = FeatureHasher(n_features=n_columns, input_type='string')
    rename_dict = {i: colname + '_hash_' + str(i) for i in range(n_columns)}
    f = h.transform(df[colname])
    hashed = pd.DataFrame(f.toarray())
    hashed.rename(columns=rename_dict, inplace=True)
    df = pd.concat([df, hashed], axis=1)
    df = df.drop(colname, axis=1)
    return df


def one_hot_column(df, colname):
    """
    Implements panda's get_dummies
    :param df: pd.DataFrame
    :param colname: column to be onehoted
    :return: pd.DataFrame with a column replaced with dummy columns
    """
    dummies = pd.get_dummies(df[colname], colname + '_onehot')
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(colname, axis=1)
    return df


def read_types(path="C:\\DMC_2018\\preprocessed_data\\types.txt"):
    """

    :param path:
    :return: a dictrionary of types
    """
    types_file = open(path, 'r')
    return {f[:f.find(',')]: f[f.find(',') + 1:-1] for f in types_file.readlines()}


def calculate_error(series1, series2):
    return np.sqrt(np.sum(np.abs(series1 - series2)))
