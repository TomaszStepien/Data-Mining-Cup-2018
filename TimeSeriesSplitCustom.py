"""scikit's time series split does not provide enough
flexibility to handle our kind of data
"""

import pandas as pd


class TimeSeriesSplitCustom:
    """
    works like scikit's TimeSeriesSplit but has option to split at exact date
    DATASET NEEDS TO HAVE 'DATE' COLUMN
    """

    def __init__(self, split_dates):
        """

        :param split_dates: a list of string dates in form yyyy-mm-dd
                            first dates of TEST set
        """
        self.split_dates = split_dates

    def split(self, dataframe):
        """

        :param dataframe: dataset to be split
        """
        for date in self.split_dates:
            train_index = dataframe[dataframe['date'] < date].index
            test_index = dataframe[dataframe['date'] >= date].index

            yield (train_index, test_index)
