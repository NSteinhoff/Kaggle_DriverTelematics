__author__ = 'nikosteinhoff'

from scipy import stats
import numpy as np


class Box_cox_transformer():
    def __init__(self):
        self.lmbdas = []
        self.shift = 1

    def fit(self, data):
        self.lmbdas = []

        data = np.absolute(data) + self.shift
        for i in range(data.shape[1]):
            column = data[:, i]
            self.fit_column(column)

    def fit_column(self, array):
        trans, l = stats.boxcox(np.absolute(array))
        self.lmbdas.append(l)

    def transform(self, data):
        data_trans = np.zeros((1, 1), dtype=float)

        if not self.lmbdas:
            self.fit(data)
        else:
            data = np.absolute(data) + self.shift

        for i in range(data.shape[1]):
            column = data[:, i]
            column_trans = stats.boxcox(np.absolute(column), self.lmbdas[i])

            if data_trans.size == 1:
                data_trans = np.copy(column_trans)
            else:
                data_trans = np.column_stack((data_trans, column_trans))

        return data_trans