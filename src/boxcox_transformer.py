__author__ = 'nikosteinhoff'

from scipy import stats
import numpy as np


class BoxCoxTransformer():
    def __init__(self):
        self.lambdas = []
        self.shift = 1

    def fit(self, data):
        self.lambdas = []

        data = np.absolute(data) + self.shift
        for i in range(data.shape[1]):
            column = data[:, i]
            # noinspection PyArgumentList
            trans, l = stats.boxcox(column/column.mean())
            self.lambdas.append(l)

    def transform(self, data):
        data_trans = np.zeros((1, 1), dtype=float)

        if not self.lambdas:
            self.fit(data)
        else:
            data = np.absolute(data) + self.shift

        for i in range(data.shape[1]):
            column = data[:, i]
            column_trans = stats.boxcox(column/column.mean(), self.lambdas[i])

            if data_trans.size == 1:
                data_trans = np.copy(column_trans)
            else:
                data_trans = np.column_stack((data_trans, column_trans))

        return data_trans