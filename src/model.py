__author__ = 'nikosteinhoff'


import numpy as np
from sklearn.metrics import roc_auc_score


class Model:
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name
        self.scores = []
        self.avg_score = 0
        self.count = 0
        self.fitted = None

    def get_score(self):
        return np.array(self.scores).mean()

    def get_variance(self):
        return np.array(self.scores).var()

    def fit_model(self, x, y):
        self.fitted = self.estimator.fit(x, y)

    def score_model(self, y_true, y_predict):
        self.scores.append(roc_auc_score(y_true, y_predict))
        self.avg_score = np.array(self.scores).mean()