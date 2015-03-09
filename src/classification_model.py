__author__ = 'nikosteinhoff'

import sklearn.preprocessing as pre_proc
import sklearn.linear_model.logistic as log_reg
import numpy as np


def classify_data(data):
    scaler = pre_proc.StandardScaler()
    scaler_fit = scaler.fit(data[:, 2:])

    X = scaler_fit.transform(data[:, 2:])
    y = data[:, 0]

    model = log_reg.LogisticRegression()

    model_fit = model.fit(X, y)

    org_data = data[data[:, 0] == 1]
    org_X = scaler_fit.transform(org_data[:, 2:])
    org_y = org_data[:, 0]

    probabilities = model_fit.predict_proba(org_X)
    accuracy = model_fit.score(org_X, org_y)

    return np.column_stack((org_data[:, 1], probabilities[:, 1]))