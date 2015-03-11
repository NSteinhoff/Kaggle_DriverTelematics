__author__ = 'nikosteinhoff'

import numpy as np
from src import feature_extraction
from sklearn import preprocessing
from sklearn.linear_model import logistic
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import metrics
from sklearn import tree
from sklearn import svm


def classify_data(data):
    X, y, trip_id = split_data_target_id(data)


    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=123)

    # Model specifications
    models = {}
    models['logistic'] = logistic.LogisticRegression()
    models['neighbors'] = neighbors.KNeighborsClassifier()
    models['tree'] = tree.DecisionTreeClassifier(max_depth=10)

    # Preprocessing
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Trainig
    fits = {}
    for name, model in models.items():
        fits[name] = model.fit(X_train, y_train)

    # Evaluation
    scores = {}
    for name, model in fits.items():
        probs = model.predict_proba(X_test)[:, 1]
        scores[name] = metrics.roc_auc_score(y_test, probs)

    best_name = pick_best_model(scores)
    best_fit = fits[best_name]

    # Prediction
    X_pred = X[y == 1]
    y_pred = y[y == 1]
    trip_id_pred = trip_id[y == 1]
    X_pred = scaler.transform(X_pred)

    probabilities = best_fit.predict_proba(X_pred)[:, 1]

    return np.column_stack((trip_id_pred, probabilities))


def split_data_target_id(data):
    X = data[:, 2:]
    y = data[:, 0]
    trip_id = data[:, 1]

    return X, y, trip_id


def fit_model(X, y, model):
    fit = model.fit(X, y)


def pick_best_model(model_scores):
    keys = list(model_scores.keys())
    values = list(model_scores.values())
    return keys[values.index(max(values))]

if __name__ == '__main__':
    test_data = feature_extraction.build_data_set(1, mp=True)
    results = classify_data(test_data)
    print("Done!")