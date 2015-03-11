__author__ = 'nikosteinhoff'

import numpy as np
from src import feature_extraction
from sklearn import preprocessing
from sklearn.linear_model import logistic
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble


def calculate_driver(driver, mp=False):
    print("Calculating driver {0}".format(driver))

    data = feature_extraction.build_data_set(driver, mp=mp)

    probabilities = classify_data(data)

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()]

    calibration = np.linspace(0, 100, 200)
    calibrated_probabilities = np.column_stack((sorted_probabilities, calibration))

    sorted_calibrated_probabilities = calibrated_probabilities[calibrated_probabilities[:, 0].argsort()]

    driver_results = np.column_stack((np.ones((sorted_calibrated_probabilities.shape[0], 1))*driver, sorted_calibrated_probabilities))
    return driver_results



def classify_data(data):
    X, y, trip_id = split_data_target_id(data)

    # Model specifications
    models = {}
    models['logistic'] = logistic.LogisticRegression()
    models['logistic_no_intercept'] = logistic.LogisticRegression(fit_intercept=False)

    models['nearest_neighbors_unif'] = neighbors.KNeighborsClassifier()
    models['nearest_neighbors_dist'] = neighbors.KNeighborsClassifier(weights='distance')

    models['tree_5'] = tree.DecisionTreeClassifier(max_depth=5)
    models['tree_10'] = tree.DecisionTreeClassifier(max_depth=10)
    models['tree_15'] = tree.DecisionTreeClassifier(max_depth=15)

    models['random_forest_5'] = ensemble.RandomForestClassifier(max_depth=5)
    models['random_forest_10'] = ensemble.RandomForestClassifier(max_depth=10)
    models['random_forest_15'] = ensemble.RandomForestClassifier(max_depth=15)



    cv_scores = {}
    for name in models.keys():
        cv_scores[name] = []

    kf = cross_validation.StratifiedKFold(y, n_folds=5)
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Preprocessing
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Trainig
        fits = {}
        for name, model in models.items():
            fits[name] = model.fit(X_train, y_train)

        # Evaluation
        for name, model in fits.items():
            probs = model.predict_proba(X_test)[:, 1]
            cv_scores[name].append(metrics.roc_auc_score(y_test, probs))

    avg_scores = {}
    for name in models.keys():
        avg_scores[name] = np.array(cv_scores[name]).mean()

    # Final fit on complete dataset
    best_name = pick_best_model(avg_scores)
    final_scaler = preprocessing.StandardScaler().fit(X)
    X_final = final_scaler.transform(X)
    final_fit = models[best_name].fit(X_final, y)

    # Prediction
    original_cases = y == 1
    X_pred = X[original_cases]
    trip_id_pred = trip_id[original_cases]
    X_pred = final_scaler.transform(X_pred)

    probabilities = final_fit.predict_proba(X_pred)[:, 1]

    return np.column_stack((trip_id_pred, probabilities))


def split_data_target_id(data):
    X = data[:, 2:]
    y = data[:, 0]
    trip_id = data[:, 1]

    return X, y, trip_id


def pick_best_model(model_scores):
    keys = list(model_scores.keys())
    values = list(model_scores.values())
    return keys[values.index(max(values))]


if __name__ == '__main__':
    data = feature_extraction.build_data_set(1)
    probs = classify_data(data)