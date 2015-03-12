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

    driver_results = np.column_stack((np.ones((sorted_calibrated_probabilities.shape[0], 1))*driver,
                                      sorted_calibrated_probabilities))
    return driver_results


def classify_data(data):
    x, y, trip_id = split_data_target_id(data)

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
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Pre-processing
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # Training
        fits = {}
        for name, model in models.items():
            fits[name] = model.fit(x_train, y_train)

        # Evaluation
        for name, model in fits.items():
            probabilities = model.predict_proba(x_test)[:, 1]
            cv_scores[name].append(metrics.roc_auc_score(y_test, probabilities))

    avg_scores = {}
    for name in models.keys():
        avg_scores[name] = np.array(cv_scores[name]).mean()

    # Final fit on complete data set
    best_name = pick_best_model(avg_scores)
    final_scale = preprocessing.StandardScaler().fit(x)
    x_final = final_scale.transform(x)
    final_fit = models[best_name].fit(x_final, y)

    # Prediction
    original_cases = y == 1
    x_predict = x[original_cases]
    trip_id_predict = trip_id[original_cases]
    x_predict = final_scale.transform(x_predict)

    probabilities = final_fit.predict_proba(x_predict)[:, 1]

    return np.column_stack((trip_id_predict, probabilities))


def split_data_target_id(data):
    x = data[:, 2:]
    y = data[:, 0]
    trip_id = data[:, 1]

    return x, y, trip_id


def pick_best_model(model_scores):
    keys = list(model_scores.keys())
    values = list(model_scores.values())
    return keys[values.index(max(values))]


if __name__ == '__main__':
    data = feature_extraction.build_data_set(1)
    probs = classify_data(data)