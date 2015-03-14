__author__ = 'nikosteinhoff'

import numpy as np
import os
from src import file_handling
from src import feature_extraction
from src.model import Model
from sklearn import preprocessing
from sklearn.linear_model import logistic
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble


def calculate_driver(driver, mp=False):
    print("Calculating driver {0}".format(driver))

    data, descriptions = feature_extraction.build_data_set(driver)

    probabilities, models, feature_importance = classify_data(data, descriptions)

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()]

    calibration = np.linspace(0, 100, 200)
    calibrated_probabilities = np.column_stack((sorted_probabilities, calibration))

    sorted_calibrated_probabilities = calibrated_probabilities[calibrated_probabilities[:, 0].argsort()]

    driver_results = np.column_stack((np.ones((sorted_calibrated_probabilities.shape[0], 1))*driver,
                                      sorted_calibrated_probabilities))

    return driver_results, models, feature_importance


def classify_data(data, feature_descriptions):
    x, y, trip_id = split_data_target_id(data)

    # Model specifications
    models = {}

    # models['logistic'] = logistic.LogisticRegression()
    # models['logistic_no_intercept'] = logistic.LogisticRegression(fit_intercept=False)

    # models['nearest_neighbors_unif_3'] = neighbors.KNeighborsClassifier(n_neighbors=3)
    # models['nearest_neighbors_unif_5'] = neighbors.KNeighborsClassifier()
    # models['nearest_neighbors_unif_7'] = neighbors.KNeighborsClassifier(n_neighbors=7)
    # models['nearest_neighbors_dist_3'] = neighbors.KNeighborsClassifier(n_neighbors=3, weights='distance')
    # models['nearest_neighbors_dist_5'] = neighbors.KNeighborsClassifier(weights='distance')
    # models['nearest_neighbors_dist_3'] = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')

    models['random_forest_5'] = ensemble.RandomForestClassifier(n_estimators=50, max_depth=5)
    models['random_forest_7'] = ensemble.RandomForestClassifier(n_estimators=50, max_depth=7)
    models['random_forest_9'] = ensemble.RandomForestClassifier(n_estimators=50, max_depth=9)
    models['random_forest'] = ensemble.RandomForestClassifier(n_estimators=50)

    models['gradiant_boosting'] = ensemble.GradientBoostingClassifier()
    models['gradiant_boosting_stochastic'] = ensemble.GradientBoostingClassifier(subsample=0.8)

    models['ada_boost_tree'] = ensemble.AdaBoostClassifier()
    models['ada_boost_tree_100'] = ensemble.AdaBoostClassifier(n_estimators=100)

    model_objects = {}
    for name, model in models.items():
        model_objects[name] = Model(model, name)

    feature_coeficients = []
    for feature in feature_descriptions:
        feature_coeficients.append([])

    kf = cross_validation.StratifiedKFold(y, n_folds=5, random_state=123)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Pre-processing
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # Training
        for model in model_objects.values():
            model.fit_model(x_train, y_train)

        # Evaluation
        for model in model_objects.values():
            y_predict = model.fitted.predict_proba(x_test)[:, 1]
            model.score_model(y_test, y_predict)

        # Feature importance
        for model in model_objects.values():
            model_feature_coefs = model.fitted.feature_importances_
            for i in range(len(model_feature_coefs)):
                feature_coeficients[i].append(model_feature_coefs[i])

    feature_importance = {}
    for i in range(len(feature_descriptions)):
        feature_importance[feature_descriptions[i]] = np.array(feature_coeficients[i]).mean()

    # Final fit on complete data set
    best_model = pick_best_model_object(model_objects)
    model_objects[best_model.name].count += 1

    final_scale = preprocessing.StandardScaler().fit(x)
    x_final = final_scale.transform(x)
    final_fit = best_model.estimator.fit(x_final, y)

    # Prediction
    original_cases = y == 1
    x_predict = x[original_cases]
    trip_id_predict = trip_id[original_cases]
    x_predict = final_scale.transform(x_predict)

    y_scores = final_fit.predict_proba(x_predict)[:, 1]

    return np.column_stack((trip_id_predict, y_scores)), model_objects, feature_importance


def split_data_target_id(data):
    x = data[:, 2:]
    y = data[:, 0]
    trip_id = data[:, 1]

    return x, y, trip_id


def pick_best_model_object(model_objects):
    models = list(model_objects.values())
    avg_scores = [m.avg_score for m in models]
    return models[avg_scores.index(max(avg_scores))]


def pick_best_model(model_scores):
    keys = list(model_scores.keys())
    values = list(model_scores.values())
    return keys[values.index(max(values))]


if __name__ == '__main__':
    data, feature_desc = feature_extraction.build_data_set(1)
    probs = classify_data(data, feature_desc)