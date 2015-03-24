__author__ = 'nikosteinhoff'

import numpy as np
from src import feature_extraction
from src.boxcox_transformer import BoxCoxTransformer
import src.model_specifications as model_specs
from sklearn import preprocessing
from sklearn import cross_validation
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def calculate_driver(driver, rebuild_dataset=False):
    print("Calculating driver {0}".format(driver))

    data = feature_extraction.build_data_set(driver, rebuild_dataset)

    probabilities, models = classify_data(data)

    sorted_probabilities = probabilities[probabilities[:, 1].argsort()]

    calibration = np.linspace(0, 100, 200)
    calibrated_probabilities = np.column_stack((sorted_probabilities, calibration))

    sorted_calibrated_probabilities = calibrated_probabilities[calibrated_probabilities[:, 0].argsort()]

    driver_results = np.column_stack((np.ones((sorted_calibrated_probabilities.shape[0], 1))*driver,
                                      sorted_calibrated_probabilities))

    return driver_results, models


def classify_data(data):
    x, y, trip_id = split_data_target_id(data)

    use_boxcox_transform = False
    models = model_specs.models

    kf = cross_validation.StratifiedKFold(y, n_folds=5, random_state=123)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Remove skewness
        if use_boxcox_transform:
            bc_transformer = BoxCoxTransformer()
            bc_transformer.fit(x_train)
            x_train = bc_transformer.transform(x_train)
            x_test = bc_transformer.transform(x_test)
        # Normalization
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)
        # Training
        for model in models.values():
            model.fit_model(x_train, y_train)
        # Evaluation
        for model in models.values():
            y_predict = model.predict_probabilities(x_test)[:, 1]
            model.score_model(y_test, y_predict)

    # Select the model with the best cv-rocauc
    best_model = pick_best_model(models)
    best_model.fitted = None

    # Pre-processing
    # Remove skewness
    bc_transformer = BoxCoxTransformer()
    if use_boxcox_transform:
        bc_transformer.fit(x)
        x = bc_transformer.transform(x)
    final_scale = preprocessing.StandardScaler().fit(x)
    x_final = final_scale.transform(x)

    # Final fit on complete data set
    best_model.fit_model(x_final, y)

    # Prediction
    original_cases = y == 1
    x_predict = x[original_cases]
    trip_id_predict = trip_id[original_cases]
    if use_boxcox_transform:
        x_predict = bc_transformer.transform(x_predict)
    x_predict = final_scale.transform(x_predict)
    y_scores = best_model.predict_probabilities(x_predict)[:, 1]
    trip_probabilities = np.column_stack((trip_id_predict, y_scores))

    return trip_probabilities, models


def split_data_target_id(data):
    x = data[:, 2:]
    y = data[:, 0]
    trip_id = data[:, 1]

    return x, y, trip_id


def pick_best_model(model_objects):
    models = list(model_objects.values())
    avg_scores = [m.avg_score for m in models]
    best_model = models[avg_scores.index(max(avg_scores))]
    best_model.count = 1
    return best_model


def explore_data(data):
    plt.interactive(False)
    pp = PdfPages('plots.pdf')

    for i in range(data.shape[1]):
        column = data[:, i]
        plt.hist(column)
        pp.savefig()
        plt.clf()
    pp.close()
    return


if __name__ == '__main__':
    for i in calculate_driver(1, False):
        print(i)
    print("Done!")