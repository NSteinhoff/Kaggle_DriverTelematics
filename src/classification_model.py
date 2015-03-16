__author__ = 'nikosteinhoff'

import numpy as np
from src import feature_extraction
from src.boxcox_transformer import BoxCoxTransformer
from src.model import Model
from sklearn import preprocessing
from sklearn.linear_model import logistic
from sklearn import neighbors
from sklearn import cross_validation
from sklearn import ensemble
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.decomposition import PCA
from sklearn import pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

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


def classify_data_random_grid_search(data):
    x, y, trip_id = split_data_target_id(data)

    # Model specifications
    gb_estimators = [('pca', PCA(n_components='mle')), ('gradient_boosting', ensemble.GradientBoostingClassifier())]
    gb_parameter_grid = {
        'gradient_boosting__subsample': [0.7, 0.8, 0.9],
        'gradient_boosting__max_depth': [2, 3, 4]
    }
    gb_full_spec = RandomizedSearchCV(pipeline.Pipeline(gb_estimators),
                                      param_distributions=gb_parameter_grid, scoring='roc_auc')

    svc_estimators = [('pca', PCA(n_components='mle')), ('svc', SVC(probability=True))]
    svc_parameter_grid = {
        'svc__C': [1, 10, 100, 1000]
    }
    svc_full_spec = RandomizedSearchCV(pipeline.Pipeline(svc_estimators),
                                       param_distributions=svc_parameter_grid, scoring='roc_auc')

    model_specifications = {
        'gradient boosting': gb_full_spec,
        'SVC': svc_full_spec
    }

    models = {}
    for name, model in model_specifications.items():
        models[name] = Model(model, name)

    kf = cross_validation.StratifiedKFold(y, n_folds=5, random_state=123)
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Training
        for model in models.values():
            model.fit_model(x_train, y_train)

        # Evaluation
        for model in models.values():
            y_predict = model.predict_probabilities_gridsearch(x_test)[:, 1]
            model.score_model(y_test, y_predict)

    # Select the model with the best cv-auc
    best_model = pick_best_model(models)

    # Final fit on complete data set
    final_fit = best_model.estimator.fit(x, y)

    # Prediction
    original_cases = y == 1
    x_predict = x[original_cases]
    trip_id_predict = trip_id[original_cases]

    y_scores = final_fit.predict_proba(x_predict)[:, 1]

    trip_probabilities = np.column_stack((trip_id_predict, y_scores))

    return trip_probabilities, models


def classify_data(data):
    x, y, trip_id = split_data_target_id(data)

    use_boxcox_transform = False
    # Model specifications
    model_specifications = {
        # 'logistic_no_intercept': logistic.LogisticRegression(fit_intercept=False),
        # 'pca_logistic_no_intercept': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     logistic.LogisticRegression(fit_intercept=False)
        # ),
        #
        # 'nearest_5_neighbors_dist': neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance'),
        # 'pca_nearest_5_neighbors_dist': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance')
        # ),
        #
        # 'random_forest': ensemble.RandomForestClassifier(n_estimators=50),
        # 'pca_random_forest': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     ensemble.RandomForestClassifier(n_estimators=50)
        # ),
        #
        # 'gradient_boosting': ensemble.GradientBoostingClassifier(),
        # 'pca_gradient_boosting': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     ensemble.GradientBoostingClassifier()
        # ),
        #
        'gradient_boosting_stochastic': ensemble.GradientBoostingClassifier(subsample=0.8)
        # 'pca_gradient_boosting_stochastic': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     ensemble.GradientBoostingClassifier(subsample=0.8)
        # ),
        #
        # 'ada_boost_tree': ensemble.AdaBoostClassifier(),
        # 'pca_ada_boost_tree': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     ensemble.AdaBoostClassifier()
        # ),
        #
        # 'SVC_rbf_c2': SVC(C=2, kernel='rbf', probability=True),
        # 'pca_SVC_rbf_c2': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     SVC(C=2, kernel='rbf', probability=True)
        # ),
        # 'SVC_rbf': SVC(kernel='rbf', probability=True),
        # 'pca_SVC_rbf': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     SVC(kernel='rbf', probability=True)
        # ),
        # 'SVC_rbf_c4': SVC(C=4, kernel='rbf', probability=True),
        # 'pca_SVC_rbf_c4': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     SVC(C=4, kernel='rbf', probability=True)
        # ),
        #
        # 'naive_bayes_gaussian': naive_bayes.GaussianNB(),
        # 'pca_naive_bayes_gaussian': pipeline.make_pipeline(
        #     PCA(n_components='mle'),
        #     naive_bayes.GaussianNB(),
        # )
    }
    
    models = {}
    for name, model in model_specifications.items():
        models[name] = Model(model, name)

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


        # Normalizing
        scale = preprocessing.StandardScaler().fit(x_train)
        x_train = scale.transform(x_train)
        x_test = scale.transform(x_test)

        # Training
        for model in models.values():
            model.fit_model(x_train, y_train)

        # Evaluation
        for model in models.values():
            y_predict = model.fitted.predict_proba(x_test)[:, 1]
            model.score_model(y_test, y_predict)

    # Select the model with the best cv-auc
    best_model = pick_best_model(models)

    #Preprocessing
    # Remove skewness
    bc_transformer = BoxCoxTransformer()
    if use_boxcox_transform:
        bc_transformer.fit(x)
        x = bc_transformer.transform(x)

    final_scale = preprocessing.StandardScaler().fit(x)
    x_final = final_scale.transform(x)

    # Final fit on complete data set
    final_fit = best_model.estimator.fit(x_final, y)

    # Prediction
    original_cases = y == 1
    x_predict = x[original_cases]
    trip_id_predict = trip_id[original_cases]
    if use_boxcox_transform:
        x_predict = bc_transformer.transform(x_predict)
    x_predict = final_scale.transform(x_predict)

    y_scores = final_fit.predict_proba(x_predict)[:, 1]

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
    data = feature_extraction.build_data_set(1, False)
    probs = classify_data_random_grid_search(data)
    print("Done!")