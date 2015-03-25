__author__ = 'nikosteinhoff'

import numpy as np
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


class Model:
    def __init__(self, estimator, name):
        self.estimator = estimator
        self.name = name
        self.scores = []
        self.avg_score = 0
        self.count = 0
        self.fitted = None

    def get_description(self):
        return self.name

    def get_score(self):
        return np.array(self.scores).mean()

    def get_variance(self):
        return np.array(self.scores).var()

    def fit_model(self, x, y):
        self.fitted = self.estimator.fit(x, y)

    def score_model(self, y_true, y_predict):
        self.scores.append(roc_auc_score(y_true, y_predict))
        self.avg_score = np.array(self.scores).mean()

    def predict_probabilities(self, x):
        assert self.fitted is not None, "Model has not been fitted yet!"
        return self.fitted.predict_proba(x)


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

