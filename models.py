import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
clf_dict = {
    "LR": LogisticRegression,
    "GB": GradientBoostingClassifier,
    "XGB": xgb.XGBClassifier,
    "KNN": KNeighborsClassifier,
    "RF": RandomForestClassifier,
}

reg_dict = {
    "LR": LinearRegression,
    "BR": BayesianRidge,
    "xgb": XGBRegressor,
    "kernel": KernelRidge,
    "SVR": SVR,
    "sgd": SGDRegressor,
    "en":  ElasticNet,
    "RFR": RandomForestRegressor
}

def get_best_classifier(X_train: np.array,
                       y_train: np.array,
                       x_valid: np.array,
                       y_valid:np.array) -> str:
    best_acc = 0
    best_model = None
    for model_name in clf_dict.keys():
        curr_clf = model_choice(model_name, X_train, y_train)
        curr_clf.fit(X_train, y_train)
        score = roc_auc_score(y_valid, curr_clf.predict_proba(x_valid)[:, 1])
        print(model_name, score)
        if score > best_acc:
            best_acc = score
            best_model = model_name
    print(f"best classifier {best_model} score: {best_acc}")
    return best_model

def get_best_regressor(X_train: np.array,
                       y_train: np.array,
                       x_valid: np.array,
                       y_valid: np.array) -> str:

    best_score = -1
    best_model = None
    for model_name in reg_dict:
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (model_name, reg_dict[model_name]())

            ]
        )
        model.fit(X_train, y_train)
        score = model.score(x_valid, y_valid)
        if score > best_score:
            best_score = score
            best_model = model_name
    print(f"best regressor score: {best_score}")
    return best_model

def model_choice(clf, xtrain=None, ytrain=None):
    param_grid_knn = {
        "knn__n_neighbors": [3, 5, 7]
    }
    if clf == "XBG":
        model = make_pipeline(
            StandardScaler(), clf_dict[clf](objective="binary:logistic")
        )
    elif clf == "KNN":
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        print("running model search")
        grid_search = GridSearchCV(model, param_grid_knn, n_jobs=-1, cv=5)

        grid_search.fit(xtrain, ytrain)
        # final model
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("knn", KNeighborsClassifier(grid_search.best_params_["knn__n_neighbors"])),
            ]
        )
    else:
        model = make_pipeline(StandardScaler(), clf_dict[clf]())
    return model