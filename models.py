import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge, LinearRegression
from xgboost.sklearn import XGBRegressor

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
    "NN": MLPClassifier,
}

reg_dict = {
    "LR": LinearRegression,
    "BR": BayesianRidge,
    "XGBR": XGBRegressor,
    "SGDR": SGDRegressor,
    "EN":  ElasticNet,
}


def model_choice_regression(clf, xtrain=None, ytrain=None):
    param_grid_nn = {
        "mlp__alpha": [0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2)]
    }
    if clf == 'NN':
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        solver="sgd",
                        hidden_layer_sizes=(8, 2),
                        random_state=1,
                        max_iter=500,
                    ),
                ),
            ]
        )

        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_nn, n_jobs=-1, cv=5)
        grid_search.fit(xtrain, ytrain)
        model = Pipeline([('scaling', StandardScaler())])
        model.steps.append(("mlp", MLPRegressor(
            solver="sgd",
            hidden_layer_sizes=grid_search.best_params_["mlp__hidden_layer_sizes"],
            random_state=1,
            max_iter=500,
            alpha=grid_search.best_params_["mlp__alpha"],
            learning_rate=grid_search.best_params_["mlp__learning_rate"],
        )))
    else:
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (clf, reg_dict[clf]())

            ]
        )
    return model


def get_best_classifier(x_train: np.ndarray,
                        y_train: np.ndarray,
                        x_valid: np.ndarray,
                        y_valid: np.ndarray) -> str:
    best_acc = 0
    best_model = None
    for model_name in clf_dict.keys():
        curr_clf = model_choice(model_name, x_train, y_train)
        curr_clf.fit(x_train, y_train)
        score = roc_auc_score(y_valid, curr_clf.predict_proba(x_valid)[:, 1])
        if score > best_acc:
            best_acc = score
            best_model = model_name
    print(f"best classifier {best_model} score: {best_acc}")
    return best_model


def get_best_regressor(x_train: np.ndarray,
                       y_train: np.ndarray,
                       x_valid: np.ndarray,
                       y_valid: np.ndarray) -> str:

    best_score = -1
    best_model = None
    for model_name in reg_dict:
        model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (model_name, reg_dict[model_name]())

            ]
        )
        model.fit(x_train, y_train)
        score = model.score(x_valid, y_valid)
        if score > best_score:
            best_score = score
            best_model = model_name
    print(f"best regressor score: {best_model} {best_score}")
    return best_model


def model_choice(clf, xtrain=None, ytrain=None):
    param_grid_knn = {
        "knn__n_neighbors": [3, 5, 7]
    }
    param_grid_nn = {
        "mlp__alpha": [0.05, 0.1],
        "mlp__learning_rate": ["constant", "adaptive"],
        'mlp__hidden_layer_sizes': [(8, 2)]
    }
    if clf == "XBG":
        model = make_pipeline(
            StandardScaler(), clf_dict[clf](objective="binary:logistic")
        )
    elif clf == "SVM":
        model = make_pipeline(
            StandardScaler(), clf_dict[clf](probability=True)
        )
    elif clf == "NN":
        temp_model = Pipeline(
            [
                ("scalar", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        solver="sgd",
                        hidden_layer_sizes=(8, 2),
                        random_state=1,
                        max_iter=500,
                    ),
                ),
            ]
        )

        print("running model search")
        grid_search = GridSearchCV(temp_model, param_grid_nn, n_jobs=-1, cv=5)
        grid_search.fit(xtrain, ytrain)
        model = Pipeline([('scaling', StandardScaler())])
        model.steps.append(("mlp", MLPClassifier(
            solver="sgd",
            hidden_layer_sizes=grid_search.best_params_["mlp__hidden_layer_sizes"],
            random_state=1,
            max_iter=500,
            alpha=grid_search.best_params_["mlp__alpha"],
            learning_rate=grid_search.best_params_["mlp__learning_rate"],
        )))
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