# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from sklearn.datasets import make_regression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from utils.storage import get_storage
import optuna

# #### データ作成

X, y = make_regression(n_samples=10**4, n_features=20, n_informative=18, noise=0.0, random_state=1234)

X = pd.DataFrame(X)
y = pd.Series(y)

print(X.info())

# #### データ分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


# #### チューニング定義

def objective(trial):

    # モデルの種類
    regressor_name = trial.suggest_categorical('classifier', ['RandomForest', 'XGBoost', 'AdaBoost'])
    # max depthのチューニング(2〜24)
    max_depth = trial.suggest_int('max_depth', 2, 24)
    # n estimatorsのチューニング(50〜4000)
    n_estimators = trial.suggest_int('n_estimators', 50, 4000)
    # learning rateのチューニング(0.0001〜0.4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 0.4)
    
    if regressor_name == 'RandomForest':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=1234)
    elif regressor_name == 'XGBoost':
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=1234)
    elif regressor_name == 'AdaBoost':
        model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1234)
        
    error_list = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')

    return error_list.mean()


# #### チューニング開始

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner(), study_name='sample', storage=get_storage(), load_if_exists=True)
study.optimize(objective, n_trials=50) 

# +
# study = optuna.load_study(study_name='sample', storage=get_storage())
# -

# #### デフォルトパラメータとチューニングしたパラメータの比較

default_model = RandomForestRegressor(random_state=1234)
default_model.fit(X_train, y_train)
default_predict = default_model.predict(X_test)
default_score = mean_squared_error(y_test, default_predict)

# +
if study.best_trial.params["classifier"] == 'RandomForest':
    best_model = RandomForestRegressor(n_estimators=study.best_trial.params["n_estimators"], max_depth=study.best_trial.params["max_depth"], random_state=1234)
elif study.best_trial.params["classifier"] == 'XGBoost':
    best_model = XGBRegressor(n_estimators=study.best_trial.params["n_estimators"], max_depth=study.best_trial.params["max_depth"], learning_rate=study.best_trial.params["learning_rate"], random_state=1234)
elif study.best_trial.params["classifier"] == 'AdaBoost':
    best_model = AdaBoostRegressor(n_estimators=study.best_trial.params["n_estimators"], learning_rate=study.best_trial.params["learning_rate"], random_state=1234)
    
best_model.fit(X_train, y_train)
best_predict = best_model.predict(X_test)
best_score = mean_squared_error(y_test, best_predict)
# -

print(f'Score of dafault parameters => {default_score}')
print(f'Score of best parameters => {best_score}')
