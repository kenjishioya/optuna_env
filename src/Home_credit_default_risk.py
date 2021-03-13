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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from utils.storage import get_storage
import optuna

application_train_df = pd.read_csv('./data/application_train.csv')

print(application_train_df.info())
print(application_train_df.describe())

y = application_train_df['TARGET']
train_df = application_train_df.drop(['SK_ID_CURR','TARGET'], axis='columns')


def process_train_df(i_df):
    train_df = i_df.copy()
    for column in train_df.columns:
        # categorical
        if train_df[column].dtype == 'object':
            # missing values
            if train_df[column].isnull().sum() > 0:
                train_df[column] = train_df[column].fillna('other')
            # label or one hot encoder
            if len(train_df[column].unique()) < 20:
                one_hot = pd.get_dummies(train_df[column], prefix=column)
                train_df = train_df.drop([column], axis='columns')
                train_df = train_df.join(one_hot)
            else:
                labelEncoder = LabelEncoder()
                train_df[column] = labelEncoder.fit_transform(train_df[column])
        elif train_df[column].dtype != 'object':
            if train_df[column].isnull().sum() > 0:
                train_df[column] = train_df[column].fillna(0)
    return train_df


X = process_train_df(train_df)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)


def objective(trial):

    # search better model from RandomForestRegressor, XGBRegressor
    regressor_name = trial.suggest_categorical('classifier', ['RandomForest', 'XGBoost'])
    # search better max_depth from 2 to 16
    max_depth = trial.suggest_int('max_depth', 2, 16)
    # search better n_estimators from 50 to 4000
    n_estimators = trial.suggest_int('n_estimators', 50, 4000)
    if regressor_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1234)
    else:
        model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, objective='binary:logistic', random_state=1234)

    
    
    error_list = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')

    return error_list.mean()  # An objective value linked with the Trial object.

study = optuna.create_study(direction='maximize', study_name='home_credit_default_risk2', storage=get_storage(), load_if_exists=True)  # Create a new study.
study.optimize(objective, n_trials=50)  # Invoke optimization of the objective function.

study.best_trial
