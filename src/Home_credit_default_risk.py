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

application_test_df = pd.read_csv('./data/application_test.csv')

bureau_df = pd.read_csv('./data/bureau.csv')

bureau_balance_df = pd.read_csv('./data/bureau_balance.csv')

print(application_train_df.info())
print(application_train_df.describe())

bureau_balance_pivot_mean_df = bureau_balance_df.pivot_table(index='SK_ID_BUREAU', values='MONTHS_BALANCE', aggfunc=np.mean, fill_value=0)
bureau_balance_pivot_len_df = bureau_balance_df.drop('MONTHS_BALANCE', axis='columns').pivot_table(index='SK_ID_BUREAU', columns='STATUS', aggfunc=len, fill_value=0)
bureau_balance_pivot_df = pd.concat([bureau_balance_pivot_mean_df, bureau_balance_pivot_len_df], axis='columns')

bureau_mered_df = bureau_df.merge(bureau_balance_pivot_df, how='left', on='SK_ID_BUREAU')
bureau_pivot_mean_df_columns = ['SK_ID_CURR','DAYS_CREDIT','CREDIT_DAY_OVERDUE','DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT','AMT_CREDIT_MAX_OVERDUE','CNT_CREDIT_PROLONG','AMT_CREDIT_SUM','AMT_CREDIT_SUM_DEBT','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_UPDATE','AMT_ANNUITY']
bureau_pivot_mean_df_columns = bureau_pivot_mean_df_columns + bureau_balance_pivot_df.columns.tolist()
bureau_pivot_mean_df = bureau_mered_df[bureau_pivot_mean_df_columns].pivot_table(index='SK_ID_CURR', aggfunc=np.mean, fill_value=0)
bureau_pivot_CREDIT_ACTIVE_df = bureau_mered_df[['SK_ID_CURR','CREDIT_ACTIVE']].pivot_table(index='SK_ID_CURR', columns='CREDIT_ACTIVE', aggfunc=len, fill_value=0)
bureau_pivot_CREDIT_TYPE_df = bureau_mered_df[['SK_ID_CURR','CREDIT_TYPE']].pivot_table(index='SK_ID_CURR', columns='CREDIT_TYPE', aggfunc=len, fill_value=0)
bureau_pivot_df = pd.concat([bureau_pivot_mean_df, bureau_pivot_CREDIT_ACTIVE_df, bureau_pivot_CREDIT_TYPE_df], axis='columns')
bureau_pivot_df

y = application_train_df['TARGET']
train_df = application_train_df.merge(bureau_pivot_df, how='left', on='SK_ID_CURR')
train_df = train_df.drop(['SK_ID_CURR','TARGET'], axis='columns')


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

test_model = XGBClassifier(random_state=1234)
test_model.fit(X, y)
feature_importance = pd.DataFrame(test_model.feature_importances_, columns=["importance"], index=X.columns)
# feature_importance.sort_values("importance", ascending=False).plot(kind="bar", figsize=(100, 70))

# feature selection
important_feature = feature_importance.sort_values("importance", ascending=False)[0:190]
print(len(important_feature))
print(important_feature)

X = X[important_feature.index.tolist()]
#print(type(important_feature.index.tolist()))

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

study = optuna.create_study(direction='maximize', study_name='home_credit_default_risk', storage=get_storage(), load_if_exists=True)  # Create a new study.
study.optimize(objective, n_trials=50)  # Invoke optimization of the objective function.


