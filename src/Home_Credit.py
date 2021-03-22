# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import gc
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from utils.storage import get_storage
import optuna


class Home_Credit:
    APPLICATION_TRAIN_PATH = './data/application_train.csv'
    APPLICATION_TEST_PATH = './data/application_test.csv'
    BUREAU_PATH = './data/bureau.csv'
    BUREAU_BALANCE_PATH = './data/bureau_balance.csv'
    PREVIOUS_APPLICATION_PATH = './data/previous_application.csv'
    CREDIT_CARD_PATH = './data/credit_card_balance.csv'
    INSTALLMENTS_PAYMENTS_PATH = './data/installments_payments.csv'
    POS_CASH_BALANCE_PATH = './data/POS_CASH_balance.csv'

    def __init__(self, debug=False):
        self.debug = debug
        self.nrows = 10000 if debug == True else None
    
    def clear_memory(self, var_list):
        for variable in var_list:
            del variable
        gc.collect()
    
    def one_hot_encoding(self, df):
        cols = df.columns.tolist()
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
        # new_cols = [new_col for new_col in df.columns if new_col not in cols]
        return df #, new_cols
    
    def fill_zero_num_cols(self, df):
        num_cols = [col for col in df.columns if df[col].dtype != 'object']
        df.loc[:, num_cols] = df[num_cols].fillna(value=0)
        return df
    
    def convert_float64_to_float32(self, df):
        num_cols = [col for col in df.columns if df[col].dtype == 'float64']
        df.loc[:, num_cols] = df[num_cols].astype('float32')
        return df

    def preprocess_aplication(self):
        train_df = pd.read_csv(self.APPLICATION_TRAIN_PATH, nrows=self.nrows)
        test_df = pd.read_csv(self.APPLICATION_TEST_PATH, nrows=self.nrows)
        all_df = train_df.append(test_df)
        all_df = all_df[all_df['CODE_GENDER'] != 'XNA']
        for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
            all_df[bin_feature], uniques = pd.factorize(all_df[bin_feature])

        encoded_df = self.one_hot_encoding(all_df)
        encoded_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        encoded_df['DAYS_EMPLOYED_PERC'] = encoded_df['DAYS_EMPLOYED'] / encoded_df['DAYS_BIRTH']
        encoded_df['INCOME_CREDIT_PERC'] = encoded_df['AMT_INCOME_TOTAL'] / encoded_df['AMT_CREDIT']
        encoded_df['INCOME_PER_PERSON'] = encoded_df['AMT_INCOME_TOTAL'] / encoded_df['CNT_FAM_MEMBERS']
        encoded_df['ANNUITY_INCOME_PERC'] = encoded_df['AMT_ANNUITY'] / encoded_df['AMT_INCOME_TOTAL']
        encoded_df['PAYMENT_RATE'] = encoded_df['AMT_ANNUITY'] / encoded_df['AMT_CREDIT']
        self.clear_memory([train_df, test_df, all_df])
        return encoded_df
    
    def preprocess_breau(self):
        bureau_df = pd.read_csv(self.BUREAU_PATH, nrows=self.nrows)
        bureau_balance_df = pd.read_csv(self.BUREAU_BALANCE_PATH, nrows=self.nrows)
        encoded_bureau_df = self.one_hot_encoding(bureau_df)
        encoded_bb_df = self.one_hot_encoding(bureau_balance_df)
        # bureau_balance aggregate
        bureau_balance_agg_df = encoded_bb_df.groupby('SK_ID_BUREAU').agg('mean')
        bureau_merged_df = encoded_bureau_df.merge(bureau_balance_agg_df, how='left', on='SK_ID_BUREAU')
        bureau_merged_df = bureau_merged_df.drop('SK_ID_BUREAU', axis='columns')
        bureau_agg_df = bureau_merged_df.groupby('SK_ID_CURR').agg('mean')
        self.clear_memory([bureau_df, bureau_balance_df, encoded_bureau_df, encoded_bb_df, bureau_balance_agg_df, bureau_merged_df])
        return bureau_agg_df

    def preprocess_prev_application(self):
        prev_df = pd.read_csv(self.PREVIOUS_APPLICATION_PATH, nrows=self.nrows)
        encoded_prev_df = self.one_hot_encoding(prev_df)
        encoded_prev_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
        encoded_prev_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
        encoded_prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
        encoded_prev_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
        encoded_prev_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
        # Add feature: value ask / value received percentage
        encoded_prev_df['APP_CREDIT_PERC'] = encoded_prev_df['AMT_APPLICATION'] / encoded_prev_df['AMT_CREDIT']
        prev_agg_df = encoded_prev_df.groupby('SK_ID_CURR').agg('mean')
        prev_agg_df = prev_agg_df.drop('SK_ID_PREV', axis='columns')
        self.clear_memory([prev_df, encoded_prev_df])
        return prev_agg_df

    def preprocess_pos_cash(self):
        pos_df = pd.read_csv(self.POS_CASH_BALANCE_PATH, nrows=self.nrows)
        encoded_pos_df = self.one_hot_encoding(pos_df)
        pos_agg_df = encoded_pos_df.groupby('SK_ID_CURR').agg('mean')
        pos_agg_df['POS_COUNT'] = encoded_pos_df.groupby('SK_ID_CURR').size()
        pos_agg_df = pos_agg_df.drop('SK_ID_PREV', axis='columns')
        self.clear_memory([pos_df, encoded_pos_df])
        return pos_agg_df

    def preprocess_installments_df(self):
        install_df = pd.read_csv(self.INSTALLMENTS_PAYMENTS_PATH, nrows=self.nrows)
        encoded_install_df = self.one_hot_encoding(install_df)
        install_agg_df = encoded_install_df.groupby('SK_ID_CURR').agg('mean')
        install_agg_df['INSTALL_COUNT'] = encoded_install_df.groupby('SK_ID_CURR').size()
        install_agg_df = install_agg_df.drop('SK_ID_PREV', axis='columns')
        self.clear_memory([install_df, encoded_install_df])
        return install_agg_df

    def preprocess_credit_card_df(self):
        credit_card_df = pd.read_csv(self.CREDIT_CARD_PATH, nrows=self.nrows)
        encoded_credit_card_df = self.one_hot_encoding(credit_card_df)
        credit_card_agg_df = encoded_credit_card_df.groupby('SK_ID_CURR').agg('mean')
        credit_card_agg_df['CREDIT_COUNT'] = encoded_credit_card_df.groupby('SK_ID_CURR').size()
        credit_card_agg_df = credit_card_agg_df.drop('SK_ID_PREV', axis='columns')
        self.clear_memory([credit_card_df, encoded_credit_card_df])
        return credit_card_agg_df

    def objective_multi_classifiers(self, trial):
        # search better model from RandomForestRegressor, XGBRegressor
        classifier_name = trial.suggest_categorical('classifier', ['RandomForest', 'XGBoost', 'LGBM'])
        # search better max_depth from 2 to 16
        max_depth = trial.suggest_int('max_depth', 2, 16)
        # search better n_estimators from 50 to 4000
        n_estimators = trial.suggest_int('n_estimators', 50, 7000)
        if classifier_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1234)
        elif classifier_name == 'XGBoost':
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, objective='binary:logistic', random_state=1234)
        else:
            model = LGBMClassifier(boosting_type='goss',n_estimators=n_estimators, max_depth=max_depth, objective='binary', num_leaves=34, random_state=1234)
        
        error_list = cross_val_score(model, self.X, self.y, cv=3, scoring='roc_auc')
        gc.collect()
        return error_list.mean()
    
    def objective_LGBM(self, trial):
        # hyper parameters for tuning
        n_estimators = trial.suggest_int('n_estimators', 50, 10000)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.5)
        max_depth = trial.suggest_int('max_depth', 2, 16)
        num_leaves = trial.suggest_int('num_leaves', 10, 50)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.01, 1)
        subsample = trial.suggest_float('subsample', 0.01, 1)
        reg_alpha = trial.suggest_float('reg_alpha', 0.001, 0.1)
        reg_lambda = trial.suggest_float('reg_lambda', 0.001, 0.1)
        min_split_gain = trial.suggest_float('min_split_gain', 0.001, 0.1)
        min_child_weight = trial.suggest_float('min_child_weight', 0.001, 50)
        
        model = LGBMClassifier(
            boosting_type='goss',
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            objective='binary',
            random_state=1234)
        
        error_list = cross_val_score(model, self.X, self.y, cv=3, scoring='roc_auc')
        gc.collect()
        return error_list.mean()

    def preprocess_data(self):
        # application
        all_df = self.preprocess_aplication()
        # bureau
        bureau_df = self.preprocess_breau()
        all_df = pd.merge(all_df, bureau_df, how='left', on='SK_ID_CURR')
        # previous application
        prev_application_df = self.preprocess_prev_application()
        all_df = pd.merge(all_df, prev_application_df, how='left', on='SK_ID_CURR')
        # POS CASH
        pos_cash_df = self.preprocess_credit_card_df()
        all_df = pd.merge(all_df, pos_cash_df, how='left', on='SK_ID_CURR')
        # installments
        installments_df = self.preprocess_installments_df()
        all_df = pd.merge(all_df, installments_df, how='left', on='SK_ID_CURR')
        # credit card
        credit_card_df = self.preprocess_credit_card_df()
        all_df = pd.merge(all_df, credit_card_df, how='left', on='SK_ID_CURR')
        all_df = all_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        train_df = all_df[all_df['TARGET'].notnull()]
        test_df = all_df[all_df['TARGET'].isnull()]
        filled_train_df = self.fill_zero_num_cols(train_df)
        converted_train_df = self.convert_float64_to_float32(filled_train_df)
        self.y = converted_train_df['TARGET']
        features = [feature for feature in converted_train_df.columns if feature not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV']]
        self.X = converted_train_df[features]
        print(f'shape pf X: {self.X.shape}')
        self.clear_memory([all_df, train_df, test_df, filled_train_df, converted_train_df, bureau_df, prev_application_df, pos_cash_df, installments_df, credit_card_df])

    def get_objective(self, objective_name):
        objective_dict = {
            "multi_classifiers": self.objective_multi_classifiers,
            "LGBM": self.objective_LGBM
        }
        return objective_dict[objective_name]
        
    def parameter_tuning(self, study_name ,objective_name):
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=get_storage(), load_if_exists=True)  # Create a new study.
        study.optimize(self.get_objective(objective_name), n_trials=60)


home_credit = Home_Credit()
home_credit.preprocess_data()
home_credit.parameter_tuning('LGBM_00', 'LGBM')
