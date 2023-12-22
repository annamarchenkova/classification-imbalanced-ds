import logging
import os
import pickle
import joblib
from datetime import datetime as dt
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

from catboost import CatBoostClassifier, EFeaturesSelectionAlgorithm, EShapCalcType, Pool
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

def load_config(cnf_dir=PROJECT_DIR, cnf_name='config.yml'):
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)

##########################################################################################
# ---------------------------------  DS PREPROCESSING  ---------------------------------- #

def get_cols_too_similar(data, threshold=0.95):
    """
    Find features with too many similar values.
    :return: the pandas dataframe of sought features with the fraction of values which are similar, 
             as well as a list containing the most present value.
    
    :data: (pd.DataFrame) dataset
    :threshold: (float, default=0.95) fraction of similar values, must be a number in [0,1] interval
    """
    
    L = len(data)
    
    cols_counts = list()

    for col in data.columns:
        try:
            unique_values, unique_counts = np.unique(data[col].values, return_counts=True)
        except TypeError:
            unique_values, unique_counts = np.unique(data[col].astype(str).values, return_counts=True)

        idx_max = np.argmax(unique_counts)
        cols_counts.append((col, unique_values[idx_max], unique_counts[idx_max]))
    
    colname_and_values = map(lambda x: (x[0], x[2]), cols_counts)
    most_present_value = map(lambda x: x[1], cols_counts)

    df_similar_values = pd.DataFrame(colname_and_values)\
        .rename(columns={0: 'col_name', 1: 'frac'})\
        .sort_values('frac', ascending=False)

    df_similar_values['frac'] = df_similar_values['frac'].apply(lambda x: x / L)
    df_similar_values.query('frac >= @threshold', inplace=True)
    
    return df_similar_values, list(most_present_value)


def fill_nan_categorical_w_mode(df, fill_with='Not Available'):
    # Fill NaNs in categorical columns with mode
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values

    for column in nan_cols_cat:
        df[column] = df[column].fillna(fill_with)
        
    return df


def fill_nan_numerical_w_median(df, fill_with='median'):
    # Fill NaNs in categorical columns with non disponibile'
    nan_cols_cat = df.isna().sum()[(df.isna().sum() > 0) & (df.dtypes == 'object')].index.values

    for column in nan_cols_cat:
        df[column] = df[column].fillna(fill_with)
        
    return df

def get_non_collinear_features_from_vif(data, vif_threshold=5, idx=0):
    """
    Find features whose variance inflation factor (VIF) exceeds the desired threshold and eliminate them.
    :return: list of feature names without the features whose VIF exceeds the threshold.
    
    :data: (pd.DataFrame) dataset
    :vif_threshold: (int, default=5) VIF threshold
    :idx: (int, default=0) DO NOT TOUCH
    """

    num_features = [i[0] for i in data.dtypes.items() if i[1] in ['float64', 'float32', 'int64', 'int32']]
    df = data[num_features].copy()
    
    if idx >= len(num_features):
        return df.columns.to_list()

    else:
        print('\rProcessing feature {}/{}'.format(idx+1, len(num_features)), end='')
        vif_ = variance_inflation_factor(df, idx)

        if vif_ > vif_threshold:
            df.drop(num_features[idx], axis=1, inplace=True)
            return get_non_collinear_features_from_vif(df, idx=idx, vif_threshold=vif_threshold)

        else:
            return get_non_collinear_features_from_vif(df, idx=idx+1, vif_threshold=vif_threshold)

def find_cols_w_2many_nan(
        data: pd.DataFrame,
        *,
        thr:float=0.95, 
        f_display:bool=False) -> Tuple[List[str], Optional[pd.DataFrame]]:
    
    na_cols = data.columns[data.isna().any()]

    df_nans = data[na_cols].copy() \
                .isna().sum() \
                .apply(lambda x: x / data.shape[0]) \
                .reset_index().rename(columns={0: 'f_nans', 'index': 'feature_name'}) \
                .sort_values(by='f_nans', ascending=False)
    
    cols_2_many_nans = df_nans.loc[df_nans.f_nans >= thr, 'feature_name'].to_list()

    if f_display:
        disp_df = df_nans.style.background_gradient(axis=0, gmap=df_nans['f_nans'], cmap='Oranges')
        return cols_2_many_nans, disp_df
    else:
        return cols_2_many_nans
    
    
def find_cols_w_single_value(data: pd.DataFrame) -> List[str]:
    return list(
        col for col, n_unique in data.nunique().items() if n_unique==1
    )

def fill_nan_categorical_w_mode_V2(data: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs in categorical columns with mode'
    df = data.copy()
    
    na_cols = df.columns[df.isna().any()]
    
    for col in na_cols:
        if df[col].dtypes == 'O':
            col_mode = df[col].mode()[0]
            df[col].fillna(col_mode, inplace=True)
    
    return df


def fill_nan_numerical_w_median_V2(data: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs in categorical columns with non disponibile'
    df = data.copy()
    
    na_cols = df.columns[df.isna().any()]
    
    for col in na_cols:
        if df[col].dtypes != 'O':
            col_median = df[col].median()
            df[col].fillna(col_median, inplace=True)
    
    return df


def drop_unnecessary_cols(df, cnf):

    df.drop(cnf['columns_to_drop'], axis=1, inplace=True, errors='ignore')

    nan_list = find_cols_w_2many_nan(df, thr=cnf['nan_value_threshold'])
    logging.info(f"Dropped cols with NaN % > {cnf['nan_value_threshold']*100}: {[i for i in nan_list]}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(nan_list, axis=1, inplace=True)
    
    single_value_list = find_cols_w_single_value(df)
    logging.info(f"Dropped single value cols: {[i for i in single_value_list]}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(single_value_list, axis=1, inplace=True)
    
    df_sim_vals, most_present_value = get_cols_too_similar(df, cnf['value_similarity_threshold'])
    cols_2similar = df_sim_vals.col_name
    logging.info(f"Dropped too similar cols: {[i for i in cols_2similar]}; thr: {cnf['value_similarity_threshold']}")
    # print([i for i in df.columns if i in ['ACCORDATO_VL_CQS_ATTUALE', 'NUM Rate mancanti MUTUO']])
    df.drop(cols_2similar, axis=1, inplace=True)

    return df

##########################################################################################
# --------------------------------  FEATURE SELECTION  --------------------------------- #


def select_features_with_shap(
        n,
        train_pool, test_pool,
        algorithm,
        iterations=250,
        feat_for_select=None,
        flg_final_model=True, 
        steps=3, 
        **kwargs):
    """
    Perform recursive feature selection with CatBoost using SHAP values. 
    :return: the summary of the search as a Dict, and possibly a model trained on the dataset with the selected
                features only.
    
    :n: (int) number of features to select
    :train_pool: (catboost.Pool) train data as catboost Pool
    :test_pool: (catboost.Pool) validation data as catboost Pool
    :algorithm: (catboost.EFeaturesSelectionAlgorithm) which algorithm to use for recursive feature selection
    :iterations: (int, default=250) number of iterations to perform
    :feat_for_select: (list, default=None) index of features to be considered in feature selection,
    :flg_final_model: (bool, default=True) whether to fit the final model (i.e. with only selected feautures) or not
        --> note that if `True` the final model is returned as well
    :steps: (int, default=3) number of steps
    :kwargs: additional arguments for the `.select_features()` method
    """
    
    if feat_for_select is None:
        feat_for_select = list()
    
    print('Algorithm:', algorithm)
    
    model = CatBoostRegressor(
        iterations=iterations,
        loss_function='Logloss',
        random_seed=42, 
    )
    
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=feat_for_select,
        num_features_to_select=n,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=flg_final_model,
        logging_level='Silent',
        plot=False,
        **kwargs
    )
    
    res = (summary, model) if flg_final_model else summary

    return res


def f_selection_catboost_shap(
        df, y,
        test_size=0.2,
        shuffle=True,
        n_features=30,
        steps=5,
        iterations=500,
        random_state=42):
    
    features = df.columns.to_list()
    cat_features = [f for f, dtype in df.dtypes.items() if dtype=='O']
    
    X_train, X_test, y_train, y_test = train_test_split(
        df, y,
        test_size=test_size,
        stratify=y,
        shuffle=shuffle, random_state=random_state
    )

    train_pool = Pool(
        X_train, y_train,
        feature_names=features,
        cat_features=cat_features
    )

    test_pool = Pool(
        X_test, y_test,
        feature_names=features,
        cat_features=cat_features
    )
    
    feat_idx_list = np.arange(train_pool.num_col())

    summary, model = select_features_with_shap(
        n_features,
        train_pool, test_pool,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        steps=steps, iterations=iterations,
        feat_for_select=feat_idx_list
    )

    return summary, model

def catboost_gridsearch(df, params):

    X = df.drop(columns=params['cols_to_exclude'], errors='ignore').copy()
    y = df[params['target']].copy()

    cat_features = np.where(X.dtypes=='object')[0]

    # CatBoostClassifier MODEL
    regr = CatBoostClassifier(params['early_stopping_rounds'],
                                cat_features=cat_features)

    X1_train, X1_test, y1_train, y1_test, _, _ = \
        train_test_split(X, y, df.index,
                         stratify=y,
                         test_size=params['test_size'],
                         random_state=params['random_state'])

    # Cross validation
    skf = StratifiedKFold(n_splits=params['cv_folds'],
                          random_state=params['random_state'],
                          shuffle=True)
    logging.info('Retraining the model')
    clf_grid = GridSearchCV(estimator=regr,
                            param_grid=params['param_grid'],
                            scoring=params['scorer'],
                            cv=skf,
                            verbose=1)
                            # n_jobs=CROSS_VALIDATION_FOLDS)

    clf_grid.fit(X1_train, y1_train,
                eval_set=(X1_test, y1_test),
                cat_features=cat_features,
                plot=False,
                verbose=False)




















