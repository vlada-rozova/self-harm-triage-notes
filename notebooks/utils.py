import os
import numpy as np
import pandas as pd
import xgboost

data_folder = 'data/processed'


def load_dataset():
    train = pd.read_parquet(f'{data_folder}/rmh_2012_2017_dev_normalised.parquet')
    test = pd.read_parquet(f'{data_folder}/rmh_2012_2017_test_normalised.parquet')
    prospective = pd.read_parquet(f'{data_folder}/rmh_2018_2022_normalised.parquet')
    external = pd.read_parquet(f'{data_folder}/lvrh_2012_2022_normalised.parquet')

    # prsp, external: year < 2022
    prospective = prospective[prospective['year'] < 2022].copy()
    external = external[external['year'] < 2022].copy()

    datasets = {
        "train": train,
        "test": test,
        "prsp": prospective,
        "external": external,
    }

    for dataset in datasets.keys():
        datasets[dataset]['binary_SH'] = get_binary_labels(datasets[dataset]['SH'])

    return datasets


def get_binary_labels(series):
    s = series.copy()

    # if already numeric (0/1)
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int).values

    # otherwise assume string labels
    return (s.astype(str).str.lower() == "positive").astype(int).values


def load_pred_features():
    with open('data/features/rmh_2012_2017_dev_SH_REFIT_ensemble_True_selected_fts_all.txt', 'r') as f:
        features = f.readlines()
    features = [x.strip() for x in features]
    return features

