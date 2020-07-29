import numpy as np
import pandas as pd
import pytest

from dptools import add_date_features
from dptools import add_text_features
from dptools import aggregate_data
from dptools import encode_factors

def test_aggregate_data_6():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = aggregate_data(df, 
                        group_var = 'gender', 
                        num_stats = ['min', 'max'], 
                        fac_stats = 'mode')   
    assert df.shape[1] == 6

def test_aggregate_data_8():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = aggregate_data(df, 
                        group_var = 'gender', 
                        num_stats = ['min', 'mean', 'max'], 
                        fac_stats = ['mode', 'count'])   
    assert df.shape[1] == 9

def test_add_text_features_9():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = add_text_features(df, 
                           text_vars = ['income'], 
                           tf_idf_feats = 5,
                           drop = True)
    assert df.shape[1] == 9

def test_add_text_features_10():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = add_text_features(df, 
                           text_vars = ['income', 'gender'], 
                           tf_idf_feats = 2,
                           drop = True)
    assert df.shape[1] == 10

def test_add_date_features_15():
    data = {'age': [27, np.nan, 30], 
        'height': [170, 168, 173], 
        'gender': ['female', 'male', np.nan],
        'date_of_birth': [np.datetime64('1993-02-10'), np.nan, np.datetime64('1990-04-08')]}
    df = pd.DataFrame(data)
    df = add_date_features(df, date_vars = 'date_of_birth', time = False)
    assert df.shape[1] == 15

def test_encode_factors_label():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = encode_factors(df, method = 'label')
    assert df.shape[1] == 4

def test_encode_factors_dummy_8():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = encode_factors(df, method = 'dummy')
    assert df.shape[1] == 8

def test_encode_factors_dummy_7():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df = encode_factors(df, factors = 'income', method = 'dummy')
    assert df.shape[1] == 7