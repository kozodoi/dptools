import numpy as np
import pandas as pd
import pytest

from dptools import add_date_features
from dptools import add_text_features
from dptools import aggregate_data

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

def test_add_text_features_inplace():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    add_text_features(df, 
                      text_vars = ['income', 'gender'], 
                      tf_idf_feats = 2,
                      drop = False,
                      inplace = True)
    assert df.shape[1] == 11