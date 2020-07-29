import numpy as np
import pandas as pd
import re
import pytest

from dptools import print_missings
from dptools import fill_missings
from dptools import print_factor_levels
from dptools import split_nested_features
from dptools import correct_colnames

def test_split_nested_features_4():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high,100', 'medium,50', 'low,25', 'low,28', 'no income,0']}
    df = pd.DataFrame(data)
    df = split_nested_features(df, split_vars = 'income', sep = ',', drop = True)
    assert df.shape[1] == 4

def test_split_nested_features_5():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high 100', 'medium 50', 'low 25', 'low 28', 'no_income 0']}
    df = pd.DataFrame(data)
    df = split_nested_features(df, split_vars = 'income', sep = ' ', drop = False)
    assert df.shape[1] == 5

def test_fill_missings_0():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high', 'medium', 'low', 'low', 'no_income']}
    df = pd.DataFrame(data)
    df = fill_missings(df, to_0_cols = 'age')
    assert df['age'][4] == 0

def test_fill_missings_unknown():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high', np.nan, 'low', 'low', 'no_income']}
    df = pd.DataFrame(data)
    df = fill_missings(df, to_0_cols = 'age', to_unknown_cols = 'income')
    assert df['income'][1] == 'unknown'

def correct_colnames():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'height': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df.columns = ['age', 'height', 'height', 'incöme']
    df = correct_colnames(df)
    assert all(df.columns == ['age', 'height', 'height_2', 'incme'])