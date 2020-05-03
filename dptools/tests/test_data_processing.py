import numpy as np
import pandas as pd
import pytest

from dptools import print_missings
from dptools import fill_missings
from dptools import print_factor_levels
from dptools import split_nested_features

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

def test_split_nested_features_inplace():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'income': ['high;100', 'medium;50', 'low;25', 'low;28', 'no income;0']}
    df = pd.DataFrame(data)
    split_nested_features(df, split_vars = 'income', sep = ';', drop = False, inplace = True)
    assert df.shape[1] == 5