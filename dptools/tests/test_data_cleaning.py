import numpy as np
import pandas as pd
import pytest

from dptools import find_constant_features
from dptools import find_correlated_features

def test_find_constant_features_1():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)
    assert find_constant_features(df) == ['gender'] 
  
def test_find_constant_features_0():
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['male', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)
    assert find_constant_features(df) == None

def find_correlated_features_1():
    data = {'age': [30, 25, 30, 35, 18], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)
    assert find_correlated_features(df, cutoff = 0.8, method = 'spearman') == ['height']

def find_correlated_features_0():
    data = {'age': [30, 25, 30, 35, 18], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)
    assert find_correlated_features(df, cutoff = 0.9, method = 'pearson') == None