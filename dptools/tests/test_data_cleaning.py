import numpy as np
import pandas as pd
import pytest

import dptools
from dptools import find_constant_features

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
    assert find_constant_features(df) == []
