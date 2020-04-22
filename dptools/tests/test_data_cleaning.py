import numpy as np
import pandas as pd
import pytest

def test_find_constant_features():

    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)
    
    assert find_constant_features(df) == ['gender']
