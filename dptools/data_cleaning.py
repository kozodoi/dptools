###############################
#                             
#     FIND CONSTANT FEATURES
#                             
###############################

import pandas as pd

def find_constant_features(df, dropna = False):
    '''
    Finds features that have just a single unique value.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - dropna (bool): whether to treat NA as a unique value

    --------------------
    Returns:
    - list of constant features

    --------------------
    Examples:

    # import dependecies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female, 'female', 'female']}
    df = pd.DataFrame(data)

    # check constant features
    from dptools import find_constant_features
    find_constant_features(df)
    '''
    
    # find constant features
    constant = df.nunique(dropna = dropna) == 1
    features = list(df.columns[constant])

    # return results
    if len(features) > 0:
        print('Found {} constant features.'.format(len(features)))
        return features 
    else:
        print('No constant features found.')