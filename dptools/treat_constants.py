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