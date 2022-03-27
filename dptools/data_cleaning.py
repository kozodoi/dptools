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

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
            'height': [170, 168, 173, 177, 165], 
            'gender': ['female', 'female', 'female', 'female', 'female']}
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



###############################
#                             
#    FIND CORRELATED FEATURES
#                             
###############################

import pandas as pd
import numpy as np

def find_correlated_features(df, cutoff = 0.9, method = 'pearson'):
    '''
    Finds features that have a pairwise Pearson or Spearman correlation exceeding a specified threshold. For each pair of features, only one feature is returned.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - cutoff (float): correlation threshold
    - method (string): correlation type: 'pearson', 'spearman' or both

    --------------------
    Returns:
    - list of correlated features

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [30, 25, 30, 35, 18], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'female', 'female', 'female', 'female']}
    df = pd.DataFrame(data)

    # check correlated features
    from dptools import find_correlated_features
    find_correlated_features(df, cutoff = 0.8, method = 'spearman')
    '''
    # extract numeric features
    numerics = [col for col in df.columns if df[col].dtype != 'object']
    df = df[numerics]

    # compute correlations
    if method == 'pearson':
        corr_matrix = df.corr().abs()
    if method == 'spearman':
        corr_matrix = df.rank().corr().abs()
    if method == 'both':
        corr_matrix_s = df.corr().abs()
        corr_matrix_p = df.rank().corr().abs()
        corr_matrix   = np.maximum(corr_matrix_p, corr_matrix_s)

    # transform to lower triangle
    corr_lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k = -1).astype(np.bool))

    # remove features
    features = [col for col in corr_lower.columns if any(corr_lower[col] > cutoff)]

    # return results
    if len(features) > 0:
        print('Found {} correlated features.'.format(len(features)))
        return features 
    else:
        print('No correlated features found.')