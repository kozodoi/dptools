###############################
#                             
#        ENCODE FACTORS       
#                             
###############################

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def label_encoding(df_train, df_valid, df_test):
    '''
    Performs label encoding of categorical features for the data set partitioned
    into three samples: training, validation and test. The values that do not
    appear in the training sample are set to missings.

    --------------------
    Arguments:
    - df_train (pandas DF): training sample
    - df_valid (pandas DF): validation sample
    - df_test (pandas DF): test sample

    --------------------
    Returns:
    - encoded pandas DF with the training sample
    - encoded pandas DF with the validation sample
    - encoded pandas DF with the test sample
    '''
    
    # list of factors
    factors = [f for f in df_train.columns if df_train[f].dtype == 'object' or df_valid[f].dtype == 'object' or df_test[f].dtype == 'object']
    
    lbl = LabelEncoder()

    # label encoding
    for f in factors:        
        lbl.fit(list(df_train[f].values) + list(df_valid[f].values) + list(df_test[f].values))
        df_train[f] = lbl.transform(list(df_train[f].values))
        df_valid[f] = lbl.transform(list(df_valid[f].values))
        df_test[f]  = lbl.transform(list(df_test[f].values))

    return df_train, df_valid, df_test




###############################
#                             
#      PRINT FACTOR LEVELS
#                             
###############################

import pandas as pd

def print_factor_levels(df, top = 5):
    '''
    Prints levels of categorical features in the dataset.
    
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - top (int): how many most frequent values to display

    --------------------
    Returns
    - None

    --------------------
    Examples:

    # import dependecies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # print factor levels
    from dptools import print_factors
    print_factors(df, top = 3)
    '''

    # find factors
    facs = [f for f in df.columns if df[f].dtype == 'object']
    
    # print results
    if len(facs) > 0:
        print('Found {} categorical features.'.format(len(facs)))
        print('')
        for fac in facs:
            print('-' * 30)
            print(fac + ': ' + str(df[fac].nunique()) + ' unique values')
            print('-' * 30)
            print(df[fac].value_counts(normalize = True, dropna = False).head(top))
            print('-' * 30)
            print('')
    else:
        print('Found no categorical features.')