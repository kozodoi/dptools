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