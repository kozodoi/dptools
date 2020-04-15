###############################
#                             
#        COUNT MISSINGS       
#                             
###############################

import pandas as pd

def print_missings(df):
    '''
    Counts missing values in a dataframe and prints the results.

    --------------------
    Arguments:
    - df (pandas DF): dataset

    --------------------
    Returns:
    - None
    '''

    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    table = table[table['Total'] > 0]

    if len(table) > 0:
        return table
    else:
        print('No missing values found.')




###############################
#                             
#        FILL MISSINGS
#                             
###############################

import numpy as np
import pandas as pd

def fill_missings(df, to_na_cols, to_0_cols, to_true_cols, to_false_cols):
    '''
    Replaces NA in the dataset with specific values.
    
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - to_0_cols (list): list of features where NA => 0
    - to_na_cols (list): list of features where NA => 'unknown'
    - to_true_cols (list): list of features where NA => True
    - to_false_cols (list): list of features where NA => False

    --------------------
    Returns
    - pandas DF with treated features
    '''
    
    df[to_na_cols]    = df[to_na_cols].fillna('Unknown')
    df[to_0_cols]     = df[to_0_cols].fillna(0)
    df[to_true_cols]  = df[to_true_cols].fillna(True)
    df[to_false_cols] = df[to_false_cols].fillna(False)
       
    return df