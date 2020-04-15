###############################
#                             
#        COUNT MISSINGS       
#                             
###############################

import pandas as pd

def count_missings(df):
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