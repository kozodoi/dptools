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
    - pandas DF with missing values

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
            'height': [170, 168, 173, 177, 165], 
            'gender': ['female', 'male', np.nan, 'male', 'female'],
            'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # count missings
    from dptools import print_missings
    print_missings(df)
    '''

    # count missing values
    total   = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count()).sort_values(ascending = False)
    table   = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
    table   = table[table['Total'] > 0]

    # return results
    if len(table) > 0:
        print('Found {} features with missing values.'.format(len(table)))
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

def fill_missings(df, 
                  to_unknown_cols = [], 
                  to_0_cols       = [], 
                  to_mean_cols    = [],
                  to_true_cols    = [], 
                  to_false_cols   = []):
    '''
    Replaces NA in the dataset with specific values.
    
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - to_0_cols (list): list of features where NA => 0
    - to_mean_cols (list): list of features where NA => mean value
    - to_unknown_cols (list): list of features where NA => 'unknown'
    - to_true_cols (list): list of features where NA => True
    - to_false_cols (list): list of features where NA => False

    --------------------
    Returns
    - pandas DF with treated features
    '''

    # copy df
    df_new = df.copy()

    # fill missings
    if len(to_unknown_cols) > 0:
        df_new[to_unknown_cols] = df_new[to_unknown_cols].fillna('unknown')

    if len(to_0_cols) > 0:
        df_new[to_0_cols] = df_new[to_0_cols].fillna(0)

    if len(to_mean_cols) > 0:
        for var in to_mean_cols:
            df_new[var] = df_new[var].fillna(df_new[var].mean())

    if len(to_true_cols) > 0:
        df_new[to_true_cols] = df_new[to_true_cols].fillna(True)

    if len(to_false_cols) > 0:
        df_new[to_false_cols] = df_new[to_false_cols].fillna(False)
    
    # return results
    return df_new



###############################
#                             
#     SPLIT NESTED FEATURES
#                             
###############################

import pandas as pd

def split_nested_features(df, 
                          split_vars, 
                          sep,
                          drop = True):
    '''
    Splits a nested string column into multiple features using a specified 
    separator and appends the creates features to the data frame.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - split_vars (list): list of string features to be split
    - sep (str): separator to split features
    - drop (bool): whether to drop the original features after split

    --------------------
    Returns:
    - pandas DF with new features

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
            'height': [170, 168, 173, 177, 165], 
            'income': ['high,100', 'medium,50', 'low,25', 'low,28', 'no income,0']}
    df = pd.DataFrame(data)

    # split nested features
    from dptools import split_nested_features
    df_new = split_nested_features(df, split_vars = 'income', sep = ',')
    '''

    # copy df
    df_new = df.copy()

    # store no. features
    n_feats = df_new.shape[1]

    # convert to list
    if not isinstance(split_vars, list):
        split_vars = [split_vars]

    # feature engineering loop
    for split_var in split_vars:
        
        # count maximum values
        max_values = int(df_new[split_var].str.count(sep).max() + 1)
        new_vars = [split_var + '_' + str(val) for val in range(max_values)]
        
        # remove original feature
        if drop:
            cols_without_split = [col for col in df_new.columns if col not in split_var]
        else:
            cols_without_split = [col for col in df_new.columns]
            
        # split feature
        df_new = pd.concat([df_new[cols_without_split], df_new[split_var].str.split(sep, expand = True)], axis = 1)
        df_new.columns = cols_without_split + new_vars
        
    # return results
    print('Added {} split-based features.'.format(df_new.shape[1] - n_feats + int(drop) * len(split_vars)))
    return df_new



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

    # import dependencies
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



###############################
#                             
#      CORRECT COLUMN NAMES
#                             
###############################

import pandas as pd
import re

def correct_colnames(df):
    '''
    Corrects column names to avoid modeling errors:
    - drops foreign symbols
    - ensures that column names are unique
    
    --------------------
    Arguments:
    - df (pandas DF): dataset

    --------------------
    Returns
    - pandas DF with corrected column names

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
            'height': [170, 168, 173, 177, 165], 
            'height': ['female', 'male', np.nan, 'male', 'female'],
            'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)
    df.columns = ['age', 'height', 'height', 'income']

    # correct column names
    from dptools import correct_colnames
    df_new = correct_colnames(df)
    '''

    # drop foreign symbols
    df_new = df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # ensure that all colnames are unique
    def uniquify(df_columns):
        seen = set()
        for item in df_columns:
            fudge = 1
            newitem = item
            while newitem in seen:
                fudge += 1
                newitem = '{}_{}'.format(item, fudge)
            yield newitem
            seen.add(newitem)
    df_new.columns = uniquify(df_new)

    # return results
    return df_new