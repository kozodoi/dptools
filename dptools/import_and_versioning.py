###############################
#                             
#       SAVE CSV VERSION
#                             
###############################

from os import path
import pandas as pd

def save_csv_version(file_path, df, min_version = 1, **args):
    '''
    Saves pandas DF as a csv file with an automatically assigned version number 
    to prevent overwriting the existing file. If no file with the same name 
    exists in the specified path, '_v1' is appended to the file name to indicate 
    the version of the saved data. If such a version already exists, the function 
    iterates over integers and saves the data as '_v[k]', where [k] stands for 
    the next available integer. 

    --------------------
    Arguments:
    - file_path (str): file path including the file name
    - df (pandas DF): dataset
    - min_version (int): minimum version number
    - **args: further arguments to pass to pd.to_csv() function

    --------------------
    Returns:
    - None

    --------------------
    Examples:
    
    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'height': [170, 168, 173, 177, 165], 
        'gender': ['female', 'male', np.nan, 'male', 'female']}
    df = pd.DataFrame(data)

    # first call saves df as 'data_v1.csv'
    from dptools import save_csv_version
    save_csv_version('data.csv', df, index = False)

    # second call saves df as 'data_v2.csv' as data_v1.csv already exists
    save_csv_version('data.csv', df, index = False)
    '''

    # initialize
    version = min_version - 1
    is_version_present = True

    # update name
    file_path_version = file_path.replace('.csv', ('_v' + str(version) + '.csv'))
    
    # export loop
    while is_version_present:

        # update file name
        version += 1
        file_path_version = file_path.replace('.csv', ('_v' + str(version) + '.csv'))

        # check for a file with the same name
        is_version_present = path.isfile(file_path_version)

    # save file
    df.to_csv(file_path_version, **args)
    print('Saved as ' + file_path_version)



###############################
#                             
#       READ CSV WITH JSON
#                             
###############################

from pandas.io.json import json_normalize
import json
import os

def read_csv_with_json(file_path, json_cols, **args):
    '''
    Imports csv where some columns are JSON-encoded as pandas DF. 

    --------------------
    Arguments:
    - file_path (str): file path including the file name
    - json_cols (list): list of JSON-encoded columns
    - **args: further arguments to pass to pd.read_csv() function

    --------------------
    Returns:
    - imported pandas DF
    '''
        
    # import data frame
    df = pd.read_csv(file_path, 
                     converters = {column: json.loads for column in json_cols}, 
                     **args)
    
    # convert to list
    if not isinstance(json_cols, list):
        json_cols = [json_cols]
    
    # extract values
    for column in json_cols:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f'{column}_{subcolumn}' for subcolumn in column_as_df.columns]
        df = df.drop(column, axis = 1).merge(column_as_df, right_index = True, left_index = True)

    # return data
    print(f'Loaded {os.path.basename(path)}: {df.shape}')
    return df