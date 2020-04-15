###############################
#                             
#      ADD DATE FEATURES      
#                             
###############################

import numpy as np
import pandas as pd

def add_date_features(df, date_var, drop = True, time = False):
    '''
    Adds basic date-based features based to the data frame.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - date_var (str): name of the date feature
    - drop (bool): whether to drop the original date feature
    - time (bool): whether to include time-based features

    --------------------
    Returns:
    - pandas DF with new features
    '''
    
    fld = df[date_var]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[date_var] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre = re.sub('[Dd]ate$', '', date_var)

    attr = ['Year', 'Month', 'Week', 'Day', 
            'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 
            'Is_year_end', 'Is_year_start']
    
    if time: 
        attr = attr + ['Hour', 'Minute', 'Second']
        
    for n in attr: 
        df[targ_pre + n] = getattr(fld.dt, n.lower())

    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    
    if drop: 
        df.drop(date_var, axis = 1, inplace = True)

    return df