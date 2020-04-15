###############################
#                             
#        AGGRGEATE DATA       
#                             
###############################

import pandas as pd

def aggregate_data(df, 
                   group_var, 
                   num_stats = ['mean', 'sum'], 
                   factors = None, 
                   var_label = None, 
                   sd_zeros = False):
    '''
    Aggregates the data by a certain categorical feature. Continuous features 
    are aggregated by computing summary statistcs by the grouping feature. 
    Categorical features are aggregated by computing the most frequent values
    and number of unique value by the grouping feature.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - group_var (str): grouping feature
    - num_stats (list): list of stats for aggregating numric features
    - factors (list): list of categorical features names
    - var_label (str): prefix for feature names after aggregation
    - sd_zeros (bool): whether to replace NA with 0 for standard deviation

    --------------------
    Returns
    - aggregated pandas DF

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

    # aggregate the data
    from dptools import aggregate_data
    df_new = aggregate_data(df, group_var = 'gender', num_stats = ['min', 'max])
    '''
    
    ##### SEPARATE FEATURES
  
    # display info
    print('- Preparing the dataset...')

    # find factors
    if factors == None:
        df_factors = [f for f in df.columns if df[f].dtype == 'object']
        factors    = [f for f in df_factors if f != group_var]
    else:
        df_factors = factors
        df_factors.append(group_var)
        
    # partition subsets
    if type(group_var) == str:
        num_df = df[[group_var] + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]
    else:
        num_df = df[group_var + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors] 
    
    # display info
    num_facs = fac_df.shape[1] - 1
    num_nums = num_df.shape[1] - 1
    print('- Extracted %.0f factors and %.0f numerics...' % (num_facs, num_nums))
    

    ##### AGGREGATION
 
    # aggregate numerics
    if (num_nums > 0):
        print('- Aggregating numeric features...')
        num_df = num_df.groupby([group_var]).agg(num_stats)
        num_df.columns = ['_'.join(col).strip() for col in num_df.columns.values]
        num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print('- Aggregating factor features...')
        fac_int_df = fac_df.copy()
        for var in factors:
            fac_int_df[var], _ = pd.factorize(fac_int_df[var])
        fac_int_df[group_var] = fac_df[group_var]
        fac_df = fac_int_df.groupby([group_var]).agg([('count'), ('mode', lambda x: pd.Series.mode(x)[0])])
        fac_df.columns = ['_'.join(col).strip() for col in fac_df.columns.values]
        fac_df = fac_df.sort_index()           


    ##### MERGER

    # merge numerics and factors
    if ((num_facs > 0) & (num_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((num_facs > 0) & (num_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((num_facs == 0) & (num_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (var_label != None):
        agg_df.columns = [var_label + '_' + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = '_std').columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print('- Final dimensions:', agg_df.shape)
    return agg_df