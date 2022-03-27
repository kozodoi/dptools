###############################
#                             
#      ADD DATE FEATURES      
#                             
###############################

import numpy as np
import pandas as pd
import re

def add_date_features(df, 
                      date_vars, 
                      drop = True, 
                      time = False):
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
    
    --------------------
    Examples:
    
    # create data frame
    data = {'age': [27, np.nan, 30], 
            'height': [170, 168, 173], 
            'gender': ['female', 'male', np.nan],
            'date_of_birth': [np.datetime64('1993-02-10'), np.nan, np.datetime64('1990-04-08')]}
    df = pd.DataFrame(data)

    # add date features
    from dptools import add_date_features
    df_new = add_date_features(df, date_vars = 'date_of_birth')
    '''
    
    # copy df
    df_new = df.copy()
    
    # store no. features
    n_feats = df_new.shape[1]

    # convert to list
    if not isinstance(date_vars, list):
        date_vars = [date_vars]

    # feature engineering loop
    for date_var in date_vars:

        var = df_new[date_var]
        var_dtype = var.dtype
        
        if isinstance(var_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            var_dtype = np.datetime64

        if not np.issubdtype(var_dtype, np.datetime64):
            df_new[date_var] = var = pd.to_datetime(var, infer_datetime_format = True)
            
        targ_pre = re.sub('[Dd]ate$', '', date_var)

        # list of day attributes
        attributes = ['year', 'month', 'week', 'day', 
                      'dayofweek', 'dayofyear',
                      'is_month_end', 'is_month_start', 
                      'is_quarter_end', 'is_quarter_start', 
                      'is_year_end', 'is_year_start']
        
        # list of time attributes
        if time: 
            attributes = attributes + ['Hour', 'Minute', 'Second']
            
        # compute features
        for att in attributes: 
            df_new[targ_pre + '_' + att.lower()] = getattr(var.dt, att)

        df_new[targ_pre + '_elapsed'] = var.astype(np.int64) // 10 ** 9
        
        # remove original feature
        if drop: 
            df_new.drop(date_var, axis = 1, inplace = True)

    # return results
    print('Added {} date-based features.'.format(df_new.shape[1] - n_feats + int(drop) * len(date_vars)))
    return df_new



###############################
#                             
#      ADD TEXT FEATURES      
#                             
###############################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse

def add_text_features(df, 
                      text_vars, 
                      tf_idf_feats = 5, 
                      common_words = 0,
                      rare_words   = 0,
                      ngram_range  = (1, 1),
                      drop         = True):
    '''
    Adds basic text-based features including word count, character count and 
    TF-IDF based features to the data frame.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - text_vars (list): list of textual features
    - tf_idf_feats (int): number of TF-IDF based features
    - common_words (int): number of the most common words to remove for TF-IDF
    - rare_words (int): number of the most rare words to remove for TF-IDF
    - ngram_range (int, int): range of n-grams for TF-IDF based features
    - drop (bool): whether to drop the original textual features

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
        'gender': ['female', 'male', np.nan, 'male', 'female'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # add text features
    from dptools import add_text_features
    df_new = add_text_features(df, text_vars = ['income', 'gender'])
    '''
    # copy df
    df_new = df.copy()

    # store no. features
    n_feats = df_new.shape[1]

    # convert to list
    if not isinstance(text_vars, list):
        text_vars = [text_vars]

    # feature engineering loop
    for text_var in text_vars:

        # replace NA with empty string
        df_new[text_var].fillna('', inplace = True)

        # remove common and rare words
        freq = pd.Series(' '.join(df_new[text_var]).split()).value_counts()[:common_words]
        freq = pd.Series(' '.join(df_new[text_var]).split()).value_counts()[-rare_words:]

        # convert to lowercase 
        df_new[text_var] = df_new[text_var].apply(lambda x: ' '.join(x.lower() for x in x.split())) 

        # remove punctuation
        df_new[text_var] = df_new[text_var].str.replace('[^\w\s]','')         

        # word count
        df_new[text_var + '_word_count'] = df_new[text_var].apply(lambda x: len(str(x).split(' ')))
        df_new[text_var + '_word_count'][df_new[text_var] == ''] = 0

        # character count
        df_new[text_var + '_char_count'] = df_new[text_var].str.len().fillna(0).astype('int64')

        # import vectorizer
        tfidf  = TfidfVectorizer(max_features = tf_idf_feats, 
                                 lowercase    = True, 
                                 norm         = 'l2', 
                                 analyzer     = 'word', 
                                 stop_words   = 'english', 
                                 ngram_range  = ngram_range)

        # compute TF-IDF
        vals = tfidf.fit_transform(df_new[text_var])
        vals = pd.DataFrame.sparse.from_spmatrix(vals)
        vals.columns = [text_var + '_tfidf_' + str(p) for p in vals.columns]
        df_new = pd.concat([df_new, vals], axis = 1)

        # remove original feature
        if drop:
            df_new.drop(text_var, axis = 1, inplace = True)
        
    # return results
    print('Added {} text-based features.'.format(df_new.shape[1] - n_feats + int(drop) * len(text_vars)))
    return df_new



###############################
#                             
#       AGGREGATE DATA       
#                             
###############################

import pandas as pd

def aggregate_data(df, 
                   group_var, 
                   num_stats = ['mean', 'sum'], 
                   fac_stats = ['count', 'mode'],
                   factors   = None, 
                   var_label = None, 
                   sd_zeros  = False):
    '''
    Aggregates the data by a certain categorical feature. Continuous features 
    are aggregated by computing summary statistics by the grouping feature. 
    Categorical features are aggregated by computing the most frequent values
    and number of unique value by the grouping feature.

    --------------------
    Arguments:
    - df (pandas DF): dataset
    - group_var (str): grouping feature
    - num_stats (list): list of stats for aggregating numeric features
    - fac_stats (list): list of stats for aggregating categorical features
    - factors (list): list of categorical features names
    - var_label (str): prefix for feature names after aggregation
    - sd_zeros (bool): whether to replace NA with 0 for standard deviation

    --------------------
    Returns
    - aggregated pandas DF

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

    # aggregate the data
    from dptools import aggregate_data
    df_new = aggregate_data(df, group_var = 'gender', num_stats = ['min', 'max'], fac_stats = 'mode')   

    '''
    
    ##### SEPARATE FEATURES

    # display info
    print('- Preparing the dataset...')

    # find factors
    if factors == None:
        df_factors = [f for f in df.columns if df[f].dtype == 'object']
        factors    = [f for f in df_factors if f != group_var]
    else:
        if not isinstance(factors, list):
            factors = [factors]
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
    n_facs = fac_df.shape[1] - 1
    n_nums = num_df.shape[1] - 1
    print('- Extracted %.0f factors and %.0f numerics...' % (n_facs, n_nums))
    

    ##### AGGREGATION

    # aggregate numerics
    if n_nums > 0:
        print('- Aggregating numeric features...')
        num_df = num_df.groupby([group_var]).agg(num_stats)
        num_df.columns = ['_'.join(col).strip() for col in num_df.columns.values]
        num_df = num_df.sort_index()

    # aggregate factors
    if n_facs > 0:
        print('- Aggregating factor features...')
        if (fac_stats == ['count', 'mode']) or (fac_stats == ['mode', 'count']):
            fac_df = fac_df.groupby([group_var]).agg([('count'), ('mode', lambda x: pd.Series.mode(x)[0])])
        if (fac_stats == 'count') or (fac_stats == ['count']):
            fac_df = fac_df.groupby([group_var]).agg([('count')])
        if (fac_stats == 'mode') or (fac_stats == ['mode']):
            fac_df = fac_df.groupby([group_var]).agg([('mode', lambda x: pd.Series.mode(x)[0])])
        fac_df.columns = ['_'.join(col).strip() for col in fac_df.columns.values]
        fac_df = fac_df.sort_index()           


    ##### MERGER

    # merge numerics and factors
    if ((n_facs > 0) & (n_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((n_facs > 0) & (n_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((n_facs == 0) & (n_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (var_label != None):
        agg_df.columns = [var_label + '_' + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if sd_zeros:
        stdevs = agg_df.filter(like = '_std').columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print('- Final dimensions:', agg_df.shape)
    return agg_df



###############################
#                             
#        ENCODE FACTORS       
#                             
###############################

import pandas as pd

def encode_factors(df, factors = None, method = 'label'):
    '''
    Performs encoding of categorical features using label or dummy encoding.

    --------------------
    Arguments:
    - df (pandas DF): pandas DF
    - factors (str): list of factors; all object features are treated as factors by default
    - method (str): encoding method ('label' or 'dummy')

    --------------------
    Returns:
    - pandas DF with encoded factors

    --------------------
    Examples:

    # import dependencies
    import pandas as pd
    import numpy as np

    # create data frame
    data = {'age': [27, np.nan, 30, 25, np.nan], 
        'gender': ['female', 'female', 'male', 'female', 'male'],
        'income': ['high', 'medium', 'low', 'low', 'no income']}
    df = pd.DataFrame(data)

    # encode factors
    from dptools import encode_factors
    df_enc = encode_factors(df, method = 'label')
    '''

    # copy df
    df_new = df.copy()

    # list factors
    if factors is None:
        factors = [f for f in df_new.columns if df_new[f].dtype == 'object']

    # convert to list
    if not isinstance(factors, list):
        factors = [factors]
    
    # label encoding
    if method == 'label':
        for var in factors:
            df_new[var], _ = pd.factorize(df_new[var])
        
    # dummy encoding
    if method == 'dummy':
        df_new = pd.get_dummies(df_new, columns = factors, drop_first = False)

    # return data
    return df_new