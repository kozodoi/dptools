###############################
#                             
#      ADD DATE FEATURES      
#                             
###############################

import numpy as np
import pandas as pd
import re

def add_date_features(df, date_vars, drop = True, time = False):
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
    
    # store no. features
    n_feats = df.shape[1]

    # convert to list
    if not isinstance(date_vars, list):
        date_vars = [date_vars]

    # feature engineering loop
    for date_var in date_vars:

        var = df[date_var]
        var_dtype = var.dtype
        
        if isinstance(var_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            var_dtype = np.datetime64

        if not np.issubdtype(var_dtype, np.datetime64):
            df[date_var] = var = pd.to_datetime(var, infer_datetime_format = True)
            
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
            df[targ_pre + '_' + att.lower()] = getattr(var.dt, att)

        df[targ_pre + '_elapsed'] = var.astype(np.int64) // 10 ** 9
        
        # remove original feature
        if drop: 
            df.drop(date_var, axis = 1, inplace = True)

    # return results
    print('Added {} date-based features.'.format(df.shape[1] - n_feats))
    return df



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
                      rare_words = 0,
                      drop = True):
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
    - drop (bool): whether to drop the original textual features

    --------------------
    Returns:
    - pandas DF with new features

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

    # add text features
    from dptools import add_text_features
    df_new = add_text_features(df, text_vars = ['income', 'gender'])
    '''

    # store no. features
    n_feats = df.shape[1]

    # convert to list
    if not isinstance(text_vars, list):
        text_vars = [text_vars]

    # feature engineering loop
    for text_var in text_vars:

        # replace NA with empty string
        df[text_var].fillna('', inplace = True)

        # remove common and rare words
        freq = pd.Series(' '.join(df[text_var]).split()).value_counts()[:common_words]
        freq = pd.Series(' '.join(df[text_var]).split()).value_counts()[-rare_words:]

        # convert to lowercase 
        df[text_var] = df[text_var].apply(lambda x: ' '.join(x.lower() for x in x.split())) 

        # remove punctuation
        df[text_var] = df[text_var].str.replace('[^\w\s]','')         

        # word count
        df[text_var + '_word_count'] = df[text_var].apply(lambda x: len(str(x).split(' ')))
        df[text_var + '_word_count'][df[text_var] == ''] = 0

        # character count
        df[text_var + '_char_count'] = df[text_var].str.len().fillna(0).astype('int64')

        # import vectorizer
        tfidf  = TfidfVectorizer(max_features = tf_idf_feats, 
                                 lowercase    = True, 
                                 norm         = 'l2', 
                                 analyzer     = 'word', 
                                 stop_words   = 'english', 
                                 ngram_range  = (1, 1))

        # compute TF-IDF
        vals = tfidf.fit_transform(df[text_var])
        vals = pd.DataFrame.sparse.from_spmatrix(vals)
        vals.columns = [text_var + '_tfidf_' + str(p) for p in vals.columns]
        df = pd.concat([df, vals], axis = 1)

        # remove original feature
        if drop:
            df.drop(text_var, axis = 1, inplace = True)
        
    # return results
    print('Added {} text-based features.'.format(df.shape[1] - n_feats))
    return df