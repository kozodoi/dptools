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



###############################
#                             
#      ADD TEXT FEATURES      
#                             
###############################

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse

def add_text_features(df, 
                      string_vars, 
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
    - string_vars (list): list of textual features
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
    df_new = add_text_features(df, string_vars = ['income', 'gender])
    '''

    ##### PROCESSING LOOP
    for var in string_vars:


        ### TEXT PREPROCESSING

        # replace NaN with empty string
        df[var].fillna('', inplace = True)

        # remove common and rare words
        freq = pd.Series(' '.join(df[var]).split()).value_counts()[:common_words]
        freq = pd.Series(' '.join(df[var]).split()).value_counts()[-rare_words:]

        # convert to lowercase 
        df[var] = df[var].apply(lambda x: ' '.join(x.lower() for x in x.split())) 

        # remove punctuation
        df[var] = df[var].str.replace('[^\w\s]','')         


        ### COMPUTE BASIC FEATURES

        # word count
        df[var + '_word_count'] = df[var].apply(lambda x: len(str(x).split(' ')))
        df[var + '_word_count'][df[var] == ''] = 0

        # character count
        df[var + '_char_count'] = df[var].str.len().fillna(0).astype('int64')


        ### COMPUTE TF-IDF FEATURES

        # import vectorizer
        tfidf  = TfidfVectorizer(max_features = tf_idf_feats, 
                                 lowercase    = True, 
                                 norm         = 'l2', 
                                 analyzer     = 'word', 
                                 stop_words   = 'english', 
                                 ngram_range  = (1, 1))

        # compute TF-IDF
        vals = tfidf.fit_transform(df[var])
        vals = pd.DataFrame.sparse.from_spmatrix(vals)
        vals.columns = [var + '_tfidf_' + str(p) for p in vals.columns]
        df = pd.concat([df, vals], axis = 1)


        ### CORRECTIONS

        # remove raw text
        if drop == True:
            del df[var]

        # print dimensions
        print(df.shape)
        
    # return df
    return df