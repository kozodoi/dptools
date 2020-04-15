###############################
#                             
#      ADD TEXT FEATURES      
#                             
###############################

def add_text_features(df, 
                      string_vars, 
                      tf_idf_feats = 5, 
                      common_words = 10,
                      rare_words = 10,
                      drop = True):
    '''
    Add basic text-based features including word count, character count and 
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
    '''

    ##### PROCESSING LOOP
    for var in string_vars:


        ### TEXT PREPROCESSING

        # replace NaN with empty string
        df[var][pd.isnull(df[var])] = ''

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
        vals = pd.SparsedfFrame(vals)
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