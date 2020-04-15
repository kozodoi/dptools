###############################
#                             
#      PRINT FACTOR LEVELS
#                             
###############################

def print_factor_levels(df, top = 5):
    '''
    Print levels of categorical features.
    
    --------------------
    Arguments:
    - df (pandas DF): dataset
    - top (int): how many most frequent values to display

    --------------------
    Returns
    - tables with factor levels
    '''

    facs = [f for f in df.columns if df[f].dtype == 'object']
    
    for fac in facs:
        print('-' * 30)
        print(fac + ': ' + str(df[fac].nunique()) + ' unique values')
        print('-' * 30)
        print(df[fac].value_counts(normalize = True, dropna = False).head(top))
        print('-' * 30)
        print('')