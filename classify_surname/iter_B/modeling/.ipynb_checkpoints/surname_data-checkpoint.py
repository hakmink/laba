import modeling.surname_common as sc
from sklearn.utils import shuffle
import glob
import os
import re
import pprint
import pandas as pd
import unicodedata
import string


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in sc.ALL_LETTERS
    )

def load_surnames_from_txt():
    df_surnames = pd.DataFrame()
    list_ = []

    for filename in glob.glob('data/names/*.txt'):
        m = re.match(r'(.*)\/(.*?)\.txt', filename)
        category = m.group(2)
        df = pd.read_csv(filename,names=['surname'])
        df['category'] = category
        list_.append(df)
    df_surnames = pd.concat(list_)   
    df_surnames['normalized'] = df_surnames['surname'].apply(lambda x: unicode_to_ascii(x))
    
    series_categories = df_surnames.groupby(['category'])['category'].count()
    df_categories = pd.DataFrame({
        'category':series_categories.index, 
        'freq':series_categories.tolist(), 
        'index':range(0,len(series_categories))
    })
    
    df_surnames = pd.merge(df_surnames, df_categories, how='left', on='category')
    
    return df_surnames, df_categories

def save_df_surnames_as_pickle():
    df_surnames, df_categories = load_surnames_from_txt()
    # train test split
    df = shuffle(df_surnames, random_state=sc.RANDOM_STATE)
    train_cnt = int(df['surname'].count()*sc.TRAIN_TEST_RATIO)
    train = df[0:train_cnt]
    test = df[train_cnt+1:]
    # save as pickle
    df_surnames.to_pickle('data/pickles/df_surnames.pickle',compression='bz2')
    df_categories.to_pickle('data/pickles/df_categories.pickle',compression='bz2')
    train.to_pickle('data/pickles/train.pickle',compression='bz2')
    test.to_pickle('data/pickles/test.pickle',compression='bz2')
    # train test stat  
    t1 = train.groupby(['category']).count().drop(['normalized'],axis=1)
    t2 = test.groupby(['category']).count().drop(['normalized'],axis=1)
    t1.columns = ['surname_train']
    t2.columns = ['surname_test']
    tt = pd.DataFrame(pd.merge(t1, t2, left_index=True, right_index=True))
    tt['ratio'] = tt['surname_train'] / (tt['surname_train'] + tt['surname_test'])
    tt.to_pickle('data/pickles/train_test_stat.pickle',compression='bz2')
    return tt


def load_df_surnames():
    df_train = pd.read_pickle('data/pickles/train.pickle',compression='bz2')
    df_test = pd.read_pickle('data/pickles/test.pickle',compression='bz2')
    
    return df_train, df_test

def load_df_categories():
    return pd.read_pickle('data/pickles/df_categories.pickle',compression='bz2')