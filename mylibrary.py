import csv
import itertools
import os
import sys
import time
from pprint import pprint

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mysql.connector
import numpy
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.preprocessing import MinMaxScaler
# from stellargraph.layer import GCN_LSTM
from tensorflow import keras
from tensorflow.keras import Model

def replace_from_dictionary(input_frame, replace_frame, firstcol, secondcol='NONE'):
    # search and replace word labels based on dictionary file
    dict_lookup = dict(zip(replace_frame['find'], replace_frame['replace']))
    cue_mask = input_frame[firstcol].isin(dict_lookup.keys())
    input_frame.loc[cue_mask, firstcol] = input_frame.loc[cue_mask, firstcol].map(dict_lookup)

    if secondcol != 'NONE':
        target_mask = input_frame[secondcol].isin(dict_lookup.keys())
        input_frame.loc[target_mask, secondcol] = input_frame.loc[target_mask, secondcol].map(dict_lookup)

    return (input_frame)

def limit_and_normalise(input_frame, limit, weight_column):
    # sort data
    input_frame = input_frame.sort_values(weight_column, ascending=False)

    # reduce to top n (limit)
    input_frame = input_frame.head(limit)
    # normalise weights

    scaler = MinMaxScaler()
    # scaler = scaler.fit(input_frame)
    input_frame[[weight_column]] = scaler.fit_transform(input_frame[[weight_column]])
    # scaler.transform(input_frame)

    return input_frame

def load_english_american(filename):
    print("loading English/American dictionary...")
    column_names = ["find", "replace"]
    col_dtypes = {'find': 'string',
                  'replace': 'string'}
    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )
    return (our_dataframe)

def load_buchanan(filename):
    print("loading Buchanan semantic norms...")
    # CUE,TARGET,root,raw,affix,cosine2013,jcn,lsa,fsg,bsg
    # abdomen,intestine,0.287703497,0.254639939,0,,12.643823,0.286157,0.013,0

    col_dtypes = {'CUE': 'string',
                  'TARGET': 'string',
                  'root': 'float',
                  'raw': 'float',
                  'affix': 'float',
                  'cosine2013': 'float',
                  'jcn': 'float',
                  'lsa': 'float',
                  'fsg': 'float',
                  'bsg': 'float'
                  }

    our_dataframe = pd.read_csv(
        filename,
        header=0,
        sep=",",  # separation mark
        dtype=col_dtypes,
        encoding="ISO-8859-1",
        engine='python'
    )
    # convert to `capitals
    for columns in our_dataframe.columns:
        our_dataframe['CUE'] = our_dataframe['CUE'].str.upper()
        our_dataframe['TARGET'] = our_dataframe['TARGET'].str.upper()

    return our_dataframe

def load_cdi_replace(filename):
    print("loading CDI Replace dictionary...")
    column_names = ["find", "replace"]
    col_dtypes = {'find': 'string',
                  'replace': 'string'}
    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )
    return (our_dataframe)

def get_wordbank_wordlist_from_mysql(credentials, targettable):
    # get full list of words being used in the observational data
    # we use this to restrict the graphs

    # Set up connection to MySQL server
    mydb = mysql.connector.connect(
        host=credentials['host'],
        user=credentials['username'],
        password=credentials['password'],
        database=credentials['database']
    )

    # Execute the query and store the result in a DataFrame
    query = "SELECT standardised AS word FROM " + targettable + " GROUP BY standardised;"
    wordsframe = pd.read_sql(query, mydb)

    # Close the connection
    mydb.close()
    return (wordsframe)
