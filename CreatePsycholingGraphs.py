# A Roxburgh 2022
# Psycholinguistic Relationship Graphs Creator
# version 1.2
import os

import pandas as pd
import helper
import mylibrary as ev


def load_glasgow_norms(filename):
    print("loading Glasgow norms...")

    col_dtypes = {
        'word': 'str',
        'len': 'int',
        'arou.m': 'float',
        'arou.sd': 'float',
        'arou.n': 'float',
        'val.m': 'float',
        'val.sd': 'float',
        'val.n': 'float',
        'dom.m': 'float',
        'dom.sd': 'float',
        'dom.n': 'float',
        'cnc.m': 'float',
        'cnc.sd': 'float',
        'cnc.n': 'float',
        'imag.m': 'float',
        'imag.sd': 'float',
        'imag.n': 'float',
        'fam.m': 'float',
        'fam.sd': 'float',
        'fam.n': 'float',
        'aoa.m': 'float',
        'aoa.sd': 'float',
        'aoa.n': 'float',
        'size.m': 'float',
        'size.sd': 'float',
        'size.n': 'float',
        'gend.m': 'float',
        'gend.sd': 'float',
        'gend.n': 'float',
    }

    our_dataframe = pd.read_csv(
        filename,
        header=0,
        sep=",",  # separation mark
        dtype=col_dtypes,
        encoding="ISO-8859-1",
        engine='python'
    )
    for columns in our_dataframe.columns:
        our_dataframe['word'] = our_dataframe['word'].str.upper()
        # our_dataframe['word2'] = our_dataframe['word2'].str.upper()
    return our_dataframe


config = helper.read_config()

credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})
limit = config['edges_threshold']['limit']
cdi_replace_file = os.getcwd() + config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = os.getcwd() + config['folders']['cleansing_dictionaries'] + config['files']['english_american']

glasgow_file = os.getcwd() + config['folders']['norms'] + config['files']['glasgow']
edges_files_folder = os.getcwd() + config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
glasgow_norms_frame = load_glasgow_norms(glasgow_file)

print("Psycholinguistic relationships...")
glasgow_norms_frame = ev.replace_from_dictionary(glasgow_norms_frame, cdi_replace_frame, 'word')  # standardise CDI


def compute_normalized_word_size(dataframe):
    dataframe['word_size'] = dataframe['word'].apply(
        lambda x: len(''.join(c for c in x if c.isalpha()))
    )
    max_word_size = dataframe['word_size'].max()
    dataframe['word_size'] = dataframe['word_size'] / max_word_size

    return dataframe


glasgow_norms_frame = compute_normalized_word_size(glasgow_norms_frame)


# function to preprocess gender data and add two new columns...
def preprocess_gender_data(dataframe, threshold=4.143):
    dataframe['male'] = dataframe['gend.m'].apply(
        lambda x: x if x >= threshold else 0
    )
    dataframe['female'] = dataframe['gend.m'].apply(
        lambda x: 6.971 - x if x < threshold else 0
    )

    # Normalizing male and female columns to range [0, 1]
    dataframe['male'] = dataframe['male'] / dataframe['male'].max()
    dataframe['female'] = dataframe['female'] / dataframe['female'].max()

    # Create 'gender' column as the maximum of the 'male' and 'female' scores
    dataframe['gender'] = dataframe[['male', 'female']].max(axis=1)

    return dataframe


# Updating the main dataframe with the new columns...
glasgow_norms_frame = preprocess_gender_data(glasgow_norms_frame)

# count possible nodes and edges
df3_merged = pd.merge(glasgow_norms_frame,
                      purecdi_dataframe,
                      left_on='word',
                      right_on='word',
                      how='outer',
                      indicator=True)
dfx = df3_merged[(df3_merged['_merge'] == 'both')]
# drop _merge column
dfx.drop('_merge', axis=1, inplace=True)
# show matching word count
print("glasgow matches " + str(len(dfx.index)) + " words.")
xdfxr = df3_merged[(df3_merged['_merge'] == 'right_only')]
# show non-matching word count
print("glasgow doesn't match " + str(len(xdfxr.index)) + " words.")

print(" Processing Psycholinguistic relationships...")
# compare CUE/CDI
glasgow_cue_match_results = pd.merge(glasgow_norms_frame,
                                     purecdi_dataframe,
                                     left_on='word',
                                     right_on='word',
                                     how='outer',
                                     indicator=True)
glasgow_filtered_cue_matches = glasgow_cue_match_results[(glasgow_cue_match_results['_merge'] == 'both')]
glasgow_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

col_dtypes = {
    'CUE': 'string',
    'TARGET': 'string',
    'WEIGHT': 'float',
    'REVSTR': 'string'
}

# Defining attributes with threshold value from config.
attributes = [
    ("aoa", "aoa.m", "glasgow_aoa_edges.csv", config['edges_threshold']['aoa']),
    ("word_size", "word_size", "glasgow_word_size_edges.csv", config['edges_threshold']['word_size']),
    ("male", "male", "glasgow_male_edges.csv", config['edges_threshold']['male']),
    ("female", "female", "glasgow_female_edges.csv", config['edges_threshold']['female']),
    ("gender", "gender", "glasgow_gender_edges.csv", config['edges_threshold']['gender']),
    ("size", "size.m", "glasgow_size_edges.csv", config['edges_threshold']['size']),
    ("arousal", "arou.m", "glasgow_arousal_edges.csv", config['edges_threshold']['arousal']),
    ("valence", "val.m", "glasgow_valence_edges.csv", config['edges_threshold']['valence']),
    ("concreteness", "cnc.m", "glasgow_concreteness_edges.csv", config['edges_threshold']['concreteness']),
    ("imageability", "imag.m", "glasgow_imageability_edges.csv", config['edges_threshold']['imageability']),
    ("familiarity", "fam.m", "glasgow_familiarity_edges.csv", config['edges_threshold']['familiarity']),
    ("dominance", "dom.m", "glasgow_dominance_edges.csv", config['edges_threshold']['dominance'])
]

dfs = {attr[0]: pd.DataFrame(columns=col_dtypes) for attr in attributes}


def generate_glasgow_connections(attribute, mean_type, csv_title, threshold, limit):
    print(ev.generate_connection_weights(
        attribute.capitalize(),
        'word',
        mean_type,
        csv_title,
        dfs[attribute],
        threshold,
        edges_files_folder,
        glasgow_filtered_cue_matches,
        purecdi_dataframe, limit
    ))


# ----------------------- generate glasgow connections ---------------------------
for attribute, mean_type, csv_title, threshold in attributes:
    generate_glasgow_connections(attribute, mean_type, csv_title, float(threshold), limit)
