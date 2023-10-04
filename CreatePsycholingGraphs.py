# A Roxburgh 2022
# Psycholinguistic Relationship Graphs Creator

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

cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']

glasgow_file = config['folders']['norms'] + config['files']['glasgow']
edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
glasgow_norms_frame = ev.load_glasgow_norms(glasgow_file)

print("Psycholinguistic relationships...")
glasgow_norms_frame = ev.replace_from_dictionary(glasgow_norms_frame, cdi_replace_frame, 'word')  # standardise CDI

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
##########
# threshold = 0.5

# ------------ Define dataframes ----------
glasgow_arousal = pd.DataFrame(columns=col_dtypes)
glasgow_valence = pd.DataFrame(columns=col_dtypes)
glasgow_dominance = pd.DataFrame(columns=col_dtypes)
glasgow_concreteness = pd.DataFrame(columns=col_dtypes)
glasgow_imageability = pd.DataFrame(columns=col_dtypes)
glasgow_familiarity = pd.DataFrame(columns=col_dtypes)
glasgow_aoa = pd.DataFrame(columns=col_dtypes)
glasgow_size = pd.DataFrame(columns=col_dtypes)
glasgow_gender = pd.DataFrame(columns=col_dtypes)

# ----------------------- generate glasgow connections ---------------------------
threshold = config['edges_threshold']['glasgow']
print(ev.generate_connection_weights('Arousal', 'word', 'arou.m', 'glasgow_arousal_edges.csv',
                                     glasgow_arousal, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Valence', 'word', 'val.m', 'glasgow_valence_edges.csv',
                                     glasgow_valence, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Dominance', 'word', 'dom.m', 'glasgow_dominance_edges.csv',
                                     glasgow_dominance, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Concreteness', 'word', 'cnc.m', 'glasgow_concreteness_edges.csv',
                                     glasgow_concreteness, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Imageability', 'word', 'imag.m', 'glasgow_imageability_edges.csv',
                                     glasgow_imageability, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Familiarity', 'word', 'fam.m', 'glasgow_familiarity_edges.csv',
                                     glasgow_familiarity, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('AOA', 'word', 'aoa.m', 'glasgow_aoa_edges.csv',
                                     glasgow_aoa, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Size', 'word', 'size.m', 'glasgow_size_edges.csv',
                                     glasgow_size, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))
print(ev.generate_connection_weights('Gender', 'word', 'gend.m', 'glasgow_gend_edges.csv',
                                     glasgow_gender, float(threshold), edges_files_folder,
                                     glasgow_filtered_cue_matches, purecdi_dataframe))