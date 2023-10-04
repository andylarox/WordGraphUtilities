# A Roxburgh 2022
# Buchanan Graph Creator
# Version 1.0

import mylibrary as ev
import pandas as pd
import helper


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


config = helper.read_config()

credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})

cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']

buchanan_file = config['folders']['norms'] + config['files']['buchanan']
edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
buchanan_frame = load_buchanan(buchanan_file)
print("BUCHANAN...")

# CUE,TARGET,root,raw,affix,cosine2013,jcn,lsa,fsg,bsg
# data cleaning
# us -> uk
buchanan_frame = ev.replace_from_dictionary(buchanan_frame, english_american_frame, 'CUE', 'TARGET')
# standardise CDI
buchanan_frame = ev.replace_from_dictionary(buchanan_frame, cdi_replace_frame, 'CUE', 'TARGET')

# compare CUE/CDI - filter out non-matches leaving only the cdi-matched words
buchanan_cue_match_results = pd.merge(buchanan_frame, purecdi_dataframe, left_on='CUE', right_on='word', how='outer', indicator=True)
buchanan_filtered_cue_matches = buchanan_cue_match_results[(buchanan_cue_match_results['_merge'] == 'both')]
buchanan_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

# compare TARGET of matched CUE rows with CDI
buchanan_cue_target_match_results = pd.merge(buchanan_filtered_cue_matches, purecdi_dataframe, left_on='TARGET', right_on='word', how='outer', indicator=True)
buchanan_filtered_cue_targ_matches = buchanan_cue_target_match_results[(buchanan_cue_target_match_results['_merge'] == 'both')]
buchanan_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
print("buchanan dataset can generate " + str(len(buchanan_filtered_cue_targ_matches.index)) + " edges.")

# create self-loops frame======================
# cue non-matches

# get cue results that do not match cdi (i.e. cdi only)
unconnected_cue_buchanan_frame = buchanan_cue_match_results[(buchanan_cue_match_results['_merge'] == 'right_only')]
unconnected_cue_buchanan_frame['CUE'] = unconnected_cue_buchanan_frame['word']
unconnected_cue_buchanan_frame['TARGET'] = unconnected_cue_buchanan_frame['word']
unconnected_cue_buchanan_frame['root'] = 0

# print(unconnected_cue_buchanan_frame.to_string())
unconnected_cue_buchanan_frame.drop(['_merge', 'word'], axis=1, inplace=True)

# get target results that do not match cdi (i.e. cdi only)
unconnected_target_buchanan_frame = buchanan_cue_target_match_results[(buchanan_cue_target_match_results['_merge'] == 'right_only')]
# print(unconnected_target_buchanan_frame.to_string())

unconnected_target_buchanan_frame['CUE'] = unconnected_target_buchanan_frame['word_y']
unconnected_target_buchanan_frame['TARGET'] = unconnected_target_buchanan_frame['word_y']
unconnected_target_buchanan_frame['root'] = 0
# print(unconnected_target_buchanan_frame.to_string())
unconnected_target_buchanan_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

# merge
unconnected_buchanan_frame = unconnected_cue_buchanan_frame.append(unconnected_target_buchanan_frame)

# limit and normalise
buchanan_filtered_cue_targ_matches = ev.limit_and_normalise(buchanan_filtered_cue_targ_matches, int(config['edges_threshold']['limit']), 'root')

combined_frame = buchanan_filtered_cue_targ_matches.append(unconnected_buchanan_frame)
buchanan_edges_output_frame = combined_frame[['CUE', 'TARGET', 'root']].copy(deep=True)
buchanan_edges_output_frame.rename(columns={'CUE': 'source', 'TARGET': 'target', 'root': 'weight'}, inplace=True)

print("buchanan dataset will have " + str(len(unconnected_buchanan_frame.index)) + " unconnected nodes.")

print('Edges file: ')
# edges file output : buchanan -------
output_filename = edges_files_folder + 'buchanan_edges.csv'
buchanan_edges_output_frame.to_csv(output_filename, index=False)
print(output_filename)

print("Buchanan edge files generated.")
