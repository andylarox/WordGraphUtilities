# A Roxburgh 2022
# Mcrae Graph Creator

import pandas as pd

import helper
import mylibrary as ev


def load_mcrae(filename):
    print("loading Mcrae semantic norms...")
    column_names = ["CUE", "TARGET", "WEIGHT"]
    col_dtypes = {'CUE': 'string',
                  'TARGET': 'string',
                  'WEIGHT': 'float'}
    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )

    return our_dataframe


config = helper.read_config()

credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})

cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']

mcrae_file = config['folders']['norms'] + config['files']['mcrae']

edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
mcrae_frame = load_mcrae(mcrae_file)

print("MCRAE...")

# data cleaning
mcrae_frame = ev.replace_from_dictionary(mcrae_frame, english_american_frame, 'CUE', 'TARGET')  # us -> uk
mcrae_frame = ev.replace_from_dictionary(mcrae_frame, cdi_replace_frame, 'CUE', 'TARGET')  # standardise CDI

# create nodes/edges data frame
# count possible nodes and edges : compare CUE words with CDI list
mcrae_cue_match_results = pd.merge(mcrae_frame, purecdi_dataframe, left_on='CUE', right_on='word', how='outer',
                                   indicator=True)

# filter out non-matches leaving only the cdi-matched words
mcrae_filtered_cue_matches = mcrae_cue_match_results[(mcrae_cue_match_results['_merge'] == 'both')]
mcrae_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

# compare target of matched cue rows with cdi
mcrae_cue_target_match_results = pd.merge(mcrae_filtered_cue_matches, purecdi_dataframe, left_on='TARGET',
                                          right_on='word', how='outer', indicator=True)
mcrae_filtered_cue_targ_matches = mcrae_cue_target_match_results[
    (mcrae_cue_target_match_results['_merge'] == 'both')]
mcrae_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
print("mcrae dataset can generate " + str(len(mcrae_filtered_cue_targ_matches.index)) + " edges.")

unconnected_cue_mcrae_frame = mcrae_cue_match_results[(mcrae_cue_match_results['_merge'] == 'right_only')]
unconnected_cue_mcrae_frame['CUE'] = unconnected_cue_mcrae_frame['word']
unconnected_cue_mcrae_frame['TARGET'] = unconnected_cue_mcrae_frame['word']
unconnected_cue_mcrae_frame['WEIGHT'] = 0
unconnected_cue_mcrae_frame.drop(['_merge', 'word'], axis=1, inplace=True)

# now check targets
# get target results that do not match cdi (i.e. cdi only)
unconnected_target_mcrae_frame = mcrae_cue_target_match_results[(mcrae_cue_target_match_results['_merge'] == 'right_only')]
unconnected_target_mcrae_frame['CUE'] = unconnected_target_mcrae_frame['word_y']
unconnected_target_mcrae_frame['TARGET'] = unconnected_target_mcrae_frame['word_y']
unconnected_target_mcrae_frame['WEIGHT'] = 0
unconnected_target_mcrae_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

# merge
unconnected_mcrae_frame = unconnected_cue_mcrae_frame.append(unconnected_target_mcrae_frame)

# limit and normalise
mcrae_filtered_cue_targ_matches = ev.limit_and_normalise(mcrae_filtered_cue_targ_matches, int(config['edges_threshold']['limit']), 'WEIGHT')

combined_frame = mcrae_filtered_cue_targ_matches.append(unconnected_mcrae_frame)
mcrae_edges_output_frame = combined_frame[['CUE', 'TARGET', 'WEIGHT']].copy(deep=True)
mcrae_edges_output_frame.rename(columns={'CUE': 'source', 'TARGET': 'target', 'WEIGHT': 'weight'}, inplace=True)

average_edge_weight = ev.calculate_edge_weight_average(mcrae_edges_output_frame, 'weight')
print("Average edge weight = " + str(average_edge_weight))
print("mcrae dataset will have " + str(len(unconnected_mcrae_frame.index)) + " unconnected nodes.")
print('Edges file: ')

output_filename = edges_files_folder + 'mcrae_edges.csv'
mcrae_edges_output_frame.to_csv(output_filename, index=False)

print(output_filename)
print("mcrae edge files generated.")
