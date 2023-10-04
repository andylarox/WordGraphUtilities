# A Roxburgh 2022
# 'No Relationship' Graph Creator
# Version 1.0

import mylibrary as ev
import pandas as pd
import helper

config = helper.read_config()


def load_norelationship(filename):
    print("loading 'no relationship' data...")
    column_names = ["WORD1", "WORD2", "WEIGHT"]
    col_dtypes = {'WORD1': 'string', 'WORD2': 'string',
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

    return (our_dataframe)


credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})

cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']

norelationship_file = config['folders']['norms'] + config['files']['norelationship']
edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)

print("norelationship...")
norelationship_frame = load_norelationship(norelationship_file)

# data cleaning
# standardise CDI
norelationship_frame = ev.replace_from_dictionary(norelationship_frame, cdi_replace_frame, 'WORD1', 'WORD2')

# create nodes/edges data frame
# count possible nodes and edges : compare CUE words with CDI list
norelationship_cue_match_results = pd.merge(norelationship_frame, purecdi_dataframe, left_on='WORD1', right_on='word', how='outer', indicator=True)

# filter out non-matches leaving only the cdi-matched words
norelationship_filtered_cue_matches = norelationship_cue_match_results[(norelationship_cue_match_results['_merge'] == 'both')]
norelationship_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

# compare TARGET of matched CUE rows with CDI
norelationship_cue_target_match_results = pd.merge(norelationship_filtered_cue_matches, purecdi_dataframe, left_on='WORD2', right_on='word', how='outer', indicator=True)
norelationship_filtered_cue_targ_matches = norelationship_cue_target_match_results[(norelationship_cue_target_match_results['_merge'] == 'both')]
norelationship_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
print("norelationship dataset can generate " + str(len(norelationship_filtered_cue_targ_matches.index)) + " edges.")

# create self-loops frame
# cue non-matches

# get cue results that do not match cdi (i.e. cdi only)
unconnected_cue_norelationship_frame = norelationship_cue_match_results[(norelationship_cue_match_results['_merge'] == 'right_only')]
unconnected_cue_norelationship_frame['WORD1'] = unconnected_cue_norelationship_frame['word']
unconnected_cue_norelationship_frame['WORD2'] = unconnected_cue_norelationship_frame['word']
unconnected_cue_norelationship_frame['WEIGHT'] = 0

unconnected_cue_norelationship_frame.drop(['_merge', 'word'], axis=1, inplace=True)

# now check targets
# print('targets')
# get target results that do not match cdi (i.e. cdi only)
unconnected_target_norelationship_frame = norelationship_cue_target_match_results[(norelationship_cue_target_match_results['_merge'] == 'right_only')]
unconnected_target_norelationship_frame['WORD1'] = unconnected_target_norelationship_frame['word_y']
unconnected_target_norelationship_frame['WORD2'] = unconnected_target_norelationship_frame['word_y']
unconnected_target_norelationship_frame['WEIGHT'] = 0
unconnected_target_norelationship_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

# merge
unconnected_norelationship_frame = unconnected_cue_norelationship_frame.append(unconnected_target_norelationship_frame)
combined_frame = norelationship_filtered_cue_targ_matches.append(unconnected_norelationship_frame)
norelationship_edges_output_frame = combined_frame[['WORD1', 'WORD2', 'WEIGHT']].copy(deep=True)
norelationship_edges_output_frame.rename(columns={'WORD1': 'source', 'WORD2': 'target', 'WEIGHT': 'weight'}, inplace=True)

# print("self loops to add: " + unconnected_norelationship_frame.to_string())
print("norelationship dataset will have " + str(len(unconnected_norelationship_frame.index)) + " unconnected nodes.")

average_edge_weight = ev.calculate_edge_weight_average(norelationship_edges_output_frame, 'weight')
print("Average edge weight = " + str(average_edge_weight))
print('Edges file: ')
# edges file output : nelson -------
output_filename = edges_files_folder + 'norelationship_edges.csv'
norelationship_edges_output_frame.to_csv(output_filename, index=False)
print(output_filename)

print("norelationship graph files generated.")
