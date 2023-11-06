# A Roxburgh 2022
# Rhyming Graph Creator
# Version 1.0
import os

import pandas as pd
import helper
import mylibrary as ev

def load_rhyming(filename):
    print("loading rhyming norms...")
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



config = helper.read_config()

credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})

cdi_replace_file = os.getcwd() + config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = os.getcwd() + config['folders']['cleansing_dictionaries'] + config['files']['english_american']
rhyming_file = os.getcwd() + config['folders']['norms'] + config['files']['rhyming']
edges_files_folder = os.getcwd() + config['folders']['edges_files_folder']
purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)
rhyming_frame = load_rhyming(rhyming_file)

print("RHYMING...")
# rhyming_frame = ev.load_rhyming(rhyming_file)
# data cleaning
rhyming_frame = ev.replace_from_dictionary(rhyming_frame, english_american_frame, 'WORD1', 'WORD2')  # us -> uk
rhyming_frame = ev.replace_from_dictionary(rhyming_frame, cdi_replace_frame, 'WORD1', 'WORD2')  # standardise CDI

# create nodes/edges data frame
# count possible nodes and edges : compare CUE words with CDI list
rhyming_cue_match_results = pd.merge(rhyming_frame, purecdi_dataframe, left_on='WORD1', right_on='word', how='outer', indicator=True)

# filter out non-matches leaving only the cdi-matched words
rhyming_filtered_cue_matches = rhyming_cue_match_results[(rhyming_cue_match_results['_merge'] == 'both')]
rhyming_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

# compare TARGET of matched CUE rows with CDI
rhyming_cue_target_match_results = pd.merge(rhyming_filtered_cue_matches, purecdi_dataframe, left_on='WORD2', right_on='word', how='outer', indicator=True)
rhyming_filtered_cue_targ_matches = rhyming_cue_target_match_results[(rhyming_cue_target_match_results['_merge'] == 'both')]
rhyming_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
print("rhyming dataset can generate " + str(len(rhyming_filtered_cue_targ_matches.index)) + " edges.")

# create self-loops frame
# cue non-matches

# get cue results that do not match cdi (i.e. cdi only)
unconnected_cue_rhyming_frame = rhyming_cue_match_results[(rhyming_cue_match_results['_merge'] == 'right_only')]
unconnected_cue_rhyming_frame['WORD1'] = unconnected_cue_rhyming_frame['word']
unconnected_cue_rhyming_frame['WORD2'] = unconnected_cue_rhyming_frame['word']
unconnected_cue_rhyming_frame['WEIGHT'] = 0

# print(unconnected_cue_rhyming_frame.to_string())

unconnected_cue_rhyming_frame.drop(['_merge', 'word'], axis=1, inplace=True)

# now check targets
# print('targets')
# get target results that do not match cdi (i.e. cdi only)
unconnected_target_rhyming_frame = rhyming_cue_target_match_results[(rhyming_cue_target_match_results['_merge'] == 'right_only')]
# print(unconnected_target_rhyming_frame.to_string())

unconnected_target_rhyming_frame['WORD1'] = unconnected_target_rhyming_frame['word_y']
unconnected_target_rhyming_frame['WORD2'] = unconnected_target_rhyming_frame['word_y']
unconnected_target_rhyming_frame['WEIGHT'] = 0
# print(unconnected_target_rhyming_frame.to_string())
unconnected_target_rhyming_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

# merge
unconnected_rhyming_frame = unconnected_cue_rhyming_frame.append(unconnected_target_rhyming_frame)
combined_frame = rhyming_filtered_cue_targ_matches.append(unconnected_rhyming_frame)
rhyming_edges_output_frame = combined_frame[['WORD1', 'WORD2', 'WEIGHT']].copy(deep=True)
rhyming_edges_output_frame.rename(columns={'WORD1': 'source', 'WORD2': 'target', 'WEIGHT': 'weight'}, inplace=True)

# print("self loops to add: " + unconnected_rhyming_frame.to_string())
print("rhyming dataset will have " + str(len(unconnected_rhyming_frame.index)) + " unconnected nodes.")

average_edge_weight = ev.calculate_edge_weight_average(rhyming_edges_output_frame, 'weight')
print("Average edge weight = " + str(average_edge_weight))
print('Edges file: ')
# edges file output : nelson -------
output_filename = edges_files_folder + 'rhyming_edges.csv'
rhyming_edges_output_frame.to_csv(output_filename, index=False)
print(output_filename)

# nodes file output : rhyming -------
# nodes_result = ev.create_nodes_file("rhyming", survey_df_adjusted, ukcdi_questions_lookup_frame, node_files_folder)
# print(f"Total node files:", nodes_result)
print("rhyming graph files generated.")
