# A Roxburgh 2022
# Nelson Graph Creator
# Version 1.0

import pandas as pd
import mylibrary as ev
import helper


def load_nelson(filename):
    print("loading Nelson cues/targets...")
    # A,ALPHABET,YES,152,10,0.066,0.046,0.0020,0.0000,2,5,0.0,0,13,9999,¥,,N,0.62,0.25,0.019,0,11,2,¥,,N,0.50,0.25,0.062,1

    column_names = ["CUE", "TARGET", "NORMED", "G", "P",
                    "FSG", "BSG", "MSG", "OSG"]

    col_dtypes = {"CUE": 'string',
                  "TARGET": 'string',
                  "NORMED": 'string',
                  "G": 'int',
                  "P": 'int',
                  "FSG": 'float',
                  "BSG": 'float',
                  "MSG": 'float',
                  "OSG": 'float'}

    our_dataframe = pd.read_csv(
        filename,
        sep=",",  # separation mark
        header=None,  # no heading row
        dtype=col_dtypes,
        names=[*column_names],
        encoding="ISO-8859-1",
        engine='python'
    )
    # convert to capitals

    return our_dataframe


config = helper.read_config()

credentials = dict({'host': config['database']['host'],
                    'username': config['database']['username'],
                    'password': config['database']['password'],
                    'database': config['database']['database']})

cdi_replace_file = config['folders']['cleansing_dictionaries'] + config['files']['cdi_replace']
english_american_file = config['folders']['cleansing_dictionaries'] + config['files']['english_american']
nelson_file = config['folders']['norms'] + config['files']['nelson']
edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)

nelson_frame = load_nelson(nelson_file)

print("NELSON...")
# data cleaning
# us -> uk
nelson_frame = ev.replace_from_dictionary(nelson_frame, english_american_frame, 'CUE', 'TARGET')
# standardise CDI
nelson_frame = ev.replace_from_dictionary(nelson_frame, cdi_replace_frame, 'CUE', 'TARGET')

# create nodes/edges data frame
# count possible nodes and edges : compare CUE words with CDI list
nelson_cue_match_results = pd.merge(nelson_frame, purecdi_dataframe, left_on='CUE', right_on='word', how='outer', indicator=True)

# filter out non-matches leaving only the cdi-matched words
nelson_filtered_cue_matches = nelson_cue_match_results[(nelson_cue_match_results['_merge'] == 'both')]
nelson_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

# compare target of matched cue rows with cdi
nelson_cue_target_match_results = pd.merge(nelson_filtered_cue_matches, purecdi_dataframe, left_on='TARGET', right_on='word', how='outer', indicator=True)
nelson_filtered_cue_targ_matches = nelson_cue_target_match_results[(nelson_cue_target_match_results['_merge'] == 'both')]
nelson_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
print("Unlimited Nelson dataset can generate " + str(len(nelson_filtered_cue_targ_matches.index)) + " edges.")

# create self-loops frame
# cue non-matches
# print('cues')
# get cue results that do not match cdi (i.e. cdi only)
unconnected_cue_nelson_frame = nelson_cue_match_results[(nelson_cue_match_results['_merge'] == 'right_only')]
unconnected_cue_nelson_frame['CUE'] = unconnected_cue_nelson_frame['word']
unconnected_cue_nelson_frame['TARGET'] = unconnected_cue_nelson_frame['word']
unconnected_cue_nelson_frame['FSG'] = 0
unconnected_cue_nelson_frame['BSG'] = 0
unconnected_cue_nelson_frame['MSG'] = 0
unconnected_cue_nelson_frame['OSG'] = 0

unconnected_cue_nelson_frame.drop(['_merge', 'word'], axis=1, inplace=True)

# now check targets
# print('targets')
# get target results that do not match cdi (i.e. cdi only)
unconnected_target_nelson_frame = nelson_cue_target_match_results[(nelson_cue_target_match_results['_merge'] == 'right_only')]


unconnected_target_nelson_frame['CUE'] = unconnected_target_nelson_frame['word_y']
unconnected_target_nelson_frame['TARGET'] = unconnected_target_nelson_frame['word_y']
unconnected_target_nelson_frame['FSG'] = 0
unconnected_target_nelson_frame['BSG'] = 0
unconnected_target_nelson_frame['MSG'] = 0
unconnected_target_nelson_frame['OSG'] = 0

unconnected_target_nelson_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

# merge
unconnected_nelson_frame = unconnected_cue_nelson_frame.append(unconnected_target_nelson_frame)
combined_frame = nelson_filtered_cue_targ_matches.append(unconnected_nelson_frame)
nelson_edges_output_frame = combined_frame[['CUE', 'TARGET', 'FSG']].copy(deep=True)
nelson_edges_output_frame.rename(columns={'CUE': 'source', 'TARGET': 'target', 'FSG': 'weight'}, inplace=True)
average_edge_weight = ev.calculate_edge_weight_average(nelson_edges_output_frame, 'weight')

# limit and normalise
nelson_edges_output_frame = ev.limit_and_normalise(nelson_edges_output_frame, int(config['edges_threshold']['limit']), 'weight')

print("self loops to add: " + unconnected_nelson_frame.to_string())
print("Nelson dataset will have " + str(len(unconnected_nelson_frame.index)) + " unweighted edges.")
print("Average edge weight = " + str(average_edge_weight))
print('nelson output:')
print('Edges file: ')

# edges file output : nelson -------
output_filename = edges_files_folder + 'nelson_edges.csv'
nelson_edges_output_frame.to_csv(output_filename, index=False)
print(output_filename)

# print(nodes_dataframe)
print("Nelson edges file generated.")
