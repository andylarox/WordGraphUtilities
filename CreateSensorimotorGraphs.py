# A Roxburgh 2022
# Sensorimnotor Graph Creator
# Version 1.1

import pandas as pd
import helper
import mylibrary as ev

def load_lancaster_sensorimotor_norms(filename):
    print("loading Lancaster norms...")
    col_dtypes = {
        'Word': 'string',
        'Auditory.mean': 'float',
        'Gustatory.mean': 'float',
        'Haptic.mean': 'float',
        'Interoceptive.mean': 'float',
        'Olfactory.mean': 'float',
        'Visual.mean': 'float',
        'Foot_leg.mean': 'float',
        'Hand_arm.mean': 'float',
        'Head.mean': 'float',
        'Mouth.mean': 'float',
        'Torso.mean': 'float',
        'Auditory.SD': 'float',
        'Gustatory.SD': 'float',
        'Haptic.SD': 'float',
        'Interoceptive.SD': 'float',
        'Olfactory.SD': 'float',
        'Visual.SD': 'float',
        'Foot_leg.SD': 'float',
        'Hand_arm.SD': 'float',
        'Head.SD': 'float',
        'Mouth.SD': 'float',
        'Torso.SD': 'float',
        'Max_strength.perceptual': 'float',
        'Minkowski3.perceptual': 'float',
        'Exclusivity.perceptual': 'float',
        'Dominant.perceptual': 'string',
        'Max_strength.action': 'float',
        'Minkowski3.action': 'float',
        'Exclusivity.action': 'float',
        'Dominant.action': 'string',
        'Max_strength.sensorimotor': 'float',
        'Minkowski3.sensorimotor': 'float',
        'Exclusivity.sensorimotor': 'float',
        'Dominant.sensorimotor': 'string',
        'N_known.perceptual': 'float',
        'List_N.perceptual': 'float',
        'Percent_known.perceptual': 'float',
        'N_known.action': 'float',
        'List_N.action': 'float',
        'Percent_known.action': 'float',
        'Mean_age.perceptual': 'float',
        'Mean_age.action': 'float',
        'List#.perceptual': 'string',
        'List#.action': 'string'}

    our_dataframe = pd.read_csv(
        filename,
        header=0,
        sep=",",  # separation mark
        dtype=col_dtypes,
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

lancaster_sensorimotor_norms = config['folders']['norms'] + config['files']['Lancaster_sensorimotor']
lancaster_sensorimotor_norms_frame = load_lancaster_sensorimotor_norms(lancaster_sensorimotor_norms)

edges_files_folder = config['folders']['edges_files_folder']

purecdi_dataframe = ev.get_wordbank_wordlist_from_mysql(credentials, config['database']['observ_data_table'])
english_american_frame = ev.load_english_american(english_american_file)
cdi_replace_frame = ev.load_cdi_replace(cdi_replace_file)

print("sensorimotor...")
# lancaster_sensorimotor_norms_frame = ev.load_lancaster(lancaster_sensorimotor_norms)
# data cleaning
lancaster_sensorimotor_norms_frame = ev.replace_from_dictionary(lancaster_sensorimotor_norms_frame, english_american_frame, 'Word')  # us -> uk
lancaster_sensorimotor_norms_frame = ev.replace_from_dictionary(lancaster_sensorimotor_norms_frame, cdi_replace_frame, 'Word')  # standardise CDI

# count possible nodes and edges
df3_merged = pd.merge(lancaster_sensorimotor_norms_frame,
                      purecdi_dataframe,
                      left_on='Word',
                      right_on='word',
                      how='outer',
                      indicator=True)
dfx = df3_merged[(df3_merged['_merge'] == 'both')]
# drop _merge column
dfx.drop('_merge', axis=1, inplace=True)
# show matching word count
print("Lancaster matches " + str(len(dfx.index)) + " words.")
xdfxr = df3_merged[(df3_merged['_merge'] == 'right_only')]
# show non-matching word count
print("Lancaster doesn't match " + str(len(xdfxr.index)) + " words.")
print(" Processing Lancaster...")
# compare CUE/CDI
lancaster_cue_match_results = pd.merge(lancaster_sensorimotor_norms_frame,
                                       purecdi_dataframe,
                                       left_on='Word',
                                       right_on='word',
                                       how='outer',
                                       indicator=True)

lancaster_filtered_cue_matches = lancaster_cue_match_results[(lancaster_cue_match_results['_merge'] == 'both')]

lancaster_filtered_cue_matches.drop('_merge', axis=1, inplace=True)
col_dtypes = {
    'CUE': 'string',
    'TARGET': 'string',
    'WEIGHT': 'float',
    'REVSTR': 'string'
}

categories = [
    "gustatory", "auditory", "haptic", "olfactory",
    "foot_leg", "hand_arm", "head", "mouth",
    "torso", "interoceptive", "visual"
]

dfs = {category: pd.DataFrame(columns=col_dtypes) for category in categories}

def generate_connections(category, sensory_type, mean_type):
    threshold_key = f'lancaster_{category}'
    threshold = float(config['edges_threshold'][threshold_key])
    file_name = f'lancaster_{category}_edges.csv'

    return ev.generate_connection_weights(
        sensory_type, 'Word', f'{mean_type}.mean', file_name,
        dfs[category], threshold, edges_files_folder,
        lancaster_filtered_cue_matches, purecdi_dataframe
    )

for category in categories:
    sensory_type = category.capitalize()
    print(generate_connections(category, sensory_type, sensory_type))

