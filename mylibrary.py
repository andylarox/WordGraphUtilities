import sys
import time
import mysql.connector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import time
import mysql.connector
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def replace_from_dictionary(input_frame, replace_frame, firstcol, secondcol='NONE'):
    # search and replace word labels based on dictionary file
    dict_lookup = dict(zip(replace_frame['find'], replace_frame['replace']))
    cue_mask = input_frame[firstcol].isin(dict_lookup.keys())
    input_frame.loc[cue_mask, firstcol] = input_frame.loc[cue_mask, firstcol].map(dict_lookup)

    if secondcol != 'NONE':
        target_mask = input_frame[secondcol].isin(dict_lookup.keys())
        input_frame.loc[target_mask, secondcol] = input_frame.loc[target_mask, secondcol].map(dict_lookup)

    return input_frame


def limit_and_normalise(input_frame, limit, weight_column):
    # sort data
    input_frame = input_frame.sort_values(weight_column, ascending=False)

    # reduce to top n (limit)
    input_frame = input_frame.head(limit)

    # normalise weights
    scaler = MinMaxScaler()
    input_frame[[weight_column]] = scaler.fit_transform(input_frame[[weight_column]])

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
    return our_dataframe


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
    # convert to capitals
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
    return our_dataframe


def get_wordbank_wordlist_from_mysql(credentials, targettable):
    # get full list of words being used in the observational data
    # we use this to limit the graphs

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
    return wordsframe


def calculate_edge_weight_average(input_frame, weightcol):
    total_edge_weight = 0.0
    count = 0
    for index, row in input_frame.iterrows():
        count = count + 1
        total_edge_weight = total_edge_weight + row[weightcol]
        average_edge_weight = total_edge_weight / count
    return average_edge_weight


def add_missing_loops(active_frame, purecdi_dataframe, source, targ, wt, framename):
    # compare CUE/CDI
    active_frame_cue_match_results = pd.merge(active_frame, purecdi_dataframe, left_on=source, right_on='word',
                                              how='outer', indicator=True)
    # get cue matches
    active_frame_filtered_cue_matches = active_frame_cue_match_results[
        (active_frame_cue_match_results['_merge'] == 'both')]
    # drop merge col
    active_frame_filtered_cue_matches.drop('_merge', axis=1, inplace=True)

    # compare target of matched cue rows with cdi
    active_frame_cue_target_match_results = pd.merge(active_frame_filtered_cue_matches, purecdi_dataframe,
                                                     left_on=targ,
                                                     right_on='word', how='outer', indicator=True)
    # get cue and target matches
    active_frame_filtered_cue_targ_matches = active_frame_cue_target_match_results[
        (active_frame_cue_target_match_results['_merge'] == 'both')]
    # drop merge col
    active_frame_filtered_cue_targ_matches.drop(['_merge'], axis=1, inplace=True)
    # print summary
    print("active_frame dataset can generate " + str(len(active_frame_filtered_cue_targ_matches.index)) + " edges.")

    # create self-loops frame
    # get cue results that do not match cdi (i.e. cdi only)
    unconnected_cue_active_frame = active_frame_cue_match_results[(active_frame_cue_match_results['_merge'] == 'right_only')]
    unconnected_cue_active_frame[source] = unconnected_cue_active_frame['word']
    unconnected_cue_active_frame[targ] = unconnected_cue_active_frame['word']
    unconnected_cue_active_frame[wt] = 0

    # drop merge cols
    unconnected_cue_active_frame.drop(['_merge', 'word'], axis=1, inplace=True)

    # now check targets
    # get target results that do not match cdi (i.e. cdi only)
    unconnected_target_active_frame = active_frame_cue_target_match_results[
        (active_frame_cue_target_match_results['_merge'] == 'right_only')]

    unconnected_target_active_frame[source] = unconnected_target_active_frame['word_y']
    unconnected_target_active_frame[targ] = unconnected_target_active_frame['word_y']
    unconnected_target_active_frame[wt] = 0
    unconnected_target_active_frame.drop(['_merge', 'word_x', 'word_y'], axis=1, inplace=True)

    # merge
    unconnected_active_frame = unconnected_cue_active_frame.append(unconnected_target_active_frame)
    combined_frame = active_frame_filtered_cue_targ_matches.append(unconnected_active_frame)
    active_frame_edges_output_frame = combined_frame[[source, targ, wt]].copy(deep=True)
    active_frame_edges_output_frame.rename(columns={source: 'source', targ: 'target', wt: 'weight'}, inplace=True)

    print(framename + " dataset will have " + str(len(unconnected_active_frame.index)) + " unconnected nodes.")

    return active_frame_edges_output_frame


def calculate_weight(cue, tgt, max, min):
    cue_scaled = (cue - min) / (max - min)
    tgt_scaled = (tgt - min) / (max - min)

    weight = cue_scaled * tgt_scaled

    return round(weight, 7)


def generate_connection_weights(data_name, wrdcolname, column_name, output_filename, frame, threshold, edges_files_folder, filtered_cue_matches_frame, purecdi_dataframe):
    # TODO: move these to the passed parameters
    verbose = False
    limited = True
    limit_value = 2000
    if limited:
        threshold = 0.3

    lancaster_column = column_name
    print(data_name + "...")
    max_weight = filtered_cue_matches_frame[lancaster_column].max()
    min_weight = filtered_cue_matches_frame[lancaster_column].min()
    start_time = time.time()
    x = 0
    average_edge_weight = 0.0
    total_edge_weight = 0.0
    beforetotal = 0
    for index, row in filtered_cue_matches_frame.iterrows():
        x = x + 1
        if verbose:
            print(row[wrdcolname] + "--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()

        # progress bar
        j = (x + 1) / (len(filtered_cue_matches_frame))
        sys.stdout.write('\r')
        sys.stdout.write("[%-50s] %d%% " % ('=' * int(50 * j), 100 * j))
        sys.stdout.flush()

        for index2, row2 in filtered_cue_matches_frame.iterrows():
            cue_weight = float(row[lancaster_column])
            tgt_weight = float(row2[lancaster_column])
            new_weight = calculate_weight(cue_weight, tgt_weight, max_weight, min_weight)

            # if both words the same, connection is 1.0
            if row[wrdcolname] == row2[wrdcolname]:
                if new_weight > threshold:
                    new_weight = 1.0
                else:
                    new_weight = 0
            else:
                new_weight = calculate_weight(cue_weight, tgt_weight, max_weight, min_weight)

            beforetotal = beforetotal + 1
            total_edge_weight = total_edge_weight + new_weight
            average_edge_weight = total_edge_weight / beforetotal
            if new_weight > threshold:
                checkword = row2[wrdcolname] + " " + row[wrdcolname]
                revstr = row[wrdcolname] + " " + row2[wrdcolname]
                found = frame[frame['REVSTR'].str.contains(checkword)]

                if found.empty:
                    found2 = frame[frame['REVSTR'].str.contains(revstr)]
                    if found2.empty:
                        new_row = {'CUE': row[wrdcolname], 'TARGET': row2[wrdcolname], 'WEIGHT': new_weight, 'REVSTR': checkword}
                        frame = frame.append(new_row, ignore_index=True)

    frame.drop('REVSTR', axis=1, inplace=True)
    if limited == True:
        frame = limit_and_normalise(frame, int(limit_value), 'WEIGHT')

    out_frame = add_missing_loops(frame, purecdi_dataframe, 'CUE', 'TARGET', 'WEIGHT', data_name)
    beforetotal = beforetotal + len(out_frame)

    output_filename = edges_files_folder + output_filename
    out_frame.to_csv(output_filename, index=False)

    return 'Generated ' + output_filename + ' @ ' + str(threshold) + ' threshold, total edges= ' + str(len(out_frame)) + ' from ' + str(beforetotal) + ' Average edge weight = ' + str(
        average_edge_weight)
