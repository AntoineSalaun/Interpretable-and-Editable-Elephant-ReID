'''
File name: get_seek_dict.py

Description: Combines all the workflow SEEK annotations by elephant ID, 
             saving the number of times each value appears for each SEEK character for each elephant,
             as well as a CSV file of the final SEEK codes for each elephant as determined by heuristic
             rules described in the "HEURISTIC RULES FUNCTIONS" section below.

Author: Ana Uribe (uribe055@umn.edu)
        Updated July 17, 2024

Derived by Antoine Salaun on Nov 12, 2024 to produce image-specific SEEK codes
'''
########################################## IMPORTS ##########################################
import os
import json
import pandas as pd
import numpy as np

########################################## VARIABLES ##########################################
origin_dir_path = '/archive/vision/beery/animal_reid/datasets/elephants_zooniverse/ele_id_project'
export_dir_path = 'CBM_reid/data/Elephant4Africa'
constants_path = os.path.join(origin_dir_path, 'data-processing/workflow_constants_traditional.json')

workflows = ['general', 'ear', 'special']
########################################## LOAD IN DATA ##########################################
# dictionary with constants
j_file = open(constants_path)
constants_dict = json.load(j_file)
j_file.close()

code_dict = constants_dict['code']  # grabbing code dict from constants_dict

# get annotation data from all the workflows
for workflow in workflows:

    workflow_constants_dict = constants_dict[workflow]  # workflow data dict
    data_path = os.path.join(origin_dir_path, code_dict['annotations dir'], workflow_constants_dict['seek annotations file name'])

    if workflow == 'general':
        try:
            g_annotations = pd.read_csv(data_path)  #, index_col=code_dict['index col'])
            # Grab only the columns we want counts for (the SEEK characters)
            g_df = g_annotations[['subject_ids','R_tusk', 'L_tusk']]
        except:
            print(f'General features CSV "{data_path}" could not be uploaded.')
            exit()
    elif workflow == 'ear':
        try:
            e_annotations = pd.read_csv(data_path) #, index_col=code_dict['index col'])
            # Grab only the columns we want counts for (the SEEK characters)
            e_df = e_annotations[['subject_ids', 'R_tear_1', 'L_tear_1', 'R_tear_2', 'L_tear_2',
                        'R_hole_1', 'R_hole_2', 'L_hole_1', 'L_hole_2', 'R_extreme',
                        'L_extreme', 'R_extreme_only', 'L_extreme_only']]
        except:
            print(f'Ear features CSV "{data_path}" could not be uploaded.')
            exit()
    else:
        try:
            s_annotations = pd.read_csv(data_path, index_col=code_dict['index col'])
            # Grab only the columns we want counts for (the SEEK characters)
            s_df = s_annotations[['subject_ids','Special_ear', 'Special_body', 'Age', 'Sex']]
        except:
            print(f'Special features CSV "{data_path}" could not be uploaded.')
            exit()

# check all the 
if (g_df.__class__ == pd.DataFrame) & (e_df.__class__ == pd.DataFrame) & (s_df.__class__ == pd.DataFrame):
    print('Upload of all data files complete.')
else:
    print('Upload of one or more data files failed.')
    exit()


########################################## HEURISTIC RULES FUNCTIONS ##########################################
def get_val(p_counts, u, most_likely, other, cutoff=.8, fraction=1):
    '''
    Determines the value of the character as follows:
        If u has a percentage > cutoff (default = .8), return u
        Else if other has a higher percentage than most_likely times a fraction (default = 1), return other. 
        Else return most_likely
    '''
    # Check if 'u' is a key and its value is greater than cuttoff
    if u in p_counts and p_counts[u] > cutoff:
        return u
    
    # Check if both 'other' and 'most_likely' are keys
    if other in p_counts and most_likely in p_counts:
        if p_counts[other] > (p_counts[most_likely] * fraction):
            return other
        else:
            return most_likely
    
    # Check if only 'other' is a key
    if other in p_counts:
        return other
    
    # Check if only 'most_likely' is a key
    if most_likely in p_counts:
        return most_likely
    
    # Default return if none of the conditions are met
    print('None of the expected values were found in dictionary:', p_counts)
    return None

def get_sex_val(p):
    '''
    Determines the value of the character as follows:
        If the underscore has a percentage > 80%, return the underscore
        Else, return C if its percentage is twice that of B, and B otherwise
    '''
    return get_val(p_counts=p, u='_', most_likely='B', other='C', fraction=2)

def get_age_val(p):
    '''
    Determines the value of the character as follows:
        If the underscore has a percentage > 80%, return the underscore
        Else, return 20 if it has a higher percentage than 00, and 00 otherwise
    '''
    return get_val(p_counts=p, u='_', most_likely='00', other='20')

def get_tusk_val(p):
    '''
    Determines the value of the character as follows:
        If the underscore has a percentage > 80%, return the underscore
        Else, return 0 if it has a higher percentage than 1, and 1 otherwise
    '''
    return get_val(p_counts=p, u='_', most_likely='1', other='0')

def get_tear_hole_val(p, cutoff=.9, frac=2):
    '''
    Determines the value of the character as follows:
        If underscore has a percentage > cutoff (default .9), return underscore
        Else if the percentage for zero > a fraction of the sum of all other percentages (default = 2), return zero
        Else return the greatest percentage that is not zero or underscore
    '''
    u = '_'
    z = '0'

    # Check if 'u' is a key and its value is greater than cuttoff
    if u in p and p[u] > cutoff:
        return u
    
    # Calculate the sum of all other percentages except zero and underscore
    other_sum = sum(value for key, value in p.items() if key != z and key != u)
    
    # Compare the zero key with this sum
    if z in p and p[z] > (other_sum * frac):
        return z
    
    # Find and return the key with the highest percentage that is not zero or underscore
    max_key = None
    max_value = -1
    for key, value in p.items():
        if key != z and key != u and value > max_value:
            max_key = key
            max_value = value
    
    return max_key
    
def get_feature_val(p):
    '''
    Determines the value of the feature as follows:
        If 1 has a percentage > 80%, return 1
        Else, return underscore if it has a higher percentage than zero, and zero otherwise
    '''
    return get_val(p_counts=p, u='1', most_likely='0', other='_', fraction=1.5)

def add_to_dict(df):
    ''' 
    Adds SEEK count information in 'df' for each elephant ID to the dictionary 'seek_dict'
    '''
    for index, row in df.iterrows():
        #ele_id = row['ele_id']
        subject_id = row['subject_ids']
        #features = row.drop('ele_id')
        features = row.drop('subject_ids')
        
        if subject_id not in image_seek_dict:
            image_seek_dict[subject_id] = {}
        
        # Loop through each column and its value
        for feature, value in features.items():
            # if this feature has never been seen, create it in the dict
            if feature not in image_seek_dict[subject_id]:
                image_seek_dict[subject_id][feature] = {}
            
            # if this possible value of this feature has never been seen, create it in the dict
            if value not in image_seek_dict[subject_id][feature]:
                image_seek_dict[subject_id][feature][value] = 0
            
            image_seek_dict[subject_id][feature][value] += 1


# Before modification
def get_final_df(image_seek_dict):
    ''' 
    From final dictionary, gets maximum value of each character (feature) for each ele_id
    '''
    # Initialize an empty dictionary to store maximum values for each feature
    max_values_dict = {}

    # Iterate over each subject in the seek_dict dictionary
    for subject_id, features in image_seek_dict.items():
        max_values_dict[subject_id] = {}
        
        # Calculate total counts for each feature
        total_counts = {feature: sum(counts.values()) for feature, counts in features.items()}

        # Find the desired value for each feature and store it in the max_values_dict
        for feature, counts in features.items():
            # get percentage of counts for each value
            percentage_counts = {val: round(np.divide(count, total_counts[feature]), 3) for val, count in counts.items()}
            
            # check the dictionary for the empty and the singular case
            if not percentage_counts:
                return None
            else:
                if feature == 'Sex':
                    max_value = get_sex_val(percentage_counts)
                elif feature == 'Age':
                    max_value = get_age_val(percentage_counts)
                elif feature in ['R_tusk', 'L_tusk']:
                    max_value = get_tusk_val(percentage_counts)
                elif feature in ['R_tear_1', 'R_hole_1', 'L_tear_1', 'L_hole_1']:
                    max_value = get_tear_hole_val(percentage_counts, cutoff= .9, frac=2)
                elif feature in ['R_tear_2', 'R_hole_2', 'L_tear_2', 'L_hole_2']:
                    max_value = get_tear_hole_val(percentage_counts, cutoff=.95, frac=2.5)
                elif feature in ['R_extreme', 'L_extreme', 'Special_ear', 'Special_body']:
                    max_value = get_feature_val(percentage_counts)
                #problem when the feature is ele_id
            
            max_values_dict[subject_id][feature] = max_value
    
    # Create DataFrames from dictionaries
    max_values_df = pd.DataFrame(max_values_dict).T

    # Rename the index to
    max_values_df.index.name = 'subject_id'

    # return result_df
    return max_values_df

def get_final_df_max_vals(seek_dict, exclude_u = True):
    ''' 
    From final dictionary, gets maximum value of each feature for each subject_id, 
    either including or excluding underscores (default = excluding)
    '''
    # Initialize an empty dictionary to store maximum values for each feature
    max_values_dict = {}

    # Iterate over each subject in the seek_dict dictionary
    for subject_id, features in seek_dict.items():
        max_values_dict[subject_id] = {}

        if exclude_u:
            # Grab the maximum value EXCLUDING underscores
            # Find the maximum value for each feature and store it in the max_values_dict
            for feature, counts in features.items():
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                max_value = next((value for value, count in sorted_counts if value != "_"), None)
                if max_value is None:
                    max_value = sorted_counts[0][0]  # If all values are underscores, take the first one
                max_values_dict[subject_id][feature] = max_value
        else:
            # Grab the maximum value INCLUDING underscores
            # Find the maximum value for each feature and store it in the max_values_dict
            for feature, counts in features.items():
                max_value = max(counts, key=counts.get)
                max_values_dict[subject_id][feature] = max_value

    # Create DataFrames from dictionaries
    max_values_df = pd.DataFrame(max_values_dict).T

    # Rename the index to
    max_values_df.index.name = 'subject_id'

    # return result_df
    return max_values_df

def get_id_ele_ids(df):
    '''
    Make a list of identified ele_ids. Currently, I can just filter by the ele_id names:
        if the ele_id starts with a 'B' or 'F', it is an identified elephant
        else, it is not (and starts with letter 'T')
    '''
    idi_ele_ids = []
    for i in df.index:
        if 'B' in i:
            idi_ele_ids.append(i)
        elif 'F' in i:
            idi_ele_ids.append(i)
        else:
            pass
    
    return idi_ele_ids


# Initialize an empty dictionary to store the SEEK character counts
image_seek_dict = {}

# Add information from each workflow df to the seek_dict
add_to_dict(g_df)
add_to_dict(s_df)
add_to_dict(e_df)

# Create JSON file
image_seek_dict_path = os.path.join( code_dict['annotations dir'], code_dict['SEEK dictionary file name'])
print(image_seek_dict_path)
os.makedirs(os.path.dirname(image_seek_dict_path), exist_ok=True)

# Save the final dictionary as a JSON file
with open(image_seek_dict_path, 'w') as f: json.dump(image_seek_dict, f, indent=4)
print(f"Result dictionary saved as JSON file: {image_seek_dict_path}")


# Get a pandas df with the final results
image_seek_df = get_final_df(image_seek_dict)

# concatonate values to get a SEEK code for each elephant ID
SP = {'1': 'Sex', 
		'2': 'Age',
		'4': 'R_tusk',
		'5': 'L_tusk',
		'6': 'R_tear_1',
		'7': 'R_hole_1',
		'8': 'R_tear_2',
		'9': 'R_hole_2',
		'10': 'L_tear_1',
		'11': 'L_hole_1',
		'12': 'L_tear_2',
		'13': 'L_hole_2',
		'14': 'R_extreme',
		'15': 'L_extreme',
		'16': 'Special_ear',
		'18': 'Special_body'}

for idx in image_seek_df.index:

    row = image_seek_df.loc[idx]
    # row_val = str(row[SP['1']])
    gen_p = 'T' + str(row[SP['4']]) + str(row[SP['5']])

    sp_p_1 = str(row[SP['1']]) + str(row[SP['2']])
    sp_p_2 = 'S' + str(row[SP['16']]) + str(row[SP['18']])
    ear_p = 'E' + str(row[SP['6']]) + str(row[SP['7']]) + str(row[SP['8']]) + str(row[SP['9']]) + '-' + str(row[SP['10']]) + str(row[SP['11']]) + str(row[SP['12']]) + str(row[SP['13']]) + 'X' + str(row[SP['14']]) + str(row[SP['15']])

    full_seek = sp_p_1 + gen_p + ear_p + sp_p_2

    image_seek_df.loc[idx, ('SEEK')] = full_seek

# Load the sub_ele_pairs.json file
sub_ele_pairs_path = os.path.join(origin_dir_path, code_dict['annotations dir'], 'sub_ele_pairs.json')
with open(sub_ele_pairs_path, 'r') as f:
	sub_ele_pairs = json.load(f)

# Debugging: Print the sub_ele_pairs dictionary and the index of image_seek_df
print("sub_ele_pairs:", sub_ele_pairs)

print('for index ', 92100839, ' I fin the ele_id ', sub_ele_pairs['92100839'])
image_seek_df['ele_id'] = [sub_ele_pairs.get(str(x), None) for x in image_seek_df.index.tolist()]

# save seek_df to a csv file
image_seek_df.to_csv(os.path.join(code_dict['annotations dir'], code_dict['SEEK code file name']))
