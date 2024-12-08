import os
import json
import argparse
import pandas as pd
import re

def read_json_lines(file_path):
        """
    Reads a file containing JSON objects, one per line, and returns a list of parsed JSON objects.

    Args:
        file_path (str): The path to the file containing JSON lines. Each line in the file
                         should be a valid JSON object.

    Returns:
        list: A list of dictionaries (or lists, depending on the structure of the JSON objects) 
              parsed from each line in the file.
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def gather_json_files(root_dir):
        """
    Recursively collects and organizes JSON files from a directory structure, returning
    a dictionary of parsed JSON contents.

    The directory structure is expected to have a hierarchy of:
        root_dir / model_name / dataset_name / note_name / [JSON files].
    
    Each JSON file name should follow a specific naming convention, allowing the
    function to extract 'prompt' information from the file name.

    Args:
        root_dir (str): The root directory containing subdirectories structured by
                        model names, dataset names, and note names, with JSON files inside.

    Returns:
        dict: A dictionary where the keys are strings of the form
              'model_name+dataset_name+note_name+prompt', and the values are dictionaries
              mapping different components of the filenames (like timestamps or other identifiers) 
              to parsed JSON objects.

              For example:
              {
                  'modelA+dataset1+noteX+prompt1_ICL+precision': {
                      '12345': [{...}, {...}],  # JSON contents parsed from file 'file_12345.json'
                      '67890': [{...}],         # JSON contents parsed from file 'file_67890.json'
                  },
                  ...
              }
    
    Notes:
        - This function prints the paths and filenames it processes for debugging purposes.
        - Files that cannot be processed due to errors (e.g., invalid JSON) are skipped.

    """
    json_files = {}
    for model_name in os.listdir(root_dir):
        model_path = os.path.join(root_dir, model_name)
        if os.path.isdir(model_path):
            for dataset_name in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset_name)
                if os.path.isdir(dataset_path):
                    for note_name in os.listdir(dataset_path):
                        note_path = os.path.join(dataset_path, note_name)
                        print(note_path)
                        if os.path.isdir(note_path):
                            for filename in os.listdir(note_path):
                                print(filename)
                                if filename.endswith('.json'):
                                    prompt = filename.split('_')[0]
                                    if "ICL" in filename:
                                        prompt = prompt + "_ICL"
                                    if "recall" in filename:
                                        prompt = prompt + "+recall"
                                    else:
                                        prompt = prompt + "+precision"
                                    file_path = os.path.join(note_path, filename)
                                    print(file_path)
                                    # with open(file_path, 'r') as f:
                                    #     file_contents = json.load(f)
                                    key = model_name + "+" + dataset_name + "+" + note_name + "+" + prompt
                                    if key not in json_files:
                                        json_files[key] = {}
                                    try:
                                        json_files[key][filename.split('_')[1].split('.')[0]] = read_json_lines(file_path)
                                    except:
                                        print("***SKIP***" + file_path)
                                        continue
    return json_files

def combine_json_dict(json_dict):
    """
    Combines a nested dictionary of JSON-like objects into a single list of dictionaries,
    adding the original key as a new field in each dictionary.

    Args:
        json_dict (dict): A dictionary where keys are strings, and values are dictionaries
                          mapping another string ('json_type') to a list of JSON objects (dictionaries).
    
    Returns:
        list: A list of dictionaries, where each dictionary contains the original 'key' 
              and the contents of the JSON object. The 'key' is added as a field in each 
              individual JSON object.
    """
    combined_json = []
    for key, value in json_dict.items():
        for json_type, json_list in value.items():
            for item in json_list:
                combined_json.append({"key": key, **item})
    return combined_json

def get_generated_facts(key_str, id):
      """
    Retrieves a model-generated fact set from a JSON file based on the given key string and ID.

    The function constructs the file path from a structured key string in the format 
    'model+dataset+note+prompt', and loads the corresponding JSON file to extract the 
    fact with the specified ID.

    Args:
        key_str (str): A string in the format 'model+dataset+note+prompt', which is used
                       to construct the path to the corresponding JSON file.
        id (str): The identifier of the fact to retrieve from the JSON data.

    Returns:
        dict: The fact corresponding to the given ID from the loaded JSON file.
    """
    key_str_split = key_str.split("+")
    model = key_str_split[0]
    dataset = key_str_split[1]
    note = key_str_split[2]
    prompt = key_str_split[3]
    json_path = f"/share/pi/nigam/rag-data/results_final/{model}/{dataset}/{note}/{prompt}.json"
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data[id]

# Generate annotation task file
json_files = gather_json_files("/share/pi/nigam/rag-data/entailment_final/")

combined = combine_json_dict(json_files)
filtered = [d for d in combined if len(d) == 6] # only get the completed results

df = pd.DataFrame(filtered)

# Check for rows that contain a number in the hypothesis
df['contains_number'] = df['hypothesis'].apply(lambda x: bool(re.search(r'\d', x)))
df['contains_number'].mean() #0.34

sample_number_df = df[df['contains_number']].sample(n=1400, random_state=42) # 1400 to maintain the prevlance of 34% hypotheses containing numbers
sample_number_df['row_number'] = range(1, len(sample_number_df) + 1)
sample_number_df['premise'] = sample_number_df['premise'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
cols = ['row_number'] + [col for col in sample_number_df.columns if col != 'row_number']
sample_number_df = sample_number_df[cols]
sample_number_df['hypothesis'] = sample_number_df['hypothesis'].str.strip()
sample_number_df['premise'] = sample_number_df['premise'].str.strip()

full_annotation_df = sample_number_df
annotator_df = sample_number_df[["row_number", "hypothesis", "premise"]]

full_annotation_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/full_annotation_task_numbers.csv")
annotator_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/annotation_task_numbers.csv")

###### Sample 20 notes for Jenelle #########
df['model'] = df.apply(lambda row: row['key'].split('+')[0], axis = 1)
df['note_type'] = df.apply(lambda row: row['key'].split('+')[2], axis = 1)
df['note_cat'] = df.apply(lambda row: "progress_note" if row['note_type'] in ['pdac', 'breastca'] else row['note_type'], axis = 1)

sampled_df = df.groupby(['model', 'note_cat'], group_keys=False).apply(lambda x: x.sample(n=1, random_state=42))
facts = sampled_df.apply(lambda row: get_generated_facts(row['key'], row['ID']), axis = 1)

records = []
for key, value in facts.items():
    for item in value:
        item['id'] = key  # Add the key as an identifier in each record
        records.append(item)

# Create the DataFrame
facts_df = pd.DataFrame(records)
facts_df['key'] = sampled_df['key'].values
facts_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/sentence_annotation_task2.csv")

####### SAMPLE 2 NOTES FROM EACH NOTE TYPE FOR JENELLE
nursing_df = df[df['key'].str.contains('nursing_note')].sample(n=1, random_state=42)
discharge_df = df[df['key'].str.contains('discharge')].sample(n=1, random_state=42)
progress_df = df[df['key'].str.contains('progress|coral')].sample(n=2, random_state=42)
sentence_annotation_df = pd.concat([nursing_df, discharge_df, progress_df])
facts = sentence_annotation_df.apply(lambda row: get_generated_facts(row['key'], row['ID']), axis = 1)

records = []
for key, value in facts.items():
    for item in value:
        item['id'] = key  # Add the key as an identifier in each record
        records.append(item)

# Create the DataFrame
facts_df = pd.DataFrame(records)
facts_df['key'] = sentence_annotation_df['key'].values
facts_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/sentence_annotation_task.csv")

####### SAMPLE GPT and RADIOLOGY PAIRS ######

orig_df = df
df = df[df['key'].str.contains('gpt|GPT|radiology')].sample(n=1500, random_state=42)
df['row_number'] = range(1, len(df) + 1)
df['premise'] = df['premise'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
cols = ['row_number'] + [col for col in df.columns if col != 'row_number']
df = df[cols]
df['hypothesis'] = df['hypothesis'].str.strip()
df['premise'] = df['premise'].str.strip()

full_annotation_df = df
annotator_df = df[["row_number", "hypothesis", "premise"]]

full_annotation_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/full_annotation_task_gpt_radiology.csv")
annotator_df.to_csv("/share/pi/nigam/rag-data/entailment_annotation/annotation_task_gpt_radiology.csv")


###  Just get the counts ####
split_df = df[['key']]
split_df = split_df['key'].str.split('+', expand=True)
split_df.columns = ['Model', 'Dataset', 'NoteType', 'Prompt', 'Metric']
combinations = split_df.apply(tuple, axis=1)
combination_counts = combinations.value_counts().reset_index()
combination_counts.columns = ['Combination', 'Count']
combination_counts.to_csv("/share/pi/nigam/rag-data/entailment_annotation/entailment_counts.csv")