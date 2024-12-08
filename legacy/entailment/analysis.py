import pandas as pd
import json
import os
from ipdb import set_trace
import gc 



def calculate_numfacts(sent_json):
    """
    Calculate the average entailment precision/recall score for each ID.
    Args:
        df (pd.DataFrame): DataFrame containing the entailment data.
    Returns:
        pd.DataFrame: DataFrame with average entailment scores for each ID.

    Edit; look for hypothesis which don't make sense and drop those rows
    """
    df = pd.DataFrame()
    data = []
    for item in sent_json:

        

        for each_fact in item['text']:
            count = 0 
            if each_fact['hypothesis'] == ':' or each_fact['hypothesis'] == ',' or each_fact['hypothesis'] == '.' or each_fact['hypothesis'] == ' ': 
               count += 1 

        if count > 0:
            print(count)

        number = len(item['text']) - count
        data.append({'ID': item['ID'], 'num_facts': number})

        df = pd.DataFrame(data)
    
    return df



def load_sents(file_path):
    """
    Load entailment data from a specified JSON file.

    Args:
        file_path (str): Path to the JSON file containing entailment data.

    Returns:
        JSON file loads the sentences.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data
                
                  

def process_num_facts(out_path, prompt):
    """
    Code to calcuate wiyh 
    """
    precision_file = os.path.join(out_path, f"{prompt}_sentence_precision.json")
    recall_file = os.path.join(out_path, f"{prompt}_sentence_recall.json")
    if not os.path.exists(precision_file):
        #raise FileNotFoundError("Precision file not found in the specified path.")
        pass
    if not os.path.exists(recall_file):
        #raise FileNotFoundError("Precision file not found in the specified path.")
        pass
        
    sent_json_prec = load_sents(precision_file)
    result_df = calculate_numfacts(sent_json_prec)

    return result_df


def extract_info_from_path(file_path):
    parts = file_path.split('/')
    model = parts[-3]
    dataset = parts[-2]
    note = parts[-1]

    return model, dataset, note

def add_columns_and_combine(csv_dict):
    combined_df = pd.DataFrame()

    for file_path, df in csv_dict.items():
        model, dataset, note_type, prompt = extract_info_from_path(file_path)
        df['model'] = model
        df['dataset'] = dataset
        df['note_type'] = note_type
        df['prompt'] = prompt
        combined_df = pd.concat([combined_df, df], axis=0)
    return combined_df

def process_all_models(root_path):
    """
    Process all model folders in the root path to calculate and save entailment results.

    Args:
        root_path (str): Root directory containing subfolders for each model.
    """

    combined_df = pd.DataFrame() 

    for model_folder in os.listdir(root_path):
        model_path = os.path.join(root_path, model_folder)
        if not os.path.isdir(model_path):  # Skip if not a directory
            continue
        print(f"Processing model folder: {model_folder}")
        for dataset_folder in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_folder)
            print(f"Processing dataset folder: {dataset_path}")
            for note_folder in os.listdir(dataset_path):
                note_path = os.path.join(dataset_path, note_folder)
                print(f"Processing note type folder: {note_path}")
                for filename in os.listdir(note_path):
                    # note_path = os.path.join(model_path, note_folder)
                    if filename.endswith('.json') and "recall" in filename:
                        prompt = filename.split('_')[0] 
                        if "ICL" in filename:
                            prompt = prompt + "_ICL"
                        if os.path.isdir(note_path):
                            try:
                                print("Processing " + prompt)
                                gc.collect()
                                output_csv = os.path.join(note_path, f"{prompt}_analysis.csv")
                                # if os.path.exists(output_csv):
                                #     print(f"Results already processed and saved as csv {output_csv}")
    
                                # else:
                                result_df = process_num_facts(note_path, prompt)
                                model, dataset, note_type = extract_info_from_path(note_path)
                                result_df['model'] = model
                                result_df['dataset'] = dataset
                                result_df['note_type'] = note_type
                                result_df['prompt'] = prompt
                            
                                # don't save but append 
                                combined_df = pd.concat([combined_df, result_df], axis=0)
                                gc.collect()
                            except FileNotFoundError as e:
                                print(f"Error processing {model_folder}: {e}")
        
    return combined_df




def main():

    in_path = "/share/pi/nigam/rag-the-facts/datasets/sentences/"
    out_path = "/share/pi/nigam/users/monreddy/rag-the-facts/num_sents.csv"

    result_df = process_all_models(in_path)

    # use this to merge, mean or whatever

    set_trace()
    #combined_df.to_csv(out_path)


if __name__ == "__main__":
    main()
