import pandas as pd
import json
import os
from ipdb import set_trace
import gc 

def calculate_average_entailment_score(df):
    """
    Calculate the average entailment precision/recall score for each ID.
    Args:
        df (pd.DataFrame): DataFrame containing the entailment data.
    Returns:
        pd.DataFrame: DataFrame with average entailment scores for each ID.

    Edit; look for hypothesis which don't make sense and drop those rows
    """

    df = df.drop(df[df.hypothesis == ':'].index)
    df = df.drop(df[df.hypothesis == ','].index)
    df = df.drop(df[df.hypothesis == '.'].index)
    df = df.drop(df[df.hypothesis == ' '].index)
    # anything else

    grouped = df.groupby('ID').agg(
            total_sentences=('hypothesis', 'count'),
            count_ones=('entailment_pred', lambda x: (x == 1).sum()),
            count_zeros=('entailment_pred', lambda x: (x == 0).sum()),
            entailment_pred=('entailment_pred', lambda x: x.mean())
        ).reset_index()
    
    return grouped


def load_entailment_data(file_path):
    """
    Load entailment data from a specified JSON file.

    Args:
        file_path (str): Path to the JSON file containing entailment data.

    Returns:
        pd.DataFrame: DataFrame containing the entailment data.
    """
    data = []
    try: 
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
    
    except:
        print("Deleted line : ", line)
        pass

    return pd.DataFrame(data)


def process_entailment_results(out_path, prompt):
    """
    Code to calcuate wiyh 
    """
    precision_file = os.path.join(out_path, f"{prompt}_sentence_precision.json")
    recall_file = os.path.join(out_path, f"{prompt}_sentence_recall.json")
    if not os.path.exists(precision_file):
        raise FileNotFoundError("Precision file not found in the specified path.")
    
    if not os.path.exists(recall_file):
        raise FileNotFoundError("Precision file not found in the specified path.")
    
    precision_df = load_entailment_data(precision_file)
    recall_df = load_entailment_data(recall_file)
    avg_precision_df = calculate_average_entailment_score(precision_df)
    avg_recall_df = calculate_average_entailment_score(recall_df)
    # stratify the 
    
    result_df = pd.merge(avg_precision_df, avg_recall_df, on='ID', suffixes=('_precision', '_recall'))
    result_df.rename(columns={'entailment_pred_precision': 'precision', 'entailment_pred_recall': 'recall'}, inplace=True)
    result_df['f1'] = 2 * (result_df['precision'] * result_df['recall']) / (result_df['precision'] + result_df['recall'])
    return result_df


def process_all_models(root_path):
    """
    Process all model folders in the root path to calculate and save entailment results.

    Args:
        root_path (str): Root directory containing subfolders for each model.
    """
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
                if not "discharge_summary" in note_path:    
                    for filename in os.listdir(note_path):
                        # note_path = os.path.join(model_path, note_folder)
                        if filename.endswith('.json'):
                            prompt = filename.split('_')[0] 
                            if "ICL" in filename:
                                prompt = prompt + "_ICL"
                        if os.path.isdir(note_path):
                            try:
                                print("Processing " + prompt)
                                gc.collect()
                                output_csv = os.path.join(note_path, f"{prompt}_entailment.csv")
                                if os.path.exists(output_csv):
                                    print(f"Results already processed and saved as csv {output_csv}")
    
                                else:
                                    result_df = process_entailment_results(note_path, prompt)
                                    result_df.to_csv(output_csv, index=False)
                                    print(f"Saved results to {output_csv}")
                                    gc.collect()
                            except FileNotFoundError as e:
                                print(f"Error processing {model_folder}: {e}")
    return None




def load_csvs_to_dict(directory):
    # Initialize an empty dictionary to store the DataFrames
    csv_dict = {}
    # Walk through the directory and all subdirectories
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file is a CSV
            if filename.endswith('.csv'):
                # Construct the full file path
                file_path = os.path.join(root, filename)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
                # Use the file name (without extension) as the dictionary key
                key = root + "/" + os.path.splitext(filename)[0]
                # Store the DataFrame in the dictionary
                csv_dict[key] = df
    return csv_dict

    
def extract_info_from_path(file_path):
    parts = file_path.split('/')
    model = parts[-4]
    dataset = parts[-3]
    note = parts[-2]
    prompt = parts[-1]
    return model, dataset, note, prompt

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



def main():

    in_path = "/share/pi/nigam/rag-data/entailment_final/"
    out_path = "/share/pi/nigam/users/monreddy/rag-the-facts/result_fp_fn_without_ds.csv"

    process_all_models(in_path)

    csv_dict = load_csvs_to_dict(in_path)
    # Print the keys and shapes of the DataFrames
    for key, df in csv_dict.items():
        print(f"File: {key}, Shape: {df.shape}")
        
    combined_df = add_columns_and_combine(csv_dict)
    set_trace()
    combined_df.to_csv(out_path)


if __name__ == "__main__":
    main()
