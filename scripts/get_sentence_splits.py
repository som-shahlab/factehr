import pandas as pd
import json
import os
import argparse
from datetime import datetime

def process_entailment_files(input_path, output_path):
    """
    Process JSONL entailment files from the input directory, extract relevant fields,
    and save the output DataFrames as a CSV file.

    Parameters:
    - input_path (str): Path to the directory containing JSONL files.
    - output_path (str): Path to save the resulting CSV file.
    
    Outputs two csvs:
    - "recall_hypotheses_{mmddyy}.csv", which contains the notes broken down into sentences
    - "precision_hypotheses_{mmddyy}.csv", which contains the fact decompositions broken down into sentences
    """
    # Initialize an empty list to store DataFrames
    dataframes = []

    # Loop through all files in the directory
    for file_name in os.listdir(input_path):
        if file_name.endswith(".jsonl"):
            file_path = os.path.join(input_path, file_name)
            # Read each jsonl file into a DataFrame and append to the list
            with open(file_path, 'r') as file:
                data = [json.loads(line) for line in file]
                df = pd.DataFrame(data)
                dataframes.append(df)

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(dataframes, ignore_index=True)

    # Extract relevant fields
    final_df['custom_id'] = final_df['metadata'].apply(lambda x: x['custom_id'])
    final_df[["doc_id", "dataset", "note_type", "prompt", 'index', 'entailment_type']] = final_df['custom_id'].apply(lambda cid: pd.Series(cid.split("|")))
    final_df['hypothesis'] = final_df['metadata'].apply(lambda x: x['hypothesis'])
    final_df['model'] = final_df['metadata'].apply(lambda x: x['model_name'])

    recall_df = final_df.loc[final_df['entailment_type'] == "recall"][["doc_id", "dataset", "note_type", "prompt", 'index', 'entailment_type', 'hypothesis']].drop_duplicates().reset_index(drop=True)
    precision_df = final_df.loc[final_df['entailment_type'] == "precision"][["doc_id", "dataset", "note_type", "prompt", 'index', 'entailment_type', "model", 'hypothesis']].drop_duplicates().reset_index(drop=True)

    mmddyy = datetime.now().strftime("%m%d%y")
    recall_df.to_csv(os.path.join(output_path, f"recall_hypotheses_{mmddyy}.csv"), index=False)
    precision_df.to_csv(os.path.join(output_path, f"precision_hypotheses_{mmddyy}.csv"), index=False)
    
    print(f"Data has been written to {output_path}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Process JSONL entailment files and save as CSV.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing JSONL files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory to save the resulting CSV files")
    
    args = parser.parse_args()

    # Execute the function with command line arguments
    process_entailment_files(args.input_path, args.output_path)
