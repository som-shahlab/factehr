import pandas as pd
import json
import os
import argparse
from factehr.utils import hash_text

def concatenate_jsonl_files(input_path, output_path):
    """
    Concatenate all JSONL files in the specified input directory into a single 
    pandas DataFrame, add a unique "uid" column, and save it to a CSV file.

    Parameters:
    - input_path (str): Path to the directory containing JSONL files.
    - output_path (str): Path to save the concatenated DataFrame as a CSV file.

    The function reads all .jsonl files from the input directory, extracts relevant 
    columns, adds a unique "uid" column for each row, and saves the resulting DataFrame 
    as a CSV at the specified output path.
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

    # Adjust columns and extract relevant information
    final_df.columns = ["messages", "choices", "metadata"]
    final_df['model'] = final_df['messages'].apply(lambda x: x['model'])
    final_df['note_uid'] = final_df['metadata'].apply(lambda x: x['metadata']['uid'])
    final_df['has_content'] = final_df['choices'].apply(lambda x: 'content' in x['choices'][0]['message'] if 'choices' in x else False)
    final_df['fact_decomp'] = final_df.apply(lambda row: row['choices']['choices'][0]['message']['content'] if row['has_content'] else None, axis=1)

    filtered_df = final_df.loc[final_df['fact_decomp'].notna()].copy()
    
    # Add a unique 'uid' column for each row
    filtered_df['decomp_id'] = filtered_df["fact_decomp"].apply(hash_text)
    filtered_df['uid'] = filtered_df.apply(lambda row: row['decomp_id'] + "|" + row['model'] + "|" + row['note_uid'], axis = 1)

    # Select the desired columns
    out_df = filtered_df[['uid', 'decomp_id', 'model', 'note_uid', 'fact_decomp']]

    # Save the output DataFrame to a CSV file
    out_df.to_csv(output_path, index=False)
    print(f"Data has been written to {output_path}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Concatenate JSONL files into a single CSV with a 'uid' column.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the directory containing JSONL files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the concatenated CSV file.")
    
    args = parser.parse_args()

    # Execute the function with command line arguments
    concatenate_jsonl_files(args.input_path, args.output_path)
