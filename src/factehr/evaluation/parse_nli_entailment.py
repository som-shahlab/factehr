"""
Script to parse specific fields from multiple JSONL files in a directory, merge the parsed data, and save it to a CSV file.

This script extracts the following fields from each row in the JSONL files:
- "custom_id" from the "metadata" field
- "model_output" from the "choices[0].message.content" field
- "model" from the root level
- "label" from the "metadata" field

It also checks for any missing values (NaNs) in the columns and reports the count before saving the file.

Arguments:
1. input_directory: Directory containing JSONL files.
2. output_csv_path: Path to save the output CSV file.

Example usage:
    python parse_nli_entailment.py /path/to/jsonl/files /path/to/output.csv
"""

import os
import sys
import json
import pandas as pd
from glob import glob
import argparse

def parse_entailment(row):
    # Check if 'entailment_prediction' exists in 'json_parsed'
    if isinstance(row['json_parsed'], dict) and 'entailment_prediction' in row['json_parsed']:
        return row['json_parsed']['entailment_prediction']
    # Check if 'model_output' ends with a digit and return it as an integer
    elif isinstance(row['model_output'], str) and row['model_output'][-1].isdigit():
        return int(row['model_output'][-1])
    # Check if 'entailment' label is in output
    elif isinstance(row['model_output'], str) and 'entailment' in row['model_output'].lower():
        return 1
    elif isinstance(row['model_output'], str) and 'neutral' in row['model_output'].lower():
        return 0
    # Check if 'yes' label is in output
    elif isinstance(row['model_output'], str) and 'yes' in row['model_output'].lower():
        return 1
    elif isinstance(row['model_output'], str) and 'no' in row['model_output'].lower():
        return 0
    else:
        return None   

# Parse the custom_id column
def parse_and_assign(custom_id):
    parsed = parse_custom_id(custom_id)
    if len(parsed) == 4:
        return pd.Series(parsed, index=['dataset', 'split', 'uid', 'prompt'])
    elif len(parsed) == 6:
        return pd.Series(parsed, index=['doc_id', 'dataset_name', 'note_type', 'prompt', 'index', 'entailment_type'])
    else:
        return pd.Series([None] * 6)
        
def parse_message_from_choices(x):
    """Parses 'choices' field and returns the content from 'message'."""
    return x['choices'][0]['message']['content'] if 'choices' in x and isinstance(x['choices'], list) else None

def parse_error_string(error_list):
    """
    Parses a list that represents error messages.

    Args:
        error_list (list): List of stringified dictionaries.

    Returns:
        list: A string of concatenated error dicts.
    """
    return "//".join(error_list)

def parse_custom_id(custom_id):
    """
    Splits the custom_id into its constituent parts: doc_id, dataset_name, note_type, prompt, index, and entailment_type.
    If the split does not return exactly 4 parts, a fallback output is used.
    
    Returns:
        A tuple containing: doc_id, dataset_name, note_type, prompt, index, entailment_type.
    """
    try:
        if isinstance(custom_id, str):
            parts = custom_id.split('|')
            if len(parts) == 4:
                dataset, dataset_type, index, prompt = parts
                return dataset, dataset_type, int(index), prompt
            else:
                # if contains 5
                doc_id, dataset_name, note_type, prompt, index, entailment_type = custom_id.split('|')
                return doc_id, dataset_name, note_type, prompt, int(index), entailment_type
        else:
            return [None, None, None, None, None, None]  # Adjusted to return 6 None values
    except ValueError:
        # Handle any malformed custom_id values
        return None, None, None, None, None, None


def extract_json_from_string(string):
    """
    Extracts the first JSON object found within a given string.
    Args:
        string (str): The string to search for JSON objects.
    Returns:
        dict: The extracted JSON object if found, otherwise None.
    """
    if not isinstance(string, str):
        return None
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        try:
            start_idx = string.index('{')
            end_idx = string.rindex('}') + 1
            return json.loads(string[start_idx:end_idx])
        except (ValueError, json.JSONDecodeError):
            return None


def process_jsonl_file(jsonl_path):
    """Processes a single JSONL file and returns a DataFrame."""
    # Load the JSONL file into a DataFrame and assume the JSONL file structure has named columns
    df = pd.read_json(jsonl_path, lines=True)
    # Rename columns if they are not named (if applicable)
    # Assuming the JSONL file has three key sections: 'model', 'choices', 'metadata'
    # Adjust the column names according to the JSONL file's structure
    df.columns = ['messages', 'choices', 'metadata']
    # Extract relevant fields
    df['model_output'] = df['choices'].apply(lambda x: parse_message_from_choices(x) if isinstance(x, dict) else parse_error_string(x)) #TODO check this works for openai
    df['custom_id'] = df['metadata'].apply(lambda x: x['metadata']['custom_id'] if 'custom_id' in x['metadata'] else None)
    df['label'] = df['metadata'].apply(lambda x: x['metadata']['label'] if 'label' in x['metadata'] else None)
    df['model'] = df['messages'].apply(lambda x: x['model'] if 'model' in x else None)
    # Select the desired columns
    df_filtered = df[['custom_id', 'model', 'label', 'model_output']]
    return df_filtered


def process_all_jsonl_files(jsonl_directory):
    """Processes all JSONL files in a directory and combines the results into a single DataFrame."""
    # List all the JSONL files in the directory
    jsonl_files = glob(os.path.join(jsonl_directory, "*.jsonl"))
    # Check if no files are found
    if not jsonl_files:
        print(f"No JSONL files found in directory: {jsonl_directory}")
        sys.exit(1)
    # Initialize an empty list to hold DataFrames
    all_dfs = []
    # Loop through each file and process it
    for jsonl_file in jsonl_files:
        print(f"Processing file: {jsonl_file}")
        df_filtered = process_jsonl_file(jsonl_file)
        all_dfs.append(df_filtered)
    # Combine all DataFrames into one
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Parse specific fields from multiple JSONL files in a directory and save to a CSV.")
    parser.add_argument("input_directory", type=str, help="Directory containing JSONL files.")
    parser.add_argument("output_csv_path", type=str, help="Path to save the output CSV file.")

    # Parse the arguments
    args = parser.parse_args()

    jsonl_directory = args.input_directory
    output_csv_path = args.output_csv_path

    # Check if the provided path is valid
    if not os.path.isdir(jsonl_directory):
        print(f"Error: {jsonl_directory} is not a valid directory.")
        sys.exit(1)

    # Process all JSONL files in the directory
    df_combined = process_all_jsonl_files(jsonl_directory)
    
    # post processing and parsing

    # Apply the function to parse and assign columns based on the length of the parsed result
    parsed_custom_id = df_combined.apply(lambda row: parse_and_assign(row['custom_id']), axis=1)

    df_combined2 = df_combined.join(parsed_custom_id)
    
    # Remove duplicates by 'uid', keeping the last occurrence
    df_combined_unique = df_combined2.drop_duplicates(subset=['custom_id', "model"], keep='last')

    # Extract JSON data and parse entailment predictions
    # parse the json if there is json, otherwise take the last character of the output
    df_combined_unique['json_parsed'] = df_combined_unique['model_output'].apply(extract_json_from_string)
    df_combined_unique['entailment_pred_raw'] = df_combined_unique.apply(parse_entailment, axis=1)

    # Handle unparseable predictions and convert them to integers
    df_combined_unique['not_parseable'] = (~df_combined_unique['entailment_pred_raw'].isin([0, 1])) 
    df_combined_unique['entailment_pred'] = df_combined_unique.apply(lambda row: 0 if row['not_parseable'] else int(row['entailment_pred_raw']), axis=1) 
    df_combined_unique['label'] = df_combined_unique['label'].apply(int)
    df_combined_unique['entailment_pred'] = df_combined_unique['entailment_pred'].apply(int)

    # Show the first few rows of the combined DataFrame
    print(df_combined_unique.head())

    # Check for missing values and count NaNs
    nan_counts = df_combined_unique.isnull().sum()
    print(f"Missing values (NaNs) in columns:\n{nan_counts}")

    # Ensure the output directory exists
    output_directory = os.path.dirname(output_csv_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the DataFrame to the specified CSV file
    df_combined_unique.to_csv(output_csv_path, index=False)
    print(f"Parsed data saved to '{output_csv_path}'")

if __name__ == "__main__":
    main()
