import argparse
import os
import pandas as pd

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Combine all CSV files in a directory into one.")
    parser.add_argument("directory", type=str, help="Path to the directory containing CSV files.")
    parser.add_argument("output_file", type=str, help="Path to save the combined CSV file.")

    # Parse arguments
    args = parser.parse_args()

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(args.directory) if f.endswith('.csv')]

    # Initialize an empty list to store dataframes
    dataframes = []

    # Read each CSV file and append to the list
    for file in csv_files:
        file_path = os.path.join(args.directory, file)
        df = pd.read_csv(file_path)
        
        if 'TEXT' in df.columns:
            df['note_text'] = df['TEXT']
        
        dataframes.append(df[['doc_id', 'note_text', 'est_token_count', 'note_type', 'dataset_name']])

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Save the combined dataframe to the output file
    combined_df.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
