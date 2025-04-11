import os
import sys
import json
import pandas as pd
from glob import glob
import argparse

from factehr.evaluation.parse_nli_entailment import *
from factehr.evaluation.entailment import *
import random

def parse_and_format_entailment_output(entailment_predictions_df):
    # Process the data
    entailment_predictions_df['uid'] = entailment_predictions_df[2].apply(lambda x: x['metadata']['metadata']['uid'])
    entailment_predictions_df.columns = ['messages', 'choices', 'metadata', 'uid']
    entailment_predictions_df['premise'] = entailment_predictions_df['metadata'].apply(lambda x: x['metadata']["premise"] if 'metadata' in x['metadata'] else None)
    entailment_predictions_df['hypothesis'] = entailment_predictions_df['metadata'].apply(lambda x: x['metadata']["hypothesis"] if 'metadata' in x['metadata'] else None)
    entailment_predictions_df['model_output'] = entailment_predictions_df['choices'].apply(lambda x: parse_message_from_choices(x) if isinstance(x, dict) else parse_error_string(x))
    entailment_predictions_df['json_parsed'] = entailment_predictions_df['model_output'].apply(extract_json_from_string)
    entailment_predictions_df['entailment_pred_raw'] = entailment_predictions_df.apply(parse_entailment, axis=1)
    entailment_predictions_df['not_parseable'] = (~entailment_predictions_df['entailment_pred_raw'].isin([0, 1])) 
    entailment_predictions_df['entailment_pred'] = entailment_predictions_df.apply(lambda row: 0 if row['not_parseable'] else int(row['entailment_pred_raw']), axis=1) 
    entailment_predictions_df['entailment_pred'] = entailment_predictions_df['entailment_pred'].apply(int)
    entailment_predictions_df['model_name'] = entailment_predictions_df['metadata'].apply(lambda x: x['metadata']["model_name"])

    # Parse custom ID
    parsed_custom_id = entailment_predictions_df.apply(lambda row: parse_and_assign(row['metadata']['metadata']["custom_id"]), axis=1)

    # Join parsed custom ID to the dataframe
    entailment_predictions_df = entailment_predictions_df.join(parsed_custom_id)
    
    return entailment_predictions_df

def main(input_path, output_path):
    # Load the entailment predictions JSONL file
    entailment_predictions_df = pd.read_json(input_path, lines=True)

    entailment_df_processed = parse_and_format_entailment_output(entailment_predictions_df)

    # Select relevant columns to save
    entailment_to_save = entailment_df_processed[["uid", "doc_id", "dataset_name", "note_type", "prompt", "index", "entailment_type", "model_name", "not_parseable", "model_output", 'entailment_pred']]

    # Save the results to the output path
    entailment_to_save.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process entailment predictions.")
    parser.add_argument("--input_path", required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_path", required=True, help="Path to save the output CSV file.")
    
    args = parser.parse_args()

    main(args.input_path, args.output_path)
