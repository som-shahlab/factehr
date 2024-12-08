"""
Script to compute classification metrics from an input CSV and save the results to an output directory.

This script performs the following steps:
1. Parses a custom_id into its constituent parts.
2. Extracts JSON objects from the 'model_output' column.
3. Computes classification metrics (accuracy, precision, recall, F1-score) stratified by groups such as model, dataset, and prompt.
4. Saves the metrics as CSV files to the output directory.

Arguments:
1. input_csv: Path to the input CSV file containing entailment predictions and labels.
2. output_directory: Path to the directory where the output metrics CSV files will be saved.

Usage:
    python compute_entailment_stats.py /path/to/input.csv /path/to/output_directory/
"""

import pandas as pd
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import argparse


# Function to compute metrics
def compute_metrics(group):
    y_true = group['label']
    y_pred = group['entailment_pred']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return pd.Series({
        'n': len(y_true),
        'not_parseable': np.mean(group['not_parseable']),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
    

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compute classification metrics from input CSV and save to output directory.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file containing entailment predictions and labels.")
    parser.add_argument("output_directory", type=str, help="Directory where the output metrics CSV files will be saved.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    input_csv = args.input_csv
    output_directory = args.output_directory

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Compute metrics grouped by model, dataset, split, and prompt
    grouped = df.groupby(['model', 'dataset', 'split', 'prompt'])
    metrics_df = grouped.apply(compute_metrics).reset_index()
    print(metrics_df)
    
    # Save the metrics to a CSV file
    metrics_output_path = os.path.join(output_directory, "metrics_by_group.csv")
    metrics_df.to_csv(metrics_output_path, index=False)
    print(f"Metrics saved to {metrics_output_path}")

    # Compute metrics grouped by model and prompt
    grouped = df.groupby(['model', 'prompt'])
    metrics_df_prompt = grouped.apply(compute_metrics).reset_index()
    print(metrics_df_prompt)
    
    # Save the prompt-level metrics to a CSV file
    metrics_prompt_output_path = os.path.join(output_directory, "metrics_by_model_and_prompt.csv")
    metrics_df_prompt.to_csv(metrics_prompt_output_path, index=False)
    print(f"Metrics by model and prompt saved to {metrics_prompt_output_path}")

if __name__ == "__main__":
    main()
