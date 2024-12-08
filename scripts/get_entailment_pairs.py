import argparse
import pandas as pd
from factehr.utils.make_entailment_csv import parse_and_format_entailment_output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse and format entailment outputs, then save to CSV.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("output_path", type=str, help="Path to save the output CSV file.")

    # Parse arguments
    args = parser.parse_args()

    # Read the JSONL file
    entailment_predictions_df = pd.read_json(args.input_path, lines=True)

    # Parse and format the entailment output
    entailment_predictions_parsed = parse_and_format_entailment_output(entailment_predictions_df)

    # Select the required columns
    out_df = entailment_predictions_parsed[[
        "doc_id", "dataset_name", "note_type", "prompt", "index",
        "entailment_type", "model_name", 'model_output', 'json_parsed', 
        'entailment_pred_raw', 'not_parseable','entailment_pred'
    ]]

    # Save the output to a CSV file
    out_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
