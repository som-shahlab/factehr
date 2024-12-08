import argparse
import pandas as pd
from factehr.utils.make_entailment_csv import parse_and_format_entailment_output

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse entailment outputs and save a sampled annotation file.")
    parser.add_argument("input_path", type=str, help="Path to the input JSONL file.")
    parser.add_argument("output_path", type=str, help="Path to save the output CSV file.")

    # Parse arguments
    args = parser.parse_args()
    
    if "jsonl" in args.input_path:
        # Load the JSONL file
        entailment_predictions_df = pd.read_json(args.input_path, lines=True)
        # Parse and format the entailment output
        entailment_predictions_parsed = parse_and_format_entailment_output(entailment_predictions_df)
        cols_to_pull = [
                            "doc_id", "dataset_name", "note_type", "prompt", "index",
                            "entailment_type", "model_name", "premise", "hypothesis"
                        ]
    elif "csv" in args.input_path:
        entailment_predictions_parsed = pd.read_csv(args.input_path)
        cols_to_pull = [
                            "doc_id", "dataset_name", "note_type", "prompt", "index",
                            "entailment_type", "model_name"
                        ]

    # Filter out the specified models
    final_models = [
        'final_merged_gemini-1.5-flash-002_max4000',
        'final_merged_shc-gpt-4o_max4000',
        'merged_meta-llama_Meta-Llama-3-8B-Instruct_split__max4000',
        'final_merged_o1-mini_max16000'
    ]
    sampling_df = entailment_predictions_parsed[
        entailment_predictions_parsed['model_name'].isin(final_models)
    ][cols_to_pull]

    # Sample 1200 rows with a fixed random state
    annotation_df = sampling_df.sample(n=1200, random_state=42)

    # Save the sampled annotation data to a CSV file
    annotation_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()


# input_path = "/share/pi/nigam/akshays/just-the-facts/data/manuscript/entailment_outputs_gemini_llama_4o_o1_101624.jsonl"
# output_path = "/share/pi/nigam/akshays/just-the-facts/data/manuscript/annotation_file_101724.csv"
