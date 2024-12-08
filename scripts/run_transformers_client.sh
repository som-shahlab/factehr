#!/bin/bash

# Runs inference using the transformers API after merging all JSONL files in the specified directory.
# The script merges the JSONL files, then processes them for each model listed in the config file.
#
# The configuration file should have the following structure:
# {
#   "requests_directory": "path/to/requests_directory",    # Directory containing the input .jsonl files
#   "save_directory": "path/to/save_directory",            # Directory to store the output files
#   "generation_config": "path/to/generation_config.json",  # Path to generation configuration JSON file
#   "models": ["meta-llama/Meta-Llama-3-8B-Instruct"]       # List of model names to run inference on
# }
#
# Usage:
#   ./src/factehr/clients/run_transformers_client.sh <config_filepath>
#
# Example:
#   ./src/factehr/clients/run_transformers_client.sh config.json
#
# The resulting output files will be saved with the format:
#   <save_directory>/<base_filename>_<model>.jsonl

# Check if correct number of arguments are passed
if [ "$#" -ne 1 ]; then
    echo "Usage: ./src/factehr/clients/run_transformers_client.sh <config_filepath>"
    exit 1
fi

# Assign the config filepath
config_filepath=$1

# Load the config file
requests_directory=$(jq -r '.requests_directory' "$config_filepath")
save_directory=$(jq -r '.save_directory' "$config_filepath")
generation_config=$(jq -r '.generation_config' "$config_filepath")
models=$(jq -r '.models[]' "$config_filepath")
uid_string=$(jq -r '.uid_string' "$config_filepath")

# Check if requests_directory exists
if [ ! -d "$requests_directory" ]; then
    echo "Error: Directory $requests_directory does not exist."
    exit 1
fi

# Create a temporary merged JSONL file
merged_jsonl="${requests_directory}/merged_requests.jsonl"

# Concatenate all JSONL files into one
cat "$requests_directory"/*.jsonl > "$merged_jsonl"

# Check if the merge was successful
if [ ! -s "$merged_jsonl" ]; then
    echo "Error: Merging JSONL files failed or merged file is empty."
    exit 1
fi

echo "Merged JSONL files into $merged_jsonl"

# Loop through each model from the config file
for model in $models; do
    echo "Running inference for model: $model on merged file"

    # Define the save path, combining the save_directory and model name
    save_filepath="${save_directory}/merged_${model}.jsonl"

    # Ensure the output directory exists
    if [ ! -d "$save_directory" ]; then
        mkdir -p "$save_directory"
    fi

    # Call the Python script using the transformers API with the specified model and merged JSONL file
    python src/factehr/clients/transformers_api.py \
    --path_to_prompted_dataset "$merged_jsonl" \
    --path_to_output_file "$save_filepath" \
    --model_name_or_path "$model" \
    --generation_config "$generation_config" \
    --dynamic_batching 40_000 \
    --uid_string "$uid_string"
done

# Clean up by removing the temporary merged JSONL file
rm "$merged_jsonl"
