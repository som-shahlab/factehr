#!/bin/bash

# Runs inference on the OpenAI client, looping across files in a specified directory and models.
# References a configuration file (JSON format) that provides paths, settings, and model details.
# The script processes each .jsonl file in the requests directory for each model listed in the config file,
# and saves the results with a model-specific filename in the designated output directory.
#
# The configuration file should have the following structure:
# {
#   "requests_directory": "path/to/requests_directory",    # Directory containing the input .jsonl files
#   "save_directory": "path/to/save_directory",            # Directory to store the output files
#   "request_url": "https://api.openai.com/v1/chat/completions",  # API endpoint URL
#   "generation_config": "path/to/generation_config.json",  # Path to generation configuration JSON file
#   "models": ["gpt-4", "gpt-3.5-turbo"]                   # List of model names to run inference on
# }
#
# The script uses jq to parse the JSON configuration file, loops over .jsonl files in the requests directory,
# and runs inference on each file using each model specified in the config.
# The output for each model is saved to the save directory with a model-specific filename.
#
# Usage:
#   ./src/factehr/clients/run_openai_client.sh <config_filepath>
#
# Example:
#   ./src/factehr/clients/run_openai_client.sh config.json
#
# The resulting output files will be saved with the format:
#   <save_directory>/<base_filename>_<model>.jsonl
# where <base_filename> is the name of the input .jsonl file without the extension and <model> is the model name.

# Check if correct number of arguments are passed
if [ "$#" -ne 1 ]; then
    echo "Usage: ./src/factehr/clients/run_openai_client.sh <config_filepath>"
    exit 1
fi

# Assign the config filepath
config_filepath=$1

# Load the config file
requests_directory=$(jq -r '.requests_directory' "$config_filepath")
save_directory=$(jq -r '.save_directory' "$config_filepath")
request_url=$(jq -r '.request_url' "$config_filepath")
generation_config=$(jq -r '.generation_config' "$config_filepath")
models=$(jq -r '.models[]' "$config_filepath")

# Check if requests_directory exists
if [ ! -d "$requests_directory" ]; then
    echo "Error: Directory $requests_directory does not exist."
    exit 1
fi

# Loop through each JSONL file in the requests_directory
for jsonl_file in "$requests_directory"/*.jsonl; do
    # Extract the base name of the file (without the extension)
    base_filename=$(basename "$jsonl_file" .jsonl)

    # Loop through each model from the config file
    for model in $models; do
        echo "Running API call for model: $model on file: $base_filename.jsonl"

        # Define the save path, combining the save_directory, base filename, and model name
        save_filepath="${save_directory}${base_filename}_${model}.jsonl"

        # Call the Python script with the model name and the current JSONL file
        python src/factehr/clients/azure_openai_api_parallel.py \
        --requests_filepath "$jsonl_file" \
        --save_filepath "$save_filepath" \
        --request_url "$request_url" \
        --max_requests_per_minute 480 \
        --max_tokens_per_minute 80000 \
        --token_encoding_name cl100k_base \
        --max_attempts 5 \
        --logging_level 20 \
        --generation_config "$generation_config" \
        --model_name "$model"
    done
done
