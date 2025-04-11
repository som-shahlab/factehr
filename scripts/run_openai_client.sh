#!/bin/bash

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
