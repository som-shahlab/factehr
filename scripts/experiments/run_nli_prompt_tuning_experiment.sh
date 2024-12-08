#!/bin/bash

##############################################
################ CONFIG ######################
##############################################

# Define request directories and max_tokens values as pairs
request_dirs=("data/datasets/prompted/binary_entailment" "data/datasets/prompted/rationale_entailment")  # Directories to prompt templates  
max_tokens_list=(25 256)  # Corresponding max token values for each request directory 

# Other necessary variables
save_directory="data/datasets/completions/test/"
generation_config="src/factehr/clients/generation_params.json"
models=("medlm-medium")  #  "meta-llama/Meta-Llama-3-8B-Instruct"  "gemini-1.5-flash-002"
client="vertex"  # Can be "transformers", "openai-batch", "vertex-batch", "vertex"
expected_cost=300  # expected cost of the job in dollars. Will throw an error if estimated cost exceeds expected cost

# Define Vertex AI parameters
project_id="som-nero-phi-nigam-starr"
dataset_id="factehr"
max_samples=100000  
table_id=""  # For batch API - set to none if doesn't exist
prediction_table_id=""  # For batch API - Set to none if doesn't exist

# Define OpenAI parameters
openai_request_url="https://shcopenaisandbox.openai.azure.com/openai/deployments/shc-gpt-4o/chat/completions?api-version=2023-03-15-preview"
max_requests_per_minute=480
max_tokens_per_minute=80000
token_encoding_name="cl100k_base"
max_attempts=5

##############################################
##############################################

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: ./scripts/experiments/run_nli_prompt_tuning_experiment.sh <csv_output_path> <final_metrics_output_path>"
    exit 1
fi

# Assign arguments to variables
CSV_OUTPUT_PATH=$1
FINAL_METRICS_OUTPUT_PATH=$2


# Step 1: Initialize all datasets
echo "Initializing all prompted datasets..."
scripts/init_nli_datasets.sh

# Step 2: Run inference

# Ensure the request_dirs and max_tokens_list arrays have the same length
if [ "${#request_dirs[@]}" -ne "${#max_tokens_list[@]}" ]; then
    echo "Error: request_dirs and max_tokens_list arrays must have the same length."
    exit 1
fi

# Loop through each pair of request directory and max tokens
for i in "${!request_dirs[@]}"; do
    requests_directory="${request_dirs[$i]}"
    max_tokens="${max_tokens_list[$i]}"

    echo "Processing request directory: $requests_directory with max tokens: $max_tokens"

    # Check if requests_directory exists
    if [ ! -d "$requests_directory" ]; then
        echo "Error: Directory $requests_directory does not exist."
        exit 1
    fi

    # Create a temporary merged JSONL file
    merged_jsonl="${requests_directory}/merged_requests.jsonl"

    # remove it if it exists already
    rm "$merged_jsonl"

    # Concatenate all JSONL files into one
    cat "$requests_directory"/*.jsonl > "$merged_jsonl"

    # Check if the merge was successful
    if [ ! -s "$merged_jsonl" ]; then
        echo "Error: Merging JSONL files failed or merged file is empty."
        exit 1
    fi

    echo "Merged JSONL files into $merged_jsonl"

    # Loop through each model
    for model in "${models[@]}"; do
        echo "Running inference for model: $model on merged file"

        # Define the save path, combining the save_directory and model name
        # Replace slashes with underscores in the model name
        model_safe_name=$(echo "$model" | sed 's/\//_/g')

        # Define the save path, combining the save_directory and the sanitized model name
        save_filepath="${save_directory}/merged_${model_safe_name}_max${max_tokens}.jsonl"

        echo "Output path: $save_filepath"
        # Ensure the output directory exists
        if [ ! -d "$save_directory" ]; then
            mkdir -p "$save_directory"
        fi

        # Check which client is being used and launch the appropriate job
        if [ "$client" == "transformers" ]; then
            # Launch the transformers client
            time python src/factehr/clients/transformers_api.py \
            --path_to_prompted_dataset "$merged_jsonl" \
            --path_to_output_file "$save_filepath" \
            --model_name_or_path "$model" \
            --generation_config "$generation_config" \
            --dynamic_batching 40_000 \
            --max_generation_length "$max_tokens"

        elif [ "$client" == "vertex-batch" ]; then
            # Launch the vertex-batch client
            echo "Launching vertex-batch client for model: $model"
            
            # Submit the batch prediction job using vertex_api_batch.py
            time python src/factehr/clients/vertex_api_batch.py \
            --project_id "$project_id" \
            --dataset_id "$dataset_id" \
            --input_jsonl "$merged_jsonl" \
            --output_folder "$save_directory" \
            --model_name "$model" \
            --max_samples "$max_samples" \
            --max_new_tokens "$max_tokens" \
            --generation_config "$generation_config" \
            --max_cost_threshold "$expected_cost" \
            --table_id "$table_id" \
            --prediction_table_id "$prediction_table_id"

        elif [ "$client" == "vertex" ]; then
            # Launch the vertex client (direct API call)
            echo "Launching vertex client for model: $model"
            
            time python src/factehr/clients/vertex_api.py \
            --input_jsonl "$merged_jsonl" \
            --output_jsonl "$save_filepath" \
            --model_name "$model" \
            --generation_config "$generation_config" \
            --max_retries 3 \
            --max_new_tokens "$max_tokens"

        elif [ "$client" == "openai-batch" ]; then
            # Launch the OpenAI batch client
            echo "Launching openai-batch client for model: $model"

            time python src/factehr/clients/azure_openai_api_parallel.py \
            --requests_filepath "$merged_jsonl" \
            --save_filepath "$save_filepath" \
            --request_url "$openai_request_url" \
            --max_requests_per_minute "$max_requests_per_minute" \
            --max_tokens_per_minute "$max_tokens_per_minute" \
            --token_encoding_name "$token_encoding_name" \
            --max_attempts "$max_attempts" \
            --logging_level 20 \
            --model_name "$model" \
            --generation_config "$generation_config" \
            --max_tokens_per_generation "$max_tokens"  \
            --max_cost_threshold "$expected_cost"

        else
            echo "Unsupported client: $client"
            exit 1
        fi
    done

    # Clean up by removing the temporary merged JSONL file
    rm "$merged_jsonl"
done

# Step 3: Parse model outputs and save results to CSV
echo "Parsing model outputs..."
python src/factehr/evaluation/parse_nli_entailment.py "$save_directory" "$CSV_OUTPUT_PATH"

# Step 4: Calculate classification metrics
echo "Calculating metrics..."
python src/factehr/utils/compute_entailment_stats.py "$CSV_OUTPUT_PATH" "$FINAL_METRICS_OUTPUT_PATH"

echo "Process complete!"
