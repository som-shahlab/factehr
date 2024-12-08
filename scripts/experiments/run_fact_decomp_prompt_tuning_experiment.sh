#!/bin/bash

# Usage: ./run_fact_decomp_prompt_tuning_experiment.sh <request_file> <model> <client> [splits]

# example 
# scripts/experiments/run_fact_decomp_prompt_tuning_experiment.sh "data/datasets/prompted_sampled/fact_decomposition_20241008.jsonl" "meta-llama/Meta-Llama-3-8B-Instruct" "transformers"

# Models include: gemini-1.5-flash-002 shc-gpt-4o meta-llama/Meta-Llama-3-8B-Instruct medlm-medium
# clients include: "transformers", "openai-batch", "vertex-batch", "vertex"

# Parse command-line arguments
REQUEST_FILE="$1"  # Input request file as first argument
models=("$2")      # Model name as second argument
client="$3"        # Client type as third argument
SPLITS="${4:-25}"  # Optional splits argument with default of 25

MAX_TOKENS=4000

# Other necessary variables
save_directory="${FACTEHR_DATA_ROOT}/datasets/completions/test/"
splits_directory="${save_directory}/splits"  # New directory for splits
generation_config="src/factehr/clients/generation_params.json"
expected_cost=300

# Define OpenAI parameters
openai_request_url="https://shcopenaisandbox.openai.azure.com/openai/deployments/shc-gpt-4o/chat/completions?api-version=2023-03-15-preview"

# clear it first, then create splits directory
if [ -d "$splits_directory" ]; then
  rm -r "$splits_directory"
fi

mkdir -p "$splits_directory"

# Function to split the input file for parallel processing
split_input_file() {
    local input_file=$1
    local num_splits=$2
    local output_dir=$3

    echo "Splitting $input_file into $num_splits parts..."
    split -l $(( $(wc -l < "$input_file") / num_splits )) -d -a 2 "$input_file" "$output_dir/split_"
}

echo "Processing request file: $REQUEST_FILE with max tokens: $MAX_TOKENS"

# Split the input file if the client is vertex or openai
if [[ "$client" == "vertex" || "$client" == "openai-batch" ]]; then

    echo "Deleting existing split files in $splits_directory..."
    rm -f "$splits_directory"/split_*.jsonl  # Removes files matching the split pattern

    split_input_file "$REQUEST_FILE" "$SPLITS" "$splits_directory"
    a=0
    for file in "$splits_directory"/split_*; do
        mv "$file" "${file}.jsonl"
        a=$((a + 1))
    done

    # Loop through each model and launch jobs in tmux
    for model in "${models[@]}"; do
        echo "Running inference for model: $model on request file"

        # Define the save path, combining the save_directory and model name
        model_safe_name=$(echo "$model" | sed 's/\//_/g')

        # Loop through all split files in the split directory
        for split_file in "$splits_directory"/split_*.jsonl; do
            i=$(basename "$split_file" | sed 's/split_//' | sed 's/.jsonl//')  # Extract split index
            save_filepath="${splits_directory}/merged_${model_safe_name}_split_${i}_max${MAX_TOKENS}.jsonl"

            # Debugging: Print the resolved file paths
            echo "Processing split $i for model: $model"
            echo "Split file path: $split_file"
            echo "Save file path: $save_filepath"
            
            # Launch the tmux session with properly resolved variables
            tmux new-session -d -s "${model_safe_name}_split_${i}" "
                echo 'Running split $i for model: $model' > ${splits_directory}/tmux_log_${model_safe_name}_split_${i}.log 2>&1;
                echo 'Save file path: ${save_filepath}' >> ${splits_directory}/tmux_log_${model_safe_name}_split_${i}.log 2>&1;

                # Run the generation depending on the client
                if [ \"$client\" == \"vertex\" ]; then
                    python src/factehr/clients/vertex_api.py \
                        --input_jsonl \"$split_file\" \
                        --output_jsonl \"$save_filepath\" \
                        --model_name \"$model\" \
                        --generation_config \"$generation_config\" \
                        --max_retries 3 \
                        --max_new_tokens \"$MAX_TOKENS\" >> ${splits_directory}/tmux_log_${model_safe_name}_split_${i}.log 2>&1;
                elif [ \"$client\" == \"openai-batch\" ]; then
                    python src/factehr/clients/azure_openai_api_parallel.py \
                        --requests_filepath \"$split_file\" \
                        --save_filepath \"$save_filepath\" \
                        --request_url \"$openai_request_url\" \
                        --model_name \"$model\" \
                        --generation_config \"$generation_config\" \
                        --max_tokens_per_generation \"$MAX_TOKENS\" \
                        --max_cost_threshold \"$expected_cost\" >> ${splits_directory}/tmux_log_${model_safe_name}_split_${i}.log 2>&1;
                fi;

                echo 'Completed split $i for model: $model' >> ${splits_directory}/tmux_log_${model_safe_name}_split_${i}.log 2>&1;
                tmux wait-for -S ${model_safe_name}_split_${i}_done
            "
        done
    done

    # Use a wait loop to ensure that all tmux jobs complete
    for model in "${models[@]}"; do
        model_safe_name=$(echo "$model" | sed 's/\//_/g')
        for split_file in "$splits_directory"/split_*.jsonl; do
            i=$(basename "$split_file" | sed 's/split_//' | sed 's/.jsonl//')
            tmux wait-for "${model_safe_name}_split_${i}_done"
        done
    done

    # Merge the output split files into a single final file
    final_output_filepath="${save_directory}/final_merged_${model_safe_name}_max${MAX_TOKENS}.jsonl"
    echo "Merging all split files into ${final_output_filepath}..."
    find "${splits_directory}" -name "merged_${model_safe_name}_split_*.jsonl" -exec cat {} + > "$final_output_filepath"

    echo "Process complete for all models!"

fi

if [ "$client" == "transformers" ]; then
# Loop through each model and launch jobs in tmux
    for model in "${models[@]}"; do
        echo "Running inference for model: $model on request file"

        # Define the save path, combining the save_directory and model name
        model_safe_name=$(echo "$model" | sed 's/\//_/g')

        save_filepath="${save_directory}/merged_${model_safe_name}_split_${i}_max${MAX_TOKENS}.jsonl"

        # Launch the transformers client
        time python src/factehr/clients/transformers_api.py \
        --path_to_prompted_dataset "$REQUEST_FILE" \
        --path_to_output_file "$save_filepath" \
        --model_name_or_path "$model" \
        --generation_config "$generation_config" \
        --dynamic_batching 20_000 \
        --max_generation_length "$MAX_TOKENS"
        

    done
fi

