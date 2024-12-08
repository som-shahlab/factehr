#!/bin/bash

# Usage example command
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <save_directory>"
    exit 1
fi

save_directory="$1"  # Directory containing the JSONL files to process
notes_data_dir="${FACTEHR_DATA_ROOT}/datasets/corpora/v2"   # Path to your notes directory
prompt_template_file="${FACTEHR_DATA_ROOT}/prompt_templates/entailment/entailment.tmpl"  # Path to the prompt template file

# Directory to store individual entailment files
entailment_dir="${save_directory}/entailment_files"
mkdir -p "$entailment_dir"

# Function to process each .jsonl file
process_jsonl_file() {
    local input_file="$1"
    local model_safe_name="$2"
    local entailment_output_file="${entailment_dir}/entailment_input_${model_safe_name}.jsonl"
    
    echo "Creating entailment dataset for model: $model_safe_name"

    python src/factehr/utils/parse_facts.py \
        --model_output "$input_file" \
        --output_file "$entailment_output_file" \
        --notes_dir "$notes_data_dir" \
        --prompt_template "$prompt_template_file" \
        --model_name "$model_safe_name"

    echo "Entailment dataset created: $entailment_output_file"
}

# Loop through each .jsonl file in the save directory
for file in "$save_directory"/*.jsonl; do
    model_safe_name=$(basename "$file" .jsonl)  # Get the file name without the extension
    process_jsonl_file "$file" "$model_safe_name"
done

# Merge all the entailment-specific JSONL files
merged_entailment_file="${save_directory}/merged_entailment.jsonl"
echo "Merging all entailment files into $merged_entailment_file"
cat "$entailment_dir"/*.jsonl > "$merged_entailment_file"

echo "All entailment files merged: $merged_entailment_file"
