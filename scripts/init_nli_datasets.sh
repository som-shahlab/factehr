#!/bin/bash

##########################################
########### BUILD NLI DATASETS ###########
##########################################
DATASETS=("factehr" "scitail" "mednli" "multinli" "snli")  # Datasets to process
DEST_DIR="${FACTEHR_DATA_ROOT}/datasets/raw/entailment"
HUGGINGFACE_DIR="${FACTEHR_DATA_ROOT}/datasets/raw/huggingface"
PROMPT_TEMPLATE_DIRS=("${FACTEHR_DATA_ROOT}/prompt_templates/binary_entailment" "${FACTEHR_DATA_ROOT}/prompt_templates/rationale_entailment")  # List of prompt template directories
SPLIT_NAME="test"
SAMPLE_PROB=1

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Helper function to check if a dataset exists in the destination directory
copy_if_not_exists() {
    local src_dir=$1
    local dest_dir=$2
    local dataset_name=$3
    
    if [[ -d "$dest_dir/$dataset_name" && "$(ls -A $dest_dir/$dataset_name)" ]]; then
        echo "Dataset $dataset_name already exists in $dest_dir, skipping copy."
    else
        echo "Copying $dataset_name to $dest_dir..."
        cp -r "$src_dir/$dataset_name" "$dest_dir/"
        echo "Finished copying $dataset_name."
    fi
}

# Copy datasets only from the DATASETS variable
for dataset in "${DATASETS[@]}"; do
    copy_if_not_exists "$HUGGINGFACE_DIR" "$DEST_DIR" "$dataset"
done

# Loop through each prompt template directory
for prompt_template_dir in "${PROMPT_TEMPLATE_DIRS[@]}"; do
    # Extract the directory name (last part of the path)
    prompt_template_dir_name=$(basename "$prompt_template_dir")
    
    echo "Using prompt template directory: $prompt_template_dir_name"
    
    # Process each dataset for the current prompt template directory
    for dataset in "${DATASETS[@]}"; do
        echo "Processing dataset: $dataset"
        
        # Path to the saved dataset on disk
        DATASET_PATH="$DEST_DIR/$dataset"
        
        # Define the output directory based on the prompt template directory name
        OUTPUT_DIR="${FACTEHR_DATA_ROOT}/datasets/prompted/${prompt_template_dir_name}"
        
        # Create the output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"
        
        # File name prefix for the prompted datasets
        FILE_NAME_PREFIX="entailment"

        # Check if a file matching the pattern exists (with wildcard for the date part)
        # The pattern will be something like: entailment_dataset_prompt_template_dir_*.jsonl
        if find "$OUTPUT_DIR" -name "${FILE_NAME_PREFIX}_${dataset}_*.jsonl" | grep -q '.'; then
            echo "Matching prompted dataset for $dataset with $prompt_template_dir_name already exists. Skipping..."
            continue
        fi
        
        # Run the Python script to generate the prompted dataset
        python scripts/build_nli_prompted_datasets.py \
            --path_to_prompt_dir "$prompt_template_dir" \
            --path_to_output_dir "$OUTPUT_DIR" \
            --dataset_path "$DATASET_PATH" \
            --dataset_name "$dataset" \
            --sample_prob "$SAMPLE_PROB" \
            --split_name "$SPLIT_NAME"
        
        echo "-----------------------------"
    done
done
