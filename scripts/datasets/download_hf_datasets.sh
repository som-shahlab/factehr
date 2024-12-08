#!/bin/bash

# Script to download HuggingFace dataset assets

# Directory to save the datasets
DATA_ROOT="data/datasets/raw"

# Check for command-line argument to override the default DATA_ROOT
if [[ -n "$1" ]]; then
  DATA_ROOT="$1"
  echo "Using DATA_ROOT from command-line argument: $DATA_ROOT"
else
  echo "Using default DATA_ROOT: $DATA_ROOT"
fi

# Python script for loading and saving datasets
PYTHON_SCRIPT="scripts/datasets/download_hf_datasets.py"

# List of datasets to process
DATASETS=("scitail" "multinli" "snli" "mednli")

# Loop through each dataset and load/save it
for dataset in "${DATASETS[@]}"; do
    echo "Processing dataset: $dataset"
    
    # Create the directory to save this dataset
    DATASET_SAVE_DIR="$DATA_ROOT/huggingface/$dataset"
    
    # Check if the directory exists and is non-empty
    if [[ -d "$DATASET_SAVE_DIR" && "$(ls -A $DATASET_SAVE_DIR)" ]]; then
        echo "Directory $DATASET_SAVE_DIR already exists and is non-empty. Skipping dataset $dataset."
    else
        # Directory does not exist or is empty, proceed to download
        echo "Downloading dataset $dataset..."
        
        # Run the Python script to load and save the dataset
        python $PYTHON_SCRIPT \
            --dataset_name "$dataset" \
            --path_to_save_dir "$DATASET_SAVE_DIR"
        
        echo "Finished processing dataset: $dataset"
    fi
    
    echo "-----------------------------"
done

echo "All datasets processed and saved."
