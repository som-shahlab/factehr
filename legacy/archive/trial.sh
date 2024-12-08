#!/bin/bash

# make sure to run chmod +x /share/pi/nigam/akshays/rag-the-facts/slurm/launch_entailment_jobs_loop.sh first
# to run: /share/pi/nigam/akshays/rag-the-facts/slurm/launch_entailment_jobs_loop.sh

# Base directories
IN_PATH="/share/pi/nigam/rag-data/results_final/"
OUT_PATH="/share/pi/nigam/rag-data/entailment_results_final/"

# Loop through each model directory in the results path
for model_dir in "$IN_PATH"*/; do
    model_name=$(basename "$model_dir")
    echo "Processing model: $model_name"

    # Loop through each dataset in the model directory
    for dataset_dir in "$model_dir"*/; do
        dataset_name=$(basename "$dataset_dir")
        echo "  Processing dataset: $dataset_name"

        # Loop through each note type in the dataset directory
        for note_type_dir in "$dataset_dir"*/; do
            note_type_name=$(basename "$note_type_dir")
            echo "    Processing note type: $note_type_name"
            
        done
    done
done