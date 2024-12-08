#!/bin/bash

# make sure to run chmod +x /share/pi/nigam/akshays/rag-the-facts/slurm/launch_entailment_jobs_loop.sh first
# to run: /share/pi/nigam/akshays/rag-the-facts/slurm/launch_entailment_jobs_loop.sh

# Base directories
IN_PATH="/share/pi/nigam/rag-the-facts/datasets/sentences/"
OUT_PATH="/share/pi/nigam/rag-data/entailment_final/"

# Loop through each model directory in the results path
for model_dir in "$IN_PATH"*/; do
    model_name=$(basename "$model_dir")
    echo "Processing model: $model_name"

    # Loop through each dataset in the model directory
    for dataset_dir in "$model_dir"*/; do
        dataset_name=$(basename "$dataset_dir")
        echo "  Processing dataset: $dataset_name"

        # Loop through each note type in the dataset directory
        for note_type_dir in "$1"*/; do
            note_type_name=$(basename "$note_type_dir")
            echo "    Processing note type: $note_type_name"

            # Create output directory
            mkdir -p "${OUT_PATH}${model_name}/${dataset_name}/${note_type_name}"

            # Submit precision job
            sbatch --job-name="${model_name}_${dataset_name}_${note_type_name}_precision" \
                   --output="${OUT_PATH}${model_name}_${dataset_name}_${note_type_name}_precision.txt" \
                   --ntasks=1 \
                   --cpus-per-task=1 \
                   --mem=50G \
                   --time=2-00:00:00 \
                   --partition=gpu \
                   --gres=gpu:1 \
                   --account=nigam \
                   --wrap="/bin/bash -c 'source /share/pi/nigam/akshays/miniconda4/etc/profile.d/conda.sh; \
                           conda activate synth-instruct; \
                           cd /share/pi/nigam/users/monreddy/rag-the-facts/entailment; \
                           export HF_HOME=\"/share/pi/nigam/\"; \
                           export PYTHONUNBUFFERED=1; \
                           python entailment_mon.py ${IN_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ ${OUT_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ precision'"

            # Submit recall job
            sbatch --job-name="${model_name}_${dataset_name}_${note_type_name}_recall" \
                   --output="${OUT_PATH}${model_name}_${dataset_name}_${note_type_name}_recall.txt" \
                   --ntasks=1 \
                   --cpus-per-task=1 \
                   --mem=50G \
                   --time=2-00:00:00 \
                   --partition=gpu \
                   --gres=gpu:1 \
                   --account=nigam \
                   --wrap="/bin/bash -c 'source /share/pi/nigam/akshays/miniconda4/etc/profile.d/conda.sh; \
                           conda activate synth-instruct; \
                           cd /share/pi/nigam/users/monreddy/rag-the-facts/entailment; \
                           export HF_HOME=\"/share/pi/nigam/\"; \
                           export PYTHONUNBUFFERED=1; \
                           python entailment_mon.py ${IN_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ ${OUT_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ recall'" 

        done
    done
done