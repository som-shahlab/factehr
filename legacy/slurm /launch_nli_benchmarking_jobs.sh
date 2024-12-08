#!/bin/bash

# Make sure to run chmod +x /share/pi/nigam/akshays/scratch/launch_nli_benchmark_jobs.sh first
# To run: /share/pi/nigam/akshays/scratch/launch_nli_benchmark_jobs.sh

# Base directories
DATASETS_DIR="/share/pi/nigam/rag-data/mednli"
OUTPUT_DIR="/share/pi/nigam/rag-data/mednli"
MODELS=(
    # "mistralai/Mixtral-8x7B-Instruct-v0.1"
    # "gemini-1.0-pro-002"
    # "MedLM-large"
    #"meta-llama/Meta-Llama-3-8B-Instruct"
    "microsoft/deberta-large-mnli"
)
DATASETS=(
    # "multi_nli.jsonl"
    # "scitail_bigbio_te.jsonl"
    # "snli.jsonl"
    # "mli_test_v1.jsonl"
    "clinician_annotations_old.jsonl"
)

# Loop through each model
for model_name in "${MODELS[@]}"; do
    # Get the last part of the model name for output naming
    model_base=$(basename "$model_name")
    
    # Loop through each dataset
    for dataset in "${DATASETS[@]}"; do
        # Get the base name of the dataset for output naming
        dataset_base=$(basename "$dataset" .jsonl)
        
        # Define job name and output file name
        job_name="${model_base}_${dataset_base}"
        output_file="${OUTPUT_DIR}/${dataset_base}_results_${model_base}.csv"
        log_file="/share/pi/nigam/akshays/scratch/${dataset_base}_results_${model_base}.txt"
        
        echo "Submitting job for model: $model_name, dataset: $dataset"
        
        # Submit the job
        sbatch --job-name="${job_name}" \
               --output="${log_file}" \
               --ntasks=1 \
               --cpus-per-task=1 \
               --mem=100G \
               --time=2-00:00:00 \
               --partition=gpu \
               --gres=gpu:1 \
               --account=nigam \
               --wrap="/bin/bash -c 'source /share/pi/nigam/akshays/miniconda4/etc/profile.d/conda.sh; \
                       conda activate entailment; \
                       cd /share/pi/nigam/akshays/rag-the-facts; \
                       export HF_HOME=\"/share/pi/nigam/akshays/hf/\"; \
                       export PYTHONUNBUFFERED=1; \
                       export CLOUDSDK_CONFIG=\"/share/pi/nigam/akshays/sdk/\"; \
                       python mednli.py ${model_name} ${output_file} ${DATASETS_DIR}/${dataset}'"
    done
done


# python mednli.py gemini-1.0-pro-002 /share/pi/nigam/rag-data/mednli/snli_results_gemini-1.0-pro-002.csv /share/pi/nigam/rag-data/mednli/snli.jsonl