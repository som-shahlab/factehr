#!/bin/bash

# make sure to run chmod +x /share/pi/nigam/akshays/scratch/launch_entailment_jobs.sh first
# to run: /share/pi/nigam/akshays/scratch/launch_entailment_jobs.sh

# Base directories
IN_PATH="/share/pi/nigam/rag-the-facts/datasets/sentences/"
OUT_PATH="/share/pi/nigam/rag-data/entailment_final/"

DATASET="coral" 
#NOTE_TYPES="breastca", "pdac"

# radiology_report, progress_note, nursing_note, discharge_summary, breastca
# GPT4, medlm, gemini


for NOTE_TYPE in "pdac"; 

do 
        model_name="GPT4"
        echo "Processing model: $model_name"
        echo "Processing note type: $NOTE_TYPE"
        mkdir -p "${OUT_PATH}${model_name}/${DATASET}/${NOTE_TYPE}"

       
        # Submit recall job

        sbatch --job-name="${model_name}_${DATASET}_${NOTE_TYPE}_recall" \
                --output="${OUT_PATH}${model_name}_${DATASET}_${NOTE_TYPE}_recall.txt" \
                --ntasks=1 \
                --cpus-per-task=1 \
                --mem=60G \
                --time=2-00:00:00 \
                --partition=gpu \
                --nodelist=secure-gpu-20 \
                --gres=gpu:1 \
                --account=nigam \
                --wrap="/bin/bash -c 'source /share/pi/nigam/akshays/miniconda4/etc/profile.d/conda.sh; \
                        conda activate synth-instruct; \
                        cd /share/pi/nigam/users/monreddy/rag-the-facts/entailment; \
                        export HF_HOME=\"/share/pi/nigam/akshays/hf/\"; \
                        export PYTHONUNBUFFERED=1; \
                        python -W ignore entailment_batching.py ${IN_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ ${OUT_PATH}${model_name}/${DATASET}/${NOTE_TYPE}/ 4 recall'"

        

       
done
