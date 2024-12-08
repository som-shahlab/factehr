#!/bin/bash

sbatch --job-name="testing_batching" \
        --output="testing_batching.txt" \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=50G \
        --time=2-00:00:00 \
        --partition=gpu \
        --gres=gpu:1 \
        --account=nigam \
        # --wrap="nvidia-smi"
        --wrap="/bin/bash -c 'source /share/pi/nigam/akshays/miniconda4/etc/profile.d/conda.sh; \
                conda activate synth-instruct; \
                cd /share/pi/nigam/users/monreddy/rag-the-facts/entailment; \
                export HF_HOME=\"/share/pi/nigam/akshays/hf/\"; \
                export PYTHONUNBUFFERED=1; \
                python -W ignore test.py'"

