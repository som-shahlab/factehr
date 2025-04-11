#!/bin/bash
#
#SBATCH --job-name=cuda-test
#SBATCH --partition=h100
#SBATCH --gres gpu:1
#SBATCH --time=2-00:00:00 # 2 days is the default limit
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G

# setup local conda
export CONDA_ENVS_DIRS="../conda/envs"
export CONDA_PKGS_DIRS="../conda/pkgs"
source ../conda/miniconda3/etc/profile.d/conda.sh
conda activate H100

# setup auth tokens and other env variables
source /home/jfries/setup-env-vars.sh

CODE_ROOT="../code/rag-the-facts"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

# run inference script
python ${CODE_ROOT}/src/factehr/clients/transformers_api.py \
--path_to_prompted_dataset ${CODE_ROOT}/data/datasets/prompted/fact_decomposition_20240829.jsonl \
--path_to_output_file ${CODE_ROOT}/debug.jsonl \
--model_name_or_path ${MODEL_NAME} \
--generation_config ${CODE_ROOT}/src/factehr/clients/generation_params.json
