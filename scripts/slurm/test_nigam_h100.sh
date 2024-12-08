#!/bin/bash
#
#SBATCH --job-name=cuda-test
#SBATCH --partition=nigam-h100
#SBATCH --gres gpu:1
#SBATCH --time=30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G

# use local conda installation
export CONDA_ENVS_DIRS="/local-scratch/nigam/users/jfries/conda/envs"
export CONDA_PKGS_DIRS="/local-scratch/nigam/users/jfries/conda/pkgs"
source /local-scratch/nigam/users/jfries/conda/miniconda3/etc/profile.d/conda.sh

# print conda debug information
which conda
conda info --envs
conda config --show

conda activate H100

# NVIDIA status dump
nvidia-smi

# test pytorch and CUDA status
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
python /share/pi/nigam/bin/test_h100.py
