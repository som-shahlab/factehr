#!/bin/bash

# Script to download PhysioNet dataset assets
# Downloading requires a PhysioNet account and signed DUAs
# See https://physionet.org/register/

# Data assets to be downloaded:
# 1. MIMIC-III Notes
# 2. MIMIC-CXR Radiology Notes
# 3. CORAL: expert-Curated medical Oncology Reports to Advance Language model inference
# 4. MedNLI - A Natural Language Inference Dataset For The Clinical Domain
# 5. RadNLI: A natural language inference dataset for the radiology domain

# Set the root directory for datasets
DATA_ROOT="data/datasets/raw/"

# Check for a command-line argument to override the default DATA_ROOT
if [[ -n "$1" ]]; then
  DATA_ROOT="$1"
  echo "Using DATA_ROOT from command-line argument: $DATA_ROOT"
else
  echo "Using default DATA_ROOT: $DATA_ROOT"
fi

# Prompt for the PhysioNet username
read -p "Enter your PhysioNet username: " USERNAME

# Prompt the user for their password without showing input
read -s -p "Enter your password: " PASSWORD
echo

# =============================================================================
# 1. MIMIC-III
# =============================================================================
MIMIC_III_URL="https://physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
MIMIC_III_TARGET="${DATA_ROOT}physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"

# Check if the MIMIC-III file already exists
if [ -f "$MIMIC_III_TARGET" ]; then
    echo "File $MIMIC_III_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$MIMIC_III_TARGET")"
    # Download the file using wget
    wget -r -N -c -np --user "$USERNAME" --password "$PASSWORD" -P "${DATA_ROOT}" "$MIMIC_III_URL"
fi

# =============================================================================
# 2. MIMIC-CXR
# =============================================================================
MIMIC_CXR_URL="https://physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
MIMIC_CXR_SPLIT_URL="https://physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz"
MIMIC_CXR_TARGET="${DATA_ROOT}physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
MIMIC_CXR_SPLIT_TARGET="${DATA_ROOT}physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz"

# Check if the MIMIC-CXR reports file already exists
if [ -f "$MIMIC_CXR_TARGET" ]; then
    echo "File $MIMIC_CXR_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$MIMIC_CXR_TARGET")"
    # Download radiology reports from MIMIC-CXR
    wget -r -N -c -np --user "$USERNAME" --password "$PASSWORD" -P "${DATA_ROOT}" "$MIMIC_CXR_URL"
fi

# Check if the MIMIC-CXR-JPG split file already exists
if [ -f "$MIMIC_CXR_SPLIT_TARGET" ]; then
    echo "File $MIMIC_CXR_SPLIT_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$MIMIC_CXR_SPLIT_TARGET")"
    # Download canonical splits from MIMIC-CXR-JPG
    wget -r -N -c -np --user "$USERNAME" --password "$PASSWORD" -P "${DATA_ROOT}" "$MIMIC_CXR_SPLIT_URL"
fi

# =============================================================================
# 3. CORAL
# =============================================================================
CORAL_URL="https://physionet.org/files/curated-oncology-reports/1.0/"
CORAL_TARGET="${DATA_ROOT}physionet.org/files/curated-oncology-reports/1.0/"

# Check if the CORAL dataset already exists
if [ -d "$CORAL_TARGET" ]; then
    echo "Directory $CORAL_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$CORAL_TARGET")"
    # Download CORAL dataset
    wget -r -N -c -np --user "$USERNAME" --password "$PASSWORD" -P "${DATA_ROOT}" "$CORAL_URL"
fi

# =============================================================================
# 4. MedNLI
# =============================================================================
MEDNLI_URL="https://physionet.org/content/mednli/get-zip/1.0.0/"
MEDNLI_TARGET="${DATA_ROOT}/physionet.org/content/mednli/get-zip/1.0.0/mednli-a-natural-language-inference-dataset-for-the-clinical-domain-1.0.0.zip"

# Check if the CORAL dataset already exists
if [ -d "$MEDNLI_TARGET" ]; then
    echo "Directory $MEDNLI_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$MEDNLI_TARGET")"
    # Download CORAL dataset
    wget --user "$USERNAME" --password "$PASSWORD" -O "$MEDNLI_TARGET" "$MEDNLI_URL"
fi

# =============================================================================
# 4. RadNLI
# =============================================================================
RADNLI_URL="https://physionet.org/files/radnli-report-inference/1.0.0/"
RADNLI_TARGET="${DATA_ROOT}physionet.org/files/radnli-report-inference/1.0.0/"

# Check if the CORAL dataset already exists
if [ -d "$RADNLI_TARGET" ]; then
    echo "Directory $RADNLI_TARGET already exists. Skipping download."
else
    # Create the target directory if it doesn't exist
    mkdir -p "$(dirname "$RADNLI_TARGET")"
    # Download CORAL dataset
    wget -r -N -c -np --user "$USERNAME" --password "$PASSWORD" -P "${DATA_ROOT}" "$RADNLI_URL"
fi
