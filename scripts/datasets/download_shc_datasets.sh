#!/bin/bash

# Download SHC datasets from BigQuery
# IMPORTANT! This is only for lab-internal use and only STARR confidential data
#
# - `release_notes` 237 patients, 44,547 notes from AAAI 2024 MedAlign paper
# - `notes`  77,200 patients, 7,947,553 notes forming the population that 
#            the release_notes cohort is drawn from 
#
# -- This SQL creates the MedAlign note table used in the AAAI 2024 paper
#
# CREATE TABLE `som-nero-nigam-starr.medalign_internal.aaai_release_note` AS
# SELECT note.* 
# FROM `som-nero-nigam-starr.medalign_internal.note` AS note
# JOIN `som-nero-nigam-starr.medalign_internal.cohort` AS cohort
# ON note.person_id = cohort.person_id
# WHERE cohort.release = True;
#

# Initialize default values
DATA_ROOT="data/datasets/raw/"
TABLE_ID="note" # Default TABLE_ID

# Process command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --table-id)
      TABLE_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-$DATA_ROOT}"

echo "Using DATA_ROOT: $DATA_ROOT"
echo "Using TABLE_ID: $TABLE_ID"
echo "Downloading SHC datasets from BigQuery to $LOCAL_OUTPUT_DIR"

# Set other variables with defaults
PROJECT_ID="${PROJECT_ID:-som-nero-phi-nigam-starr}"
DATASET_ID="${DATASET_ID:-medalign_aaai_2024_release_dua}"
GCP_BUCKET="${GCP_BUCKET:-som_nero_phi_nigam_starr_extract_scratch}"


# Determine the appropriate output file name based on the TABLE_ID
case "$TABLE_ID" in
  aaai_release_note)
    TABLE_ID="aaai_release_note"
    OUTPUT_FILE="${OUTPUT_FILE:-medalign-aaai_release_notes.parquet}"
    GCS_OUTPUT_PATH="gs://$GCP_BUCKET/$OUTPUT_FILE"
    LOCAL_OUTPUT_PATH="$LOCAL_OUTPUT_DIR/$OUTPUT_FILE"
    ;;
  note)
    TABLE_ID="note"
    OUTPUT_FILE="${OUTPUT_FILE:-medalign-aaai_confidential_notes}"
    # Full path for the cloud and local output with wildcard for sharding
    GCS_OUTPUT_PATH="gs://$GCP_BUCKET/${OUTPUT_FILE}_*.parquet"
    LOCAL_OUTPUT_PATH="$LOCAL_OUTPUT_DIR/$OUTPUT_FILE/"
    ;;
  *)
    OUTPUT_FILE="${OUTPUT_FILE:-$TABLE_ID.parquet}"
    GCS_OUTPUT_PATH="gs://$GCP_BUCKET/$OUTPUT_FILE"
    LOCAL_OUTPUT_PATH="$LOCAL_OUTPUT_DIR/$OUTPUT_FILE"
    ;;
esac

# Create the output directory if it doesn't exist
mkdir -p "$LOCAL_OUTPUT_DIR"

# Download the BigQuery table as a compressed Parquet file with sharding
echo "Starting BigQuery extract..."
if bq extract --destination_format=PARQUET \
              --compression=SNAPPY \
              "$PROJECT_ID:$DATASET_ID.$TABLE_ID" \
              "$GCS_OUTPUT_PATH"; then
  echo "BigQuery extract successful: $GCS_OUTPUT_PATH"
else
  echo "BigQuery extract failed" >&2
  exit 1
fi

# Download the files from Google Cloud Storage bucket
echo "Downloading files from GCS to local directory..."
if [ "$TABLE_ID" = "note" ]; then
  # For the confidential note table, download to a folder
  mkdir -p "$LOCAL_OUTPUT_PATH"
  if gsutil -m cp "$GCS_OUTPUT_PATH" "$LOCAL_OUTPUT_PATH"; then
    echo "Files downloaded successfully to $LOCAL_OUTPUT_PATH"
  else
    echo "File download failed" >&2
    exit 1
  fi
else
  # For the release note table, download to a single file
  if gsutil cp "$GCS_OUTPUT_PATH" "$LOCAL_OUTPUT_PATH"; then
    echo "File downloaded successfully to $LOCAL_OUTPUT_PATH"
  else
    echo "File download failed" >&2
    exit 1
  fi
fi

# Output the result
echo "Table downloaded to $LOCAL_OUTPUT_PATH"
