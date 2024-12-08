#!/bin/bash
#
# Usage Examples:
# export FACTEHR_DATA_ROOT="/path/to/data"
# ./init_all_datasets.sh
#  
# ./init_all_datasets.sh --data-root data/ --legacy-docs /path/to/legacy/docs
#

FACTEHR_VERSION="v2"
MAX_STRATA_SIZE=10  # Global variable to limit the size of strata for prompt tuning dataset. Set to 0 to skip generating this dataset.

# Process command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --data-root)
      FACTEHR_DATA_ROOT="$2"
      shift 2
      ;;
    --legacy-docs)
      FACTEHR_LEGACY_DOCS="$2"
      shift 2
      ;;
    --max-strata-size)
      MAX_STRATA_SIZE="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

echo "FACTEHR_DATA_ROOT is set to: ${FACTEHR_DATA_ROOT}"
echo "FACTEHR_LEGACY_DOCS is set to: ${FACTEHR_LEGACY_DOCS}"
echo "MAX_STRATA_SIZE is set to: ${MAX_STRATA_SIZE}"

##########################################
#### BUILD FACT DECOMPOSITION DATASET ####
##########################################

TARGET_FILE_PATTERN="${FACTEHR_DATA_ROOT}/datasets/prompted/fact_decomposition_*.jsonl"
SMALL_TARGET_FILE_PATTERN="${FACTEHR_DATA_ROOT}/datasets/prompted/fact_decomposition_small_*.jsonl"
DOCBIN_FILE="${FACTEHR_DATA_ROOT}/datasets/factehr_${FACTEHR_VERSION}.docbin"

# Check for both full-size and small datasets
fact_decomposition_exists=false
fact_decomposition_small_exists=false
docbin_exists=false

if ls $TARGET_FILE_PATTERN 1> /dev/null 2>&1; then
    echo "Full fact decomposition dataset already exists. Skipping creation."
    fact_decomposition_exists=true
fi

if ls $SMALL_TARGET_FILE_PATTERN 1> /dev/null 2>&1; then
    echo "Small fact decomposition dataset already exists. Skipping creation."
    fact_decomposition_small_exists=true
fi

if [ -f "$DOCBIN_FILE" ]; then
    echo "DocBin dataset already exists: ${DOCBIN_FILE}. Skipping creation."
    docbin_exists=true
else
    echo "DocBin dataset not found: ${DOCBIN_FILE}. Proceeding with dataset creation."
fi

if [ "$fact_decomposition_exists" == true ] && [ "$fact_decomposition_small_exists" == true ] && [ "$docbin_exists" == true ]; then
    echo "All required datasets already exist. Nothing to do."
else
    echo "At least one of the datasets is missing. Proceeding with dataset creation..."

    # sample source corpora only if DocBin file doesn't exist
    if [ "$docbin_exists" == false ]; then
        if [ "$FACTEHR_VERSION" == "v1" ]; then
          # DEPRECATED -- replicates original sampling approach
          python scripts/hotfixes/get_note_provenance.py \
          --path_to_legacy ${FACTEHR_LEGACY_DOCS} \
          --args.path_to_input ${FACTEHR_DATA_ROOT}/datasets/ \
          --args.path_to_output ${FACTEHR_DATA_ROOT}/datasets/
        else
          # Any version other than v1 (e.g., v2+)
          python scripts/sample_source_datasets.py \
          --path_to_input ${FACTEHR_DATA_ROOT}/datasets/raw/ \
          --path_to_output ${FACTEHR_DATA_ROOT}/datasets/corpora/${FACTEHR_VERSION}/ \
          --tokenizer tiktoken \
          --min_doc_length 64 \
          --max_doc_length 3840
        fi

        # create prompt templates
        python scripts/build_prompt_templates.py \
        --path_to_output ${FACTEHR_DATA_ROOT}/prompt_templates

        # NLP preprocess documents to create DocBin dataset
        python scripts/build_docbin_dataset.py \
        --path_to_input ${FACTEHR_DATA_ROOT}/datasets/corpora/${FACTEHR_VERSION}/ \
        --path_to_output ${FACTEHR_DATA_ROOT}/datasets/ \
        --n_procs 4 \
        --batch_size 100 \
        --nlp_framework trove \
        --file_name_prefix factehr_${FACTEHR_VERSION}
    fi

    # Create the smaller fact decomposition dataset if it's missing
    # if [ "$fact_decomposition_small_exists" == false ] && [ "$MAX_STRATA_SIZE" -gt 0 ]; then
    #     echo "Creating smaller fact decomposition dataset with MAX_STRATA_SIZE=${MAX_STRATA_SIZE}..."
    #     python scripts/build_fact_decomp_prompted_dataset.py \
    #     --path_to_input ${FACTEHR_DATA_ROOT}/datasets/factehr_${FACTEHR_VERSION}.docbin \
    #     --path_to_prompt_dir ${FACTEHR_DATA_ROOT}/prompt_templates/fact_decomposition/ \
    #     --path_to_output_dir ${FACTEHR_DATA_ROOT}/datasets/prompted/ \
    #     --file_name_prefix fact_decomposition_small \
    #     --completion_format messages \
    #     --max_strata_size ${MAX_STRATA_SIZE}
    # fi

    # Create the full fact decomposition dataset if it's missing
    if [ "$fact_decomposition_exists" == false ]; then
        echo "Creating the full fact decomposition dataset..."
        python scripts/build_fact_decomp_prompted_dataset.py \
        --path_to_input ${FACTEHR_DATA_ROOT}/datasets/factehr_${FACTEHR_VERSION}.docbin \
        --path_to_prompt_dir ${FACTEHR_DATA_ROOT}/prompt_templates/fact_decomposition_delimiter/ \
        --path_to_output_dir ${FACTEHR_DATA_ROOT}/datasets/prompted/ \
        --file_name_prefix fact_decomposition \
        --completion_format messages
    fi
fi
