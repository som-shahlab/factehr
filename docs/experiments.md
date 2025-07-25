# Experiment Pipelines

This document provides detailed instructions for running specific subtasks in experiments, covering data preprocessing and LLM evaluation workflows.

## Overview

The experiment pipeline consists of two main phases:
1. **Sample & Preprocess Documents** - Download, sample, and prepare datasets
2. **Running LLM Experiments** - Execute entailment and fact decomposition tasks

---

## 1. Sample & Preprocess Documents

### Status Overview
- ✅ Download source datasets (PhysioNet and SHC)
- ✅ Sample documents from MIMIC-CXR, MIMIC-III, MedAlign, CORAL to create FactEHR dataset
- ✅ NLP sentence tokenize FactEHR documents and serialize to disk (spaCy DocBin)
- ✅ Generate fact decomposition prompted dataset using prompt templates and FactEHR documents

### Prerequisites

Before starting, ensure you have:
- `wget` installed (macOS: `brew install wget`)
- Google Cloud CLI installed and authenticated (`gcloud auth login`)
- PhysioNet credentialed account with signed DUAs
- Stanford VPN connection (required for SHC datasets)
- Hugging Face authentication (`huggingface-cli login`)

### Step 1: Download Source Datasets

Download datasets from PhysioNet and SHC. Files will be saved to `data/datasets/`.

```bash
./scripts/datasets/download_physionet_datasets.sh
./scripts/datasets/download_hf_datasets.sh
```

**Manual Step Required:** Download the MedAlign dataset from [Redivis](https://stanford.redivis.com/datasets/48nr-frxd97exb) and save the notes table as:
```
data/datasets/raw/medalign-aaai_confidential_notes/medalign-aaai_confidential_notes_000000000000.parquet
```

### Step 2: Sample and Create FactEHR Dataset

Sample documents from source datasets with length constraints to control context during inference.

```bash
python scripts/sample_source_datasets.py \
  --path_to_input data/datasets/raw/ \
  --path_to_output data/datasets/corpora/v2/ \
  --file_name_prefix factehr_v2 \
  --tokenizer tiktoken \
  --min_doc_length 64 \
  --max_doc_length 3840
```

### Step 3: NLP Tokenization and Serialization

Tokenize and sentence-split clinical documents using your preferred NLP framework (`medspacy`, `spacy`, or `trove`). This generates a spaCy `DocBin` file named `factehr_YYYYMMDD.docbin`.

> **Note:** See [speed benchmarks](docs/nlp_benchmarks.md) for framework performance comparisons.

```bash
python scripts/build_docbin_dataset.py \
  --path_to_input data/datasets/corpora/v2/ \
  --path_to_output data/datasets/ \
  --n_procs 4 \
  --batch_size 100 \
  --nlp_framework trove \
  --file_name_prefix factehr_v2
```

### Step 4: Generate Prompted Dataset

Create the prompted version of the FactEHR dataset for fact decomposition tasks.

```bash
python scripts/build_fact_decomp_prompted_dataset.py \
  --path_to_input data/datasets/factehr_v2.docbin \
  --path_to_prompt_dir data/prompt_templates/fact_decomposition/ \
  --path_to_output_dir data/datasets/prompted/ \
  --file_name_prefix fact_decomposition \
  --completion_format messages
```

---

## 2. Running LLM Experiments

### Environment Setup

Set required environment variables:

```bash
export HUGGINGFACE_HUB_TOKEN={your_token_here}  # Only needed for transformers client
export FACTEHR_DATA_ROOT={path_to_data_folder}
```

### Experiment 1: Entailment Prompt Tuning

This experiment evaluates entailment performance across different prompt formats: entailment-only, entailment+rationale, and entailment+rationale+CoT.

#### Dataset Preparation

1. Download entailment datasets:
   ```bash
   ./scripts/datasets/download_physionet_datasets.sh
   ./scripts/datasets/download_hf_datasets.sh
   ```

2. **Manual Step:** Copy the FactEHR dev set (`factehr_dev_set.csv`) to:
   ```
   {$FACTEHR_DATA_ROOT}/datasets/raw/entailment/factehr/factehr.csv
   ```

#### Configuration

Adjust settings in `scripts/experiments/run_nli_prompt_tuning_experiment.sh`:

```bash
models=("medlm-medium")  # Options: "meta-llama/Meta-Llama-3-8B-Instruct", "gemini-1.5-flash-002"
client="vertex"          # Options: "transformers", "openai-batch", "vertex-batch", "vertex"
```

> **Important:** If using GPT models, define your API endpoint URL in `scripts/experiments/run_inference_client.sh`.

#### Execution

Run the complete experiment pipeline:

```bash
scripts/experiments/run_nli_prompt_tuning_experiment.sh <csv_output_path> <final_metrics_output_path> | tee output.log
```

The script automatically runs:
1. Binary entailment prompts (`max_new_tokens = 25`)
2. Rationale entailment prompts (`max_new_tokens = 256`)

### Experiment 2: Fact Decomposition and Entailment

This workflow involves running prompts through an LLM client using the `run_inference_client.sh` script.

#### Script Usage

```bash
scripts/experiments/run_inference_client.sh \
  [PATH_TO_PROMPTED_JSONL_DATA] \
  [MODEL_NAME] \
  [CLIENT_NAME] \
  [N_TMUX_SESSIONS] \
  [MAX_OUTPUT_TOKENS] \
  [GPT_ENDPOINT_URL]
```

#### Example Command

```bash
scripts/experiments/run_inference_client.sh \
  data/datasets/prompted/fact_decomposition_20241009_medalign.jsonl \
  "gemini-1.5-flash-002" \
  "vertex" \
  5 \
  4000
```

#### Complete Workflow

Follow these steps in order:

1. **Generate prompted dataset for fact decomposition:**
   ```bash
   scripts/init_all_datasets.sh
   ```

2. **Perform fact decomposition:**
   ```bash
   scripts/experiments/run_inference_client.sh [dataset] [model] [client] [sessions] [tokens]
   ```

3. **Create entailment pairs from fact decomposition:**
   ```bash
   scripts/experiments/create_entailment_file_from_fact_decomp.sh
   ```

4. **Perform entailment evaluation using LLM as judge:**
   ```bash
   scripts/experiments/run_inference_client.sh [entailment_data] [model] [client] [sessions] [tokens]
   ```

---

## Additional Resources

- [NLP Framework Benchmarks](docs/nlp_benchmarks.md)
- PhysioNet: Requires credentialed account and signed DUAs
- Stanford VPN: Required for SHC dataset access
- [MedAlign Dataset](https://stanford.redivis.com/datasets/48nr-frxd97exb)
