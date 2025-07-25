
# Experiment Pipelines

Initial detailed documentation on running specific subtasks in experiments.

## 1. Sample & Preprocess Documents

### 🚧 Status: Subtasks
- ✅ Download source datasets (PhysioNet and SHC)
- ✅ Sample documents from MIMIC-CXR, MIMIC-III, MedAlign, CORAL to create FactEHR dataset
- ✅ NLP sentence tokenize FactEHR documents and serialize to disk (spaCy DocBin).
- ✅ Generate fact decomposition prompted dataset using prompt templates and FactEHR documents


#### 1. Download source datasets (PhysioNet and SHC)

Download source datasets from PhysioNet and SHC. Saves to `data/datasets/` 

***Requirements***

- `wget` installed (on MacOS use [homebrew](https://brew.sh/) `brew install wget`)
- gcloud CLI installed (see [instructions](https://cloud.google.com/sdk/docs/install)) and authenticated `gcloud auth login`
- PhysioNet credentialed account and per-dataset signed DUAs
- You must be connected to the Stanford VPN to download SHC datasets (MedAlign)
- Authenticated with huggingface `huggingface-cli login`

```bash
./scripts/datasets/download_physionet_datasets.sh
./scripts/datasets/download_hf_datasets.sh
```

Download the Medalign dataset from Redivis (https://stanford.redivis.com/datasets/48nr-frxd97exb) and put the notes table here saved as a parquet:

```
data/datasets/raw/medalign-aaai_confidential_notes/medalign-aaai_confidential_notes_000000000000.parquet
```

#### 2. Sample documents from MIMIC-CXR, MIMIC-III, MedAlign, CORAL to create FactEHR dataset

Sampling from source datasets, assign primary key and export to CSVs for preprocessing. We impose min/max length constraints to control for extreme context lengths during inference.

```bash
python scripts/sample_source_datasets.py \
--path_to_input data/datasets/raw/ \
--path_to_output data/datasets/corpora/v2/ \
--file_name_prefix factehr_v2 \
--tokenizer tiktoken \
--min_doc_length 64 \
--max_doc_length 3840
```

#### 3. NLP sentence tokenize FactEHR documents and serialize to disk (spaCy DocBin).

Tokenize and sentence split clinical documents using NLP framework ∈ {`medspacy`, `spacy`, `trove`}. See framework [speed benchmarks](docs/nlp_benchmarks.md) for more details. This will generate a spaCy `DocBin` file named `factehr_YYYYMMDD.docbin`.

```bash
python scripts/build_docbin_dataset.py \
--path_to_input data/datasets/corpora/v2/ \
--path_to_output data/datasets/ \
--n_procs 4 \
--batch_size 100 \
--nlp_framework trove \
--file_name_prefix factehr_v2
```

#### 4. Generate fact decomposition prompted dataset using prompt templates and FactEHR documents

Materialize the prompted version of the FactEHR dataset. 

```bash
python scripts/build_fact_decomp_prompted_dataset.py \
--path_to_input data/datasets/factehr_v2.docbin \
--path_to_prompt_dir data/prompt_templates/fact_decomposition/ \
--path_to_output_dir data/datasets/prompted/ \
--file_name_prefix fact_decomposition \
--completion_format messages
```


## 2. Running LLM Experiments

#### 1. Prompt tuning for entailment
Evaluate the performance of entailment, entailment+rationale, and entailment+rationale+CoT for performing entailment. This experiment
leverages shc-gpt-4o (requires SHC VPN) and vertex API (requires Full Traffic VPN). 

First set the following:
```bash
export HUGGINGFACE_HUB_TOKEN={your token here — only needed when running transformers client}
export FACTEHR_DATA_ROOT={path to the data folder}
```

Next download the entailment datasets to the data directory. Because MedNLI comes from Physionet, run this first:

`./scripts/datasets/download_physionet_datasets.sh`

Download the entailment datasets by running:

`./scripts/datasets/download_hf_datasets.sh`

The one dataset that will not be downloaded from the above scripts is FactEHR (v0 — clinician annotated dev set). This is `factehr_dev_set.csv` (release file).

Copy that file into this location: `{$FACTEHR_DATA_ROOT}/datasets/raw/entailment/factehr/factehr.csv`

To run the experiment pipeline, first adjust the config settings here as needed: `scripts/experiments/run_nli_prompt_tuning_experiment.sh`

The most important setting is the client you want to run. For example: 
```bash
models=("medlm-medium") #  "meta-llama/Meta-Llama-3-8B-Instruct"  "gemini-1.5-flash-002"
client="vertex"  # Can be "transformers", "openai-batch", "vertex-batch", "vertex"
```

If using GPT, make sure to define your API endpoint url in `scripts/experiments/run_inference_client.sh`. 

It first launches the job for binary entailment prompts (`max_new_tokens = 25`) followed by the job for rationale entailment prompts (`max_new_tokens=256`).

This command runs the full experiment pipeline. 

`scripts/experiments/run_nli_prompt_tuning_experiment.sh <csv_output_path> <final_metrics_output_path> | tee output.log` 

#### 2. Running fact decomposition and entailment 

Fact decomposition and entailment both involve running prompts through an LLM client. Once you have a prompted dataset (a jsonl file with the prompted inputs), you can run `scripts/experiments/run_inference_client.sh` with a command like

```bash
scripts/experiments/run_inference_client.sh [PATH TO PROMPTED JSONL DATA] [MODEL NAME] [CLIENT NAME] [N TMUX SESSIONS] [MAX OUTPUT TOKENS] [GPT_ENDPOINT URL]
```

If using GPT, make sure to define your API endpoint url in `scripts/experiments/run_inference_client.sh`. 

Example command:
```bash
scripts/experiments/run_inference_client.sh data/datasets/prompted/fact_decomposition_20241009_medalign.jsonl "gemini-1.5-flash-002" "vertex" 5 4000
```

The flow for our experiments is:

1. Generate a prompted dataset for fact decomposition (`scripts/init_all_datasets.sh`)
2. Perform fact decomposition (`scripts/experiments/run_inference_client.sh`)
3. Break down the fact decomposition into entailment pairs (`scripts/experiments/create_entailment_file_from_fact_decomp.sh`)
4. Perform entailment evaluation using LLM as judge (`scripts/experiments/run_inference_client.sh`)

