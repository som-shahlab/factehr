
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
./scripts/datasets/download_shc_datasets.sh
./scripts/datasets/download_hf_datasets.sh
```

#### 2. Sample documents from MIMIC-CXR, MIMIC-III, MedAlign, CORAL to create FactEHR dataset

>[!IMPORTANT]  
> **LEGACY** Link original data to the legacy sampled documents, assign primary key and export CSVs for preprocessing.
>
> ```bash
> python scripts/hotfixes/get_note_provenance.py \
> --path_to_legacy $FACTEHR_LEGACY_DOCS \
> --path_to_datasets data/datasets/ \
> --path_to_output data/datasets/
> ```

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
--path_to_input data/datasets/factehr_v2_20240825.docbin \
--path_to_prompt_dir data/prompt_templates/fact_decomposition/ \
--path_to_output_dir data/datasets/prompted/ \
--file_name_prefix fact_decomposition \
--completion_format messages
```


## 2. Running LLM Experiments

### 🚧 Status: Subtasks
- ✅ Run LLM fact decomposition inference on all documents and serialize to disk
- ✅ Prompt tuning for entailment
- ✅ NLP sentence tokenize fact list and serialize to disk
- ✅ Benchmark existing NLI datasets (MedNLI, SNLI, MultiNLI, SciTail) on SOTA LLMs
- ✅ Generate all entailment pairs for fact precision (`I[note ⇒ fact]`) and fact recall (`I[fact-list ⇒ sentence]`) and serialize to disk
- ✅ Run LLM-as-a-judge inference on all entailment pairs and serialize raw generation outputs to disk

#### 1. Prompt tuning for entailment
Evaluate the performance of entailment, entailment+rationale, and entailment+rationale+CoT for performing entailment. This experiment
leverages shc-gpt-4o (requires SHC VPN) and vertex API (requires Full Traffic VPN). 

First set the following:
```bash
export HUGGINGFACE_HUB_TOKEN={your token here — only needed when running transformers client}
export FACTEHR_DATA_ROOT={something like /share/pi/nigam/akshays/just-the-facts/data/}
```

Next download the entailment datasets to the data directory. Because MedNLI comes from Physionet, run this first:

`./scripts/datasets/download_physionet_datasets.sh`

Download the entailment datasets by running:

`./scripts/datasets/download_hf_datasets.sh`

The one dataset that will not be downloaded from the above scripts is FactEHR (v0 — clinician annotated dev set). This is currently saved on carina here:

`/share/pi/nigam/akshays/just-the-facts/data/datasets/raw/entailment/factehr.csv`

Copy that file into this location: `{$FACTEHR_DATA_ROOT}/datasets/raw/entailment/factehr.csv`

As of 10/2/24, the following path on carina contains all NLI test sets needed for this experiment — instead of compiling the datasets locally you can copy over the contents of this folder into your directory: `/share/pi/nigam/akshays/just-the-facts/data/datasets/raw/entailment/`

To run the experiment pipeline, first adjust the config settings here as needed: `scripts/experiments/run_nli_prompt_tuning_experiment.sh`

The most importatnt setting is the client you want to run. For example: 
```bash
models=("medlm-medium") #  "meta-llama/Meta-Llama-3-8B-Instruct"  "gemini-1.5-flash-002"
client="vertex"  # Can be "transformers", "openai-batch", "vertex-batch", "vertex"
```

It first launches the job for binary entailment prompts (`max_new_tokens = 25`) followed by the job for rationale entailment prompts (`max_new_tokens=256`).

This command runs the full experiment pipeline. 

`scripts/experiments/run_nli_prompt_tuning_experiment.sh <csv_output_path> <final_metrics_output_path> | tee output.log` 

