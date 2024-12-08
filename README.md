# Fact Decomposition of Clinical Notes [PRIVATE]

**Shah Lab internal code** to replicate the generation and LLM-as-a-judge evaluation of the FactEHR dataset. 

## Table of Contents
- [I. Installation](#i-installation)
- [II. Data Dependencies](#ii-data-dependencies)
- [III. Experiment Pipeline [WIP]](#iii-experiment-workflow-wip)
- [IV. Experiment Runtimes & Costs](docs/runtimes.md)

## I. Installation


Use editable mode during development.

```bash
python -m pip install -e .
```

To run all unit tests in `tests/` run this from the project root

```bash
pytest
```

## II. Data Dependencies

See [detailed overview](docs/dataset_summary.md) of the FactEHR dataset. 

### 1. Legacy `FactEHR` (v1)  

> [!CAUTION]
> *This dataset version has several problems, including incorrect MedAlign note sampling, undersampling of available UCSF notes, and biased sampling of MIMIC-III.*
> 
> - **Source Docs**: `carina:/share/pi/nigam/rag-the-facts/datasets/sentences`
> - **LLM Generations**:  `carina:/share/pi/nigam/rag-data/entailment_final`
> - **Annotations**: `carina:/share/pi/nigam/datasets/rag-data//annotations/`

### 2. Refactored `FactEHR` (v2)

- **Source Datasets**: `carina:/share/pi/nigam/projects/just_the_facts/data/datasets/raw/`
- **Sampled Docs**: `carina:/share/pi/nigam/projects/just_the_facts/data/datasets/corpora/`
- **LLM Generations**: `carina:/share/pi/nigam/projects/just_the_facts/data/datasets/completions/`
- **Annotations**: `TBD`
 
### 3. Release `FactEHR` (v3)

- **TODO** Replace `starr-confidential` MedAlign sampled notes with `starr-release-dua`
 
## III. Experiment Pipeline [WIP]

### 1. Sample & Preprocess Documents

#### A. Quick Start

This bash script will run all data preprocessing steps. See [complete breakdown of subtasks](docs/experiments.md#1-sample--preprocess-documents). 

```bash
export FACTEHR_DATA_ROOT=/share/pi/nigam/projects/just_the_facts/data/datasets/
export FACTEHR_LEGACY_DOCS=/share/pi/nigam/rag-the-facts/datasets/sentences/

scripts/init_all_datasets.sh $FACTEHR_DATA_ROOT $FACTEHR_LEGACY_DOCS
```

#### B. Conceptual Pipeline

1. Download datasets and export to CSV.
2. Stratify by note type and unform random sample documents up to budget `k=2000` of documents.
3. NLP preprocess documents (e.g., sentence tokenization) and materialize as a single, versioned (i.e., date-stamped) experiment dataset.
4. Define 4 Fact Decomposition task `PromptTemplate` objects
5. Materialize prompts (`PromptTemplate` applied to document) to create our prompted dataset, `FactDecomp`, stored as a `JSONL` file in a [Chat Completions API](https://platform.openai.com/docs/guides/text-generation)-style format

### 2. Run LLM Experiments

#### A. Quick Start

```
TBD
```


#### B. Conceptual Pipeline

- Run `FactDecomp` prompted dataset JSONL in all 4 LLMs {GPT-4, GPT-4o, MedLM, GeminiPro-1.5, Llama3-8b} to generate fact lists.
- Postprocess and split fact lists
- Materialize entailment pairs, `FactEntail` for computing Fact Recall and Fact Precision
- Run `FactEntail` with {Llama3-8b} 
- Sample pairs up to budget `n=3000` for clinical review

#### C. Pipeline Subtasks

- [ ] Run LLM fact decomposition inference on all documents and serialize to disk

Specific code to run this depends on the API

- **AzureOpenAI** [`scripts/run_azure_openai_fact_decomp_jobs.sh`](scripts/run_azure_openai_fact_decomp_jobs.sh) 
- **VertexAI** `TBD`
- **HuggingFace** `TBD`

- [ ] NLP sentence tokenize fact list and serialize to disk
- [ ] Benchmark existing NLI datasets (MedNLI, SNLI, MultiNLI, SciTail) on SOTA LLMs
- [ ] Generate all entailment pairs for fact precision (`I[note ⇒ fact]`) and fact recall (`I[fact-list ⇒ sentence]`) and serialize to disk
- [ ] Run LLM-as-a-judge inference on all entailment pairs and serialize raw generation outputs to disk

### 3. Score Experiments

- [x] Compute entailment scores given LLM outputs and write to TSV

```bash
python scripts/score_entailment_pairs.py \
--path_to_input <ENTAILMENT_LLM_OUTPUT_DIR> \
--path_to_output data/manuscript/entailment_pair_scores.tsv \
--map_unparseable_json 0
```

### 4. Generate Paper Tables/Figures

```bash
python scripts/create_latex_tables.py 
```

## IV. Experiment Runtimes & Costs

See current estimates [here](docs/runtimes_and_costs.md)