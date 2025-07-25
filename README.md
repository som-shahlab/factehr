# Fact Decomposition of Clinical Notes

Code to replicate the generation and LLM-as-a-judge evaluation of the FactEHR dataset. 

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

See [here](docs/release_files.md) for summary of release files.
 
## III. Experiment Pipeline 
See [experiments](docs/experiments.md) for an overview of the experiment pipeline.
