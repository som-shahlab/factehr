# Fact Decomposition of Clinical Notes

Code to replicate the generation and LLM-as-a-judge evaluation of the FactEHR dataset. 

📝 [Paper](https://arxiv.org/abs/2412.12422)
💾 [Dataset](https://stanford.redivis.com/datasets/bckk-15p0mwmz7)

**NOTE**
The Stanford Dataset DUA prohibts sharing data with third parties including LLM API providers. We follow the guidelines for responsible use as originally outlined by PhysioNet:
If you are interested in using the GPT family of models, we suggest using one of the following services:
- Azure OpenAI service. You'll need to opt out of human review of the data via this form. Reasons for opting out are: 1) you are processing sensitive data where the likelihood of harmful outputs and/or misuse is low, and 2) you do not have the right to permit Microsoft to process the data for abuse detection due to the data use agreement you have signed.
- Amazon Bedrock. Bedrock provides options for fine-tuning foundation models using private labeled data. After creating a copy of a base foundation model for exclusive use, data is not shared back to the base model for training.
- Google's Gemini via Vertex AI on Google Cloud Platform. Gemini doesn't use your prompts or its responses as data to train its models. If making use of additional features offered through the Gemini for Google Cloud Trusted Tester Program, you should obtain the appropriate opt-outs for data sharing, or otherwise not perform tasks that require the sharing of data.
- Anthropic Claude. Claude does not use your prompts or its responses as data to train its models by default, and routine human review of data is not performed.

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

## II. Data summary

See [detailed overview](docs/dataset_summary.md) of the FactEHR dataset. 

See [here](docs/release_files.md) for summary of release files.
 
## III. Experiment Pipeline 
See [experiments](docs/experiments.md) for an overview of the experiment pipeline.

For running fact decomposition and entailment scoring using an LLM judge, see [here](https://github.com/som-shahlab/factehr/blob/main/docs/experiments.md#2-running-llm-experiments).
