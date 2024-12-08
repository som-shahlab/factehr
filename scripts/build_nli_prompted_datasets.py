
"""
Build a prompted dataset from a document dataset and a set of prompt templates.
This version is specific to the entailment templates and the following NLI datasets:

SciTail
MedNLI
SNLI
MultiNLI


Example usage:

    python scripts/build_nli_prompted_datasets.py \
        --path_to_prompt_dir ${FACTEHR_DATA_ROOT}/prompt_templates/entailment/ \
        --path_to_output_dir ${FACTEHR_DATA_ROOT}/datasets/prompted/ \
        --dataset_path "$DATASET_PATH" \
        --sample_prob 0.025

"""

import os
import json
import timeit
import numpy as np
import logging
import argparse
from datasets import load_from_disk
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from typing import Dict
from datetime import datetime
import copy
import pandas as pd


########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Generate entailment prompted dataset")
parser.add_argument(
    "--path_to_prompt_dir",
    type=str,
    help="Path to the directory containing prompt templates",
    required=True,
)

parser.add_argument(
    "--path_to_output_dir",
    type=str,
    help="Path to the output directory where the JSONL file will be saved",
    required=True,
)

parser.add_argument(
    "--dataset_path",
    type=str,
    help="path of the dataset to load",
    required=True,
)

parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of the dataset to load (SciTail, MedNLI, etc.)",
    required=True,
)

parser.add_argument(
    "--sample_n",
    type=int,
    help="Sampling number for the datasets",
    default=None
)

parser.add_argument(
    "--sample_prob",
    type=float,
    help="Sampling probability for the dataset",
    default=None
)

parser.add_argument(
    "--split_name",
    type=str,
    help="train, validation, or test split",
    default=None
)



########## Function Definitions ##########

def load_prompt_templates(path_to_prompt_dir: str):
    """Load prompt templates from a directory."""
    prompt_templates = []
    for file in os.listdir(path_to_prompt_dir):
        if file.endswith(".tmpl"):
            with open(os.path.join(path_to_prompt_dir, file), "r") as f:
                text = f.read()
                prompt_templates.append(
                    PromptTemplate.from_template(template=text, name=file.split(".")[0])
                )
    logger.info(f"Loaded {len(prompt_templates)} prompt templates")
    return prompt_templates


def get_batch_jsonl(prompts: Dict[str, Dict[str, str]]):
    """Create OpenAI API batch jsonl data from prompts."""
    batch_json_tmpl = {
        "metadata": {
            "custom_id": None,   
            "label": None     
            }
    }
    data = []

    for uid, prompt_data in prompts.items():
        prompt_text = prompt_data["text"]
        label = prompt_data["label"]
        # Deep copy to avoid modifying the template
        batch_instance = copy.deepcopy(batch_json_tmpl)
        batch_instance["metadata"]["custom_id"] = uid
        batch_instance["metadata"]["label"] = label
        batch_instance["messages"] = [{"role": "user", "content": prompt_text}]
        data.append(batch_instance)
    
    return data


def load_and_sample_dataset(dataset_path: str, sample_n: int, sample_prob: float, split_name: str):
    """Load and sample a fixed number of data points or a fraction of data points from the given dataset saved locally on disk."""

    # Initialize logger
    logger = logging.getLogger(__name__)
    
    # Load the dataset from disk
    if "factehr" in dataset_path.lower():
        dataset = pd.read_csv(os.path.join(dataset_path, "factehr.csv"))
        
        total_samples = len(dataset)
    
        if sample_n:
            if sample_n > total_samples:
                raise ValueError(f"Requested {sample_n} samples, but the dataset only contains {total_samples} samples.")
            
            sampled_df = dataset.sample(n=sample_n, random_state=12345)
        elif sample_prob:
            if not (0 < sample_prob <= 1):
                raise ValueError(f"sample_prob must be between 0 and 1. Provided value: {sample_prob}")
            
            sampled_df = dataset.sample(frac=sample_prob, random_state=12345)
        else:
            raise ValueError("Either `sample_n` or `sample_prob` must be specified.")
        
        # Convert to a list of (index, row) tuples
        samples = [(idx, row) for idx, row in sampled_df.iterrows()]
        
    else:
        dataset = load_from_disk(dataset_path)
    
        # Get the total number of data points in the dataset
        
        # Adjust for multinli split names as per guidance here: http://aclweb.org/anthology/N18-1101
        if "multinli" in dataset_path:
            if split_name == "validation":
                split_name = "validation_matched"
            if split_name == "test":
                split_name = "validation_mismatched"
        
        total_samples = len(dataset[split_name])
        rng = np.random.default_rng(12345)
        
        if sample_n:
            if sample_n > total_samples:
                raise ValueError(f"Requested {sample_n} samples, but the dataset only contains {total_samples} samples.")
            
            # Use RNG to randomly select `sample_n` indices
            sampled_indices = rng.choice(total_samples, size=int(sample_n), replace=False)
            
            # Collect the sampled data points
            samples = [(idx, dataset[split_name][int(idx)]) for idx in sampled_indices]
        
        elif sample_prob:
            if not (0 < sample_prob <= 1):
                raise ValueError(f"sample_prob must be between 0 and 1. Provided value: {sample_prob}")
            
            # Calculate the number of samples based on the sample probability
            sample_n = int(sample_prob * total_samples)
            
            # Use RNG to randomly select `sample_n` indices
            sampled_indices = rng.choice(total_samples, size=sample_n, replace=False)
            
            # Collect the sampled data points
            samples = [(idx, dataset[split_name][int(idx)]) for idx in sampled_indices]
        
        else:
            raise ValueError("Either `sample_n` or `sample_prob` must be specified.")

    logger.info(f"Sampled {len(samples)} data points from dataset at {dataset_path}")
    return samples


def save_prompted_dataset(data, dataset_name, output_dir, file_name_prefix):
    """Save the prompted dataset to a JSONL file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    version_str = datetime.now().strftime("%Y%m%d")
    output_file_path = f"{output_dir}/{file_name_prefix}_{dataset_name}_{version_str}.jsonl"

    with open(output_file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

    logger.info(f"Saved prompted dataset with {len(data)} records to {output_file_path}")

def main():
    # Parse arguments
    args = parser.parse_args()

    # Load prompt templates
    prompt_templates = load_prompt_templates(args.path_to_prompt_dir)
    logger.info(f"Processed prompt templates")

    # Load and sample dataset
    samples = load_and_sample_dataset(args.dataset_path, args.sample_n, args.sample_prob, args.split_name)
    logger.info(f"Loaded dataset")

    # Prepare the prompts
    prompts = {}
    for ptmpl in prompt_templates:
        prompt_id = ptmpl.name
        for idx, item in samples:
            uid = f"{args.dataset_name}|{args.split_name}|{idx}|{prompt_id}"
            
            # Check if the keys are premise/hypothesis or sentence1/sentence2
            if "premise" in item and "hypothesis" in item:
                premise = item["premise"]
                hypothesis = item["hypothesis"]
            elif "sentence1" in item and "sentence2" in item:
                premise = item["sentence1"]
                hypothesis = item["sentence2"]
            elif args.dataset_name.lower() == "factehr":
                premise = item["premise"]
                hypothesis = item["hypothesis"]
            else:
                raise ValueError(f"Unrecognized key format in item {idx}")
            
            # Extract the label
            if args.dataset_name == "mednli" and "gold_label" in item:
                label = int(item["gold_label"] == "entailment")
            elif args.dataset_name == "scitail" and "gold_label" in item:
                label = int(item["gold_label"] == "entailment")
            elif args.dataset_name == "snli" and "label" in item:
                label = int(int(item["label"]) == 0)
            elif args.dataset_name == "multinli" and "label" in item:
                label = int(int(item["label"]) == 0)
            elif args.dataset_name.lower() == "factehr":
                label = int(item["human_label"])  # assuming human_label is binary or convertible to int
            else:
                label = None

            # Format the prompt with the appropriate keys
            prompts[uid] = {
                "text": ptmpl.template.format(premise=premise, hypothesis=hypothesis),
                "label": label
            }


    # Generate batch JSONL data
    batch_data = get_batch_jsonl(prompts)

    # Save to file
    save_prompted_dataset(batch_data, args.dataset_name, args.path_to_output_dir, "entailment")

    logger.info(f"Processed {len(batch_data)} entailment pairs")

if __name__ == "__main__":
    # change this for getting run statistics
    num_runs = 1
    wall_times = np.array(
        [
            timeit.timeit("main()", setup="from __main__ import main", number=1)
            for _ in range(num_runs)
        ]
    )
    logger.info(
        f"Execution time: Mean (SD) = {np.mean(wall_times):.1f} ({np.std(wall_times):.1f}) seconds"
    )
    
