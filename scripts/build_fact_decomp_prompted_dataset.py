"""
Build a prompted dataset from a document dataset and a set of prompt templates.
This version is specific to the FactGen dataset and the Fact Decomposition prompt templates.

TODO Make this more general somehow to support pairing (dataset, prompt_template)

Example usage:

python scripts/build_fact_decomp_prompted_dataset.py \
--path_to_input data/datasets/factehr_v2_20240827.docbin \
--path_to_prompt_dir data/prompt_templates/fact_decomposition/ \
--path_to_output_dir data/datasets/prompted/ \
--file_name_prefix fact_decomposition \
--completion_format messages

=================================================================
# FactEHR v2  data/datasets/factehr_v2_20240824.docbin
=================================================================
INFO - Range                Count      Frequency
INFO - [94.00 - 491.80)     3733       43.0%
INFO - [491.80 - 889.60)    1637       18.8%
INFO - [889.60 - 1287.40)   501        5.8%
INFO - [1287.40 - 1685.20)  511        5.9%
INFO - [1685.20 - 2083.00)  468        5.4%
INFO - [2083.00 - 2480.80)  485        5.6%
INFO - [2480.80 - 2878.60)  473        5.4%
INFO - [2878.60 - 3276.40)  392        4.5%
INFO - [3276.40 - 3674.20)  310        3.6%
INFO - [3674.20 - 4072.00)  178        2.0%

Total count: 8688

Total count: 8688 (2,172 documents X 4 prompt templates)
"""

import os
import json
import spacy
import timeit
import logging
import tiktoken
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from spacy.tokens import DocBin
from langchain_core.prompts import PromptTemplate
from datasets import Dataset, DatasetDict, load_from_disk

########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("entailment_scores.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Generate prompt dataset")
parser.add_argument(
    "-i",
    "--path_to_input",
    type=str,
    help="Path to the input directory containing files to process",
    # required=True,
)

parser.add_argument(
    "-p",
    "--path_to_prompt_dir",
    type=str,
    help="Path to the directory containing prompt templates",
    # required=True,
)

parser.add_argument(
    "-o",
    "--path_to_output_dir",
    type=str,
    help="Path to the output JSONL file",
)

parser.add_argument(
    "-n",
    "--file_name_prefix",
    type=str,
    help="Prefix name of prompted dataset file",
    default="factehr",
)

parser.add_argument(
    "-f",
    "--completion_format",
    type=str,
    help=(
        "Completion format to use for JSONL file expoert. `messages` is used with "
        "chat completions (most common) and `prompt` with single prompt completions."
    ),
    choices=["messages", "prompt"],
    default="mesages",
)

parser.add_argument(
    "-m", "--max_tokens", type=int, help="maximum tokens per completion", default=4096
)

########## Misc Data Loaders ##########


def numpy_histogram_plot(data, bins):
    """
    Creates a histogram of the data and prints the frequency of each bin along
    with the range, count, and percentage.
    """
    # Calculate the histogram
    hist, bin_edges = np.histogram(data, bins=bins)

    # Calculate the total count
    total_count = sum(hist)

    # Print the histogram with counts and frequencies
    logging.info(f"{'Range':<20} {'Count':<10} {'Frequency':<10}")
    for i in range(len(hist)):
        range_label = f"[{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f})"
        frequency = (hist[i] / total_count) * 100
        logging.info(f"{range_label:<20} {hist[i]:<10} {frequency:.1f}%")

    # Print the total count
    logging.info(f"\nTotal count: {total_count}")


def load_prompt_templates(path_to_prompt_dir: str):
    """Load prompt templates from a directory. Name of prompt is the same
    as the file name without the extension"""
    prompt_templates = []

    for file in os.listdir(path_to_prompt_dir):
        if file.endswith(".tmpl"):
            text = open(os.path.join(path_to_prompt_dir, file), "r").read()
            prompt_templates.append(
                PromptTemplate.from_template(template=text, name=file.split(".")[0])
            )
    logging.info(f"Loaded {len(prompt_templates)} prompt templates")
    return prompt_templates


def load_document_dataset(fpath: str):
    """TODO Make memory efficient"""
    with open(fpath, "rb") as fp:
        bytes_data = fp.read()

    nlp = spacy.blank("en")
    return list(DocBin().from_bytes(bytes_data).get_docs(nlp.vocab))


def main():

    args = parser.parse_args()

    dataset = load_document_dataset(args.path_to_input)
    templates = load_prompt_templates(args.path_to_prompt_dir)

    # load fast tokenizer (just for estimating token count)
    enc = tiktoken.encoding_for_model("gpt-4o")

    prompt_token_lengths = {
        tmpl.name: len(enc.encode(tmpl.template)) for tmpl in templates
    }

    prompt_dataset = []
    token_cache = {}

    for tmpl in templates:
        for doc in dataset:
            doc_id = doc.user_data["doc_id"]
            dataset_name = doc.user_data["dataset_name"]
            note_type = doc.user_data["note_type"]

            # prompt unique id
            uid = "|".join([doc_id, dataset_name, note_type, tmpl.name])
            prompt = tmpl.format(text=doc.text)

            # get token count for prompt
            token_cache[prompt] = (
                token_cache[prompt]
                if prompt in token_cache
                else len(enc.encode(prompt))
            )

            record = {
                "metadata": {
                    "uid": uid,
                    "doc_id": doc_id,
                    "dataset_name": dataset_name,
                    "note_type": note_type,
                    "prompt_template_name": tmpl.name,
                    "n_prompt_tokens": prompt_token_lengths[tmpl.name],
                    "n_tokens": token_cache[prompt],
                }
            }
            if args.completion_format == "messages":
                record["messages"] = [{"role": "user", "content": prompt}]
            elif args.completion_format == "prompt":
                record["prompt"] = prompt
            else:
                raise ValueError("Invalid prompt format")

            record["max_tokens"] = args.max_tokens
            prompt_dataset.append(record)

    # simple histogram plot of token counts
    numpy_histogram_plot(list(token_cache.values()), bins=10)

    # write to file
    if not os.path.exists(args.path_to_output_dir):
        os.makedirs(args.path_to_output_dir)

    version_str = datetime.now().strftime("%Y%m%d")
    with open(
        f"{args.path_to_output_dir}/{args.file_name_prefix}_{version_str}.jsonl", "w"
    ) as f:
        for item in prompt_dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Saved prompts n= {len(prompt_dataset)}")


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
