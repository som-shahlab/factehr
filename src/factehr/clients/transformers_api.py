#
# Make sure to set HF authentication token in environment variable "HUGGINGFACE_HUB_TOKEN"
# ex. export HUGGINGFACE_HUB_TOKEN=hf_xxxxxxxxxx
#
# Example Usage:
#
# python src/factehr/clients/transformers_api.py \
# --path_to_prompted_dataset data/datasets/prompted/fact_decomposition_20240829.jsonl \
# --path_to_output_file debug.jsonl \
# --model_name_or_path "meta-llama/Meta-Llama-3.1-8B-Instruct" \
# --generation_config src/factehr/clients/generation_params.json \
# --dynamic_batching 40_000 \
# --max_generation_length 4096 \
# --generation_multiplier 1.3

import os
import time
import json
import timeit
import argparse
import numpy as np
from typing import List, Set
from multiprocessing import Process, Queue
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb

########## Argparse Setup ##########

parser = argparse.ArgumentParser(
    description="LLM Inference with the Hugging Face transformers API"
)

parser.add_argument(
    "--path_to_prompted_dataset",
    type=str,
    help="Path to the prompted dataset JSONL file",
    required=True,
)

parser.add_argument(
    "--path_to_output_file",
    type=str,
    help="Path to the output JSONL file",
    required=True,
)

parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="",
    default=None,
    required=True,
)

parser.add_argument(
    "--generation_config",
    type=str,
    help="TOML file containing generation parameters",
    default=None,
)

parser.add_argument(
    "--resume",
    type=str,
    help="resume previous job",
)

parser.add_argument(
    "--quantization",
    choices=["8bit", "4bit", "none"],
    default="none",
    help="quantization mode: 8bit, 4bit, or none",
)

parser.add_argument(
    "--attention",
    choices=["flash_attention_2", "eager", "sdpa"],
    default="flash_attention_2",
    help="attention implementation",
)

parser.add_argument(
    "--dynamic_batching",
    default=None,
    type=int,
    help="maximum tokens per batch",
)

parser.add_argument(
    "--uid_string",
    default="uid",
    type=str,
    help="field name for unique identifier",
)

parser.add_argument(
    "--generation_multiplier",
    default=1.3,
    type=float,
    help="generation length multiplier",
)

parser.add_argument(
    "--max_generation_length",
    default=4096,
    type=int,
    help="maximum generation length",
)

########## Data Loaders ##########


def load_jsonl_prompted_dataset(file_path: str, filter_for=None):
    """Load a JSON Lines file into memory"""
    # only include prompts that match the filter
    filter_for = filter_for if filter_for else {}

    with open(file_path, "r") as file:
        for line in file:
            item = json.loads(line)
            if filter_for:
                # Requires a metadata object in the JSONL
                if any(
                    item["metadata"].get(key) in value
                    for key, value in filter_for.items()
                ):
                    yield item
            else:
                yield item


########## Main ##########


def main():

    args = parser.parse_args()
    quantization = args.quantization if args.quantization != "none" else None
    # load canonical generation parameters
    generation_params = json.load(open(args.generation_config, "r"))

    # authenticate with the Hugging Face Hub
    access_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    login(token=access_token)

    # =========================================================================
    # Init the model
    # =========================================================================

    # TODO 8bit currently does not work on Hopper chips (H100)
    if quantization is not None and quantization == "8bit":
        torch_dtype = torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    elif quantization is not None and quantization == "4bit":
        torch_dtype = torch.bfloat16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        torch_dtype = (
            torch.bfloat16
            if args.attention == "flash_attention_2"
            else torch.get_default_dtype()
        )
        quantization_config = None

    model_params = {
        "attn_implementation": args.attention,
        "quantization_config": quantization_config,
        "device_map": "auto",
        "torch_dtype": torch_dtype,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, **model_params
    )

    # =========================================================================
    # Batchify the input data
    # =========================================================================

    def tokenize_message(message):
        # Tokenize the message and return the token length
        return tokenizer(message, return_length=True, truncation=True)["length"]

    def create_batches(messages, max_tokens=40_000):
        batches = []
        current_batch = []
        current_token_count = 0

        for metadata, msg in messages:
            token_count = tokenize_message(msg)[0]

            if current_token_count + token_count > max_tokens:
                # If adding this message exceeds the max_tokens, finalize the current batch
                batches.append(current_batch)
                current_batch = [(metadata, msg)]
                current_token_count = token_count
            else:
                # Otherwise, add the message to the current batch
                current_batch.append((metadata, msg))
                current_token_count += token_count

        # Add the last batch if not empty
        if current_batch:
            batches.append(current_batch)

        return batches

    messages = []
    for item in load_jsonl_prompted_dataset(args.path_to_prompted_dataset):
        metadata = item["metadata"]
        msg = item["messages"]
        msg = tokenizer.apply_chat_template(
            [msg], add_generation_prompt=True, tokenize=False
        )[0]
        messages.append((metadata, msg))

    # sort from shortest to longest and create batches
    messages.sort(key=lambda x: tokenize_message(x[1]))
    batches = create_batches(messages, max_tokens=args.dynamic_batching)

    # =========================================================================
    # Inference over batches
    # =========================================================================

    # heuristic generation length (set based on prior runs)
    generation_multiplier = 1.3

    # Ensure only the save directory exists, not an unintended intermediate directory
    output_directory = os.path.dirname(args.path_to_output_file)
    print(f"Output directory: {output_directory}")
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for batch in batches:
        start_time = time.time()
        metadatas, messages = zip(*batch)
        
        # Tokenize input messages
        inputs = tokenizer(messages, padding="longest", return_tensors="pt")
        
        # Determine maximum new tokens based on input size
        n, m = inputs["input_ids"].shape
        max_new_tokens = min(args.max_generation_length, int(args.generation_multiplier * m))
        inputs = {key: val.to(model.device) for key, val in inputs.items()}
        
        # Generate sequences
        generated_sequences = model.generate(
            **inputs,
            # num_beams=2,
            max_new_tokens=max_new_tokens,
            temperature=generation_params["generation"]["temperature"],
            # top_k=2
            # early_stopping=True
            top_p=generation_params["generation"]["top_p"]
        )
        
        # Decode generated sequences
        # generated_texts = [
        #     tokenizer.decode(g, skip_special_tokens=True) for g in generated_sequences
        # ]
        
        # Retrieve only the generated text without the input
        generated_texts = tokenizer.batch_decode(generated_sequences[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        
        # Process each output and structure it into the expected JSONL format
        for metadata, text in zip(metadatas, generated_texts):
            # Create the full structure as per your example
            completion = [
                {
                    "messages": [{"role": "user", "content": messages[metadatas.index(metadata)]}],
                    "model": args.model_name_or_path,
                    "temperature": generation_params["generation"]["temperature"],
                    "max_tokens": max_new_tokens
                },
                {
                    "choices": [{"message": {"content": text, "role": "assistant"}}]
                },
                {
                    "metadata": metadata  
                }
            ]
            
            # Append the completion to the output file
            print(f"Writing to {args.path_to_output_file}")
            with open(args.path_to_output_file, "a") as file:
                file.write(json.dumps(completion) + "\n")
        
        # Track and print the time taken for each batch
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Batch {n}x{m} max_new_tokens={max_new_tokens} processed in {elapsed_time:.2f} seconds")



if __name__ == "__main__":

    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    print(f"Execution time: {elapsed_time:.2f} seconds")
