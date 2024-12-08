from utils import EntailmentDataset, ModelOutput, Message, EntailmentPair, Output, BatchedEntailmentDatasetInstance, extract_json_from_string, data_loader_collate_fn
import transformers
from transformers.pipelines import Pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from tqdm import tqdm
import json
from typing import Any
from argparse import ArgumentParser, Namespace
import os
import gc
from torch.utils.data import DataLoader

from ipdb import set_trace


ACCESS_TOKEN =  os.getenv('HF_TOKEN')

DEFAULT_BATCH_SIZE = 2
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--in_path', type=str, help="Input directory containing JSON files.")
    parser.add_argument('--out_path', type=str, help="Output directory to save results.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for the model.")
    parser.add_argument('--mode', type=str, choices=['precision', 'recall', 'f1'], help="Mode of operation: precision, recall, or f1.")

    return parser.parse_args()


def prompt_model_batch(messages: list[list[Message]], pipeline: Pipeline, batch_size: int = DEFAULT_BATCH_SIZE) -> list:
    """Prompts the model using batchs"""
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
        batch_size=batch_size
    )

    return outputs


def get_clean_model_output(output: dict[str, list[dict[Any, Any]]]) -> ModelOutput:
    """Cleans the output and gets the verdict and explaination."""
    result = extract_json_from_string(output[0]["generated_text"][-1]["content"])

    if result:
        try:
            entailment_prediction = result.get("entailment_prediction")
            explanation = result.get("explanation")
            if entailment_prediction == 1:
                verdict = 1
            else:
                verdict = 0
        except:
            # Handle case where JSON could not be parsed
            verdict = 0
            explanation = "The model's response could not be parsed as JSON."
    else:
        verdict = 0
        explanation = "The model's response could not be parsed as JSON."

    return ModelOutput(verdict=verdict, explanation=explanation)

def write_output_batch(output_batch: list, data_batch: list[EntailmentPair], output_path: str) -> None:
    """Writes the output to the given location"""
    output_file = open(output_path, "a")

    for idx, output in enumerate(output_batch):
        clean_output = get_clean_model_output(output=output)
        input_data = data_batch[idx]

        data_to_write = Output(
            ID=input_data["doc_id"],
            premise=input_data["premise"],
            hypothesis=input_data["hypothesis"],
            entailment_pred=clean_output["verdict"],
            explanation=clean_output["explanation"]
        )

        json.dump(data_to_write, output_file)
        output_file.write("\n")

    output_file.close()


def load_existing_entries(file_path):
    if not os.path.exists(file_path):
        return set()

    with open(file_path, 'r') as f:
        # don't understand this ; change this to if the ID exists (not the numner of idx)
        # test this
        existing_entries = set((json.loads(line)['ID']) for line in f)
    return existing_entries



def run_inference(in_path: str, out_path: str, batch_size: int, model, tokenizer) -> None:
    """Run inference for a given dataset and save the results."""
    print(in_path)
    print(out_path)

    existing_entries = load_existing_entries(out_path) if out_path else set()

    dataset = EntailmentDataset(file_path=in_path, ids_to_discard=existing_entries)
    print(f"Total data points: {len(dataset)}")

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=data_loader_collate_fn)

    model.generation_config.pad_token_id = model.config.eos_token_id

    pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
    pipeline.tokenizer.pad_token_id = model.config.eos_token_id

    oom_instances = 0

    for batch in tqdm(data_loader, desc=f"OOM Errors: {oom_instances} Progress:"):
        batch: BatchedEntailmentDatasetInstance

        try:
            output_batch = prompt_model_batch(messages=batch["message"], pipeline=pipeline, batch_size=batch_size)
        
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            oom_instances += 1
            continue

        write_output_batch(output_batch=output_batch, data_batch=batch["entailment_pair"], output_path=out_path)
    
    print(f"Total OOM errors: {oom_instances}")

def process_all_jsons(in_path, out_path, batch_size, mode, model, tokenizer):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    for filename in os.listdir(in_path):
        # Ech prompt files are run in for loop and only load "mode" file.
        if filename.endswith('.json') and mode in filename:
            json_path = os.path.join(in_path, filename)

            base_filename = os.path.splitext(filename)[0]
            print(base_filename)

            out_file = os.path.join(out_path, f"{base_filename}.json")

            print(f"Calculating {mode}...")

            start = time.time()
            run_inference(in_path=json_path, out_path=out_file, batch_size=batch_size, model=model, tokenizer=tokenizer)
            end = time.time()
            
            print(f"Processed {mode} for {filename} in {end - start} seconds")

def main():
    """Run entailment"""
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto", token=ACCESS_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=ACCESS_TOKEN, padding_side="left")
    
    process_all_jsons(args.in_path, args.out_path, args.batch_size, args.mode, model, tokenizer)  

if __name__ == "__main__":
    main()
