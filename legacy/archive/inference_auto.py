import argparse
import os
import pandas as pd
from inference import inference
from modules.utils import load_model_and_tokenizer
import prompts
from ipdb import set_trace
import time

MODEL_PATHS = {
    # "Llama-2-13b": "/share/pi/nigam/pretrained/llama-2-13b-chat_huggingface",
    # "Llama-2-7b": "/share/pi/nigam/pretrained/llama-2-7b-chat_huggingface",
    "flan-t5-large": "google/flan-t5-large",
    "flan-t5-small": "google/flan-t5-small"
}

DATASETS = {
    1: "medalign",
    #2: "mimiciii"
    #2: "radgraph",
}

NOTE_TYPE = {
    "medalign" : [
        "discharge_summary", 
        "nursing_note", 
        "progress_note"
    ], 
    "mimiciii":[
        "discharge_summary", 
        "nursing_note", 
        "progress_note"
    ]
}

PROMPT_MAP = {
    prompts._prompt1 :  "PROMPT1", 
    prompts._prompt2 : "PROMPT2", 
    prompts._prompt1_icl : "PROMPT1_ICL", 
    prompts._prompt2_icl : "PROMPT2_ICL"
}

all_prompts = [prompts._prompt1, prompts._prompt2, prompts._prompt1_icl, prompts._prompt2_icl]

def inference_all(args):

    model_path = MODEL_PATHS[args.model]
    model, tokenizer, model_config = load_model_and_tokenizer(
        model_path
    )
    datasets = list(args.datasets.values())
    kwargs_list = []

    for instruction in all_prompts: 
        print(instruction)

        for dataset in datasets:
            print(dataset)
            
            note_types = args.note_types[dataset]

            for note_type in note_types:
                output_path = os.path.join(
                    args.root_dir, args.model, dataset, note_type
                )

                kwargs = {
                    "model_path": model_path,
                    "root_path": args.root_dir, 
                    "output_path": output_path,
                    "prompt": instruction,
                    "dataset_name": dataset, 
                    "note_type": note_type,
                    #"truncation_strategy": "split",
                    "file_name": PROMPT_MAP[instruction],
                    "model": model,
                    "tokenizer": tokenizer,
                    "model_config": model_config,
                }
                kwargs_list.append(kwargs)

            print(f"Total number of inferences: {len(kwargs_list)}")

            for i, kwargs in enumerate(kwargs_list):
                if os.path.exists(os.path.join(kwargs["output_path"], "predict_logit.json")):
                    print(f"Skipping inference {i+1}/{len(kwargs_list)}; already exists!")
                    continue

                print(f"Running inference {i+1}/{len(kwargs_list)}")
                try:
                    
                    inference(**kwargs)
                except Exception as e:
                    print(f"Error in inference {i+1}/{len(kwargs_list)}")
                    print(kwargs["dataset_name"])
                    print(kwargs["model_path"])
                    print(e)
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="flan-t5-small",
        choices=[
            "flan-t5-large",
            "flan-t5-small",
        ],
    )

    parser.add_argument("--datasets", type=dict, default=DATASETS)
    parser.add_argument("--note_types", type=dict, default=NOTE_TYPE)

    parser.add_argument(
        "--root_dir", type=str, default="/share/pi/nigam/rag-data/", 
    )
    
    parser.add_argument(
        "--source", type=str, default="medalign", 
    )

    args = parser.parse_args()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir, exist_ok=True)

    inference_all(args)


if __name__ == "__main__":
    main()