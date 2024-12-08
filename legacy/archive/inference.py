import argparse
import json
import os
import pdb
import sys
from ipdb import set_trace
import torch
import tqdm
from lightning import Fabric, Trainer, seed_everything
from prompts import Prompts
import uuid
from typing import Union
import time

torch.set_float32_matmul_precision("high")

from modules.utils import (
    load_data_module,
    load_model_and_tokenizer,
)

sys.path.append("/share/pi/nigam/users/monreddy/rag-the-facts/modules")


def inference(
    dataset_name: str,
    note_type: str, 
    model_path: str,
    root_path: str,
    output_path: str, 
    prompt: Union[Prompts, str],
    file_name:str, 
    seed: int = 42,
    total_samples: int = None,
    model=None,
    tokenizer=None,
    model_config=None
):
    if isinstance(prompt, Prompts):
        #output_filename = f"{prompt.name}.json"
        instruction = prompt.value
    else:
        instruction = prompt
        #output_filename = f"{str(uuid.uuid4())}.json"

    output_filename = file_name + ".json"
    print("Output file name:", output_filename)
    print("Instruction: \n", instruction) 

    full_output_path =  os.path.join(output_path, output_filename)
    if os.path.exists(full_output_path):
        print(f"Skipping becasue {full_output_path} already exists")
        return

    torch.set_float32_matmul_precision("high")
    if seed is not None:
        seed_everything(seed)

    if not os.path.exists(path=output_path):
        os.makedirs(output_path)

    if model is None or tokenizer is None or model_config is None:
        model, tokenizer, model_config = load_model_and_tokenizer(
            model_path
        )

    data_module = load_data_module(
        dataset_name=dataset_name,
        note_type=note_type, 
        tokenizer=tokenizer,
        root_path=root_path,
        model_config=model_config,
        total_samples=total_samples,
    )

    data_module.customize_instructions(instruction=instruction)
    data_module.prepare_data()

    print("Start inference...")
    start = time.time()
    
    fabric = Fabric(accelerator="gpu", devices=1, precision="bf16-mixed")
    dataloader = data_module.predict_dataloader()
    model = fabric.setup(model)
    dataloader = fabric.setup_dataloaders(dataloader)
    # set eval mode
    model.eval()
    all_full_pred_text = []
    all_idxs = []
    all_input_text = []

    for batch in tqdm.tqdm(dataloader, desc="Inference"):
        
        idxs = batch["idx"]
        input_text = batch["input_text"]

        batch_full_text = {
            "input_ids": batch["input_ids"], 
            "attention_mask": batch["attention_mask"],
            "idx": batch["idx"],
        }
        _, _, pred_ids = model.predict_step(batch_full_text)
        full_preds_text = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        full_preds_text = [
            full_preds_text[i].replace(
                batch["input_text"][i].replace("<s>", ""), ""
            )
            for i in range(len(full_preds_text))
        ]
        all_input_text.extend(input_text)
        all_full_pred_text.extend(full_preds_text)
        all_idxs.extend(idxs.cpu().numpy())

        set_trace()

    output_dict = {}

    for idx, input_text, full_pred_text in zip(
        all_idxs,
        all_input_text,
        all_full_pred_text,
    ):
        idx = int(idx)
        if idx not in output_dict:
            output_dict[idx] = []
        output_dict[idx].append(
            {
                "inputs": input_text,
                "full_pred_text": full_pred_text,
            }
        )
    

    end = time.time()
    
    with open("output.txt", "a") as f:
        end = time.time()
        print("Minutes since epoch =", (end - start) / 60, file=f )
        print(f"Output stored at {full_output_path}.json")

    f.close()

    with open(
        full_output_path, "w", encoding="utf-8"
    ) as f:
        json.dump(output_dict, f, indent=4)

    
   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="radgraph"
    )
    parser.add_argument(
        "--model_path", type=str, default="/share/pi/nigam/pretrained/llama-2-13b-chat_huggingface"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_samples", type=int, default=None)

    parser.add_argument("--root_path", type=str, default="/share/pi/nigam/rag-data/")
    parser.add_argument("--output_path", type=str, default="/share/pi/nigam/rag-data/sample_results/")
    parser.add_argument(
        "--instruction",
        type=str,
        default="Please breakdown the following text into independent facts: ",
    )
    parser.add_argument(
        "--all-instructions",
        action=argparse.BooleanOptionalAction
    )

    args = parser.parse_args()

    if args.all_instructions:
        for instruction in Prompts:
            inference(
                dataset_name=args.dataset_name,
                note_type=args.note_type, 
                model_path=args.model_path,
                seed=args.seed,
                total_samples=args.total_samples,
                root_path=args.root_path,
                file_name=args.file_name,
                output_path=args.output_path,
                prompt=instruction
            )
    else:
        inference(
            dataset_name=args.dataset_name,
            model_path=args.model_path,
            note_type=args.note_type, 
            seed=args.seed,
            total_samples=args.total_samples,
            file_name=args.file_name, 
            root_path=args.root_path,
            output_path=args.output_path,
            prompt=args.instruction
        )


# def debug():
#     if os.environ.get("DEBUG") == "1":
#         set_trace()


if __name__ == "__main__":
    main()
