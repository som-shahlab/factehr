from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import transformers
import pandas as pd
import json
import re
import time
import os
import argparse
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Strip the newline character and parse the JSON
            data.append(json.loads(line.strip()))
    return data

def main():
    parser = argparse.ArgumentParser(description="Process JSON files for entailment.")
    parser.add_argument('model_name', type=str, help="Input directory containing JSON files.")
    parser.add_argument('out_path', type=str, help="Output directory to save results.")
    parser.add_argument('dataset_path', type=str, help="Output directory to save results.")
    args = parser.parse_args()

    print("Processing...")
    MODEL_ID = args.model_name
    MEDNLI_PATH = args.dataset_path 

    if 'deberta' in MODEL_ID:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        for param in model.parameters():
            # Check if parameter dtype is  Float (float32)
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.float16)
        HF_PIPELINE = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0)
    elif 'gemini' in MODEL_ID or 'MedLM' in MODEL_ID:
        HF_PIPELINE = None     
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        HF_PIPELINE = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)
        
    data = read_jsonl(MEDNLI_PATH)
    mednli_df = pd.DataFrame(data)

    mednli_df['pred_label'] = mednli_df.apply(lambda row: check_sentence_entailment(row['sentence1'], 
                                                                                    row['sentence2'], 
                                                                                    pipeline=HF_PIPELINE, 
                                                                                    model_id=MODEL_ID), axis=1)

    mednli_df.to_csv(args.out_path)
    

if __name__ == "__main__":
    main()
    
    
# ds = load_dataset("stanfordnlp/snli")

# # Function to rename keys in the dataset
# def rename_keys(example):
#     return {
#         "sentence1": example["premise"],
#         "sentence2": example["hypothesis"],
#         "gold_label": example["label"]
#     }

# # Apply the renaming function to the dataset
# renamed_ds = ds.map(rename_keys, remove_columns=["premise", "hypothesis", "label"])

# # Save the dataset as JSONL
# output_file = "/share/pi/nigam/rag-data/mednli/snli.jsonl"
# with open(output_file, 'w') as f:
#     for example in renamed_ds['test']:  # You can change 'train' to the appropriate split if needed
#         json.dump(example, f)
#         f.write('\n')

#####