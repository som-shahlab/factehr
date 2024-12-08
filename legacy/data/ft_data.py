import pandas as pd
from ipdb import set_trace
from datasets import Dataset, load_from_disk, DatasetDict
import json
import sys
sys.path.append('/share/pi/nigam/users/monreddy/rag-the-facts/')
from main import prompts
import os

root_path = "share/pi/nigam/rag-data/finetuning_data/"



instruction = prompts._prompt2_icl

source_names = ["mimiciii"]
dataset_names = ["discharge_summary", "nursing_note", "progress_note", "radiology_report"]
#dataset_names = ["radiology_report"]

output_list = []

model_name = "gemini"
for source_name in source_names: 
    for dataset_name in dataset_names:

        path = f"/share/pi/nigam/rag-data/training_data/gemini/{source_name}/{dataset_name}/PROMPT2_ICL.json"

        with open(path) as f:   
            data = json.load(f)

        output_path = f"/share/pi/nigam/rag-data/finetuning_data/{model_name}/{source_name}/"

        
        for key, content in data.items():
            
            output_dict = {}
            input_text = content[0]['inputs']
            output_text = content[0]['full_pred_text']


            output_dict["instruction"] = instruction
            output_dict['input'] = input_text
            output_dict['output'] = output_text

            output_list.append(output_dict)

            # print majorkey
            # for subkey, value in subdict.iteritems():
            #         print subkey, value

    
set_trace()
isExist = os.path.exists(output_path)
if not isExist:
    print("creating path")
    os.makedirs(output_path)
with open(
    os.path.join(output_path, "train_data.json"), "w", encoding="utf-8"
) as f:
    json.dump(output_list, f, indent=4)
    


"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

"""
{
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule."
    },
"""