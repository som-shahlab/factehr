import os 
import json
from ipdb import set_trace
import random
from datasets import Dataset, load_from_disk, DatasetDict
import pandas as pd


#path = "/share/pi/nigam/rag-data/results/{model_name}/{source_name}/{dataset_name}/"

path_to_test = "/share/pi/nigam/rag-data/cxr_splits/test/"

data = pd.DataFrame(columns=['note_id', 'note_text'])
  
note_id = []
note_text = []
# iterate through all file 
for file in os.listdir(path_to_test): 
    # Check whether file is in text format or not 
    if file.endswith(".txt"): 

        if len(note_id) < 500:
            file_path = f"{path_to_test}{file}"

            with open(file_path, 'r') as f: 
                note = f.read()
            
            file_name = file[:-4]
        
            note_id.append(file_name)
            note_text.append(note)
            print("count")  
        
        else:
            break



dataset_dict = {
        'id': note_id,
        'text': note_text
        # Add more columns as needed
    }

# Create a Hugging Face Dataset
dataset = Dataset.from_dict(dataset_dict)
dataset = DatasetDict({"test": dataset})
dataset.save_to_disk(f"/share/pi/nigam/rag-data/mimiciii/radiology_report.hf")


