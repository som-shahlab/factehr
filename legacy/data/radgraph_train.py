import os 
import json
from ipdb import set_trace
import random
#from datasets import Dataset, load_from_disk, DatasetDict
import pandas as pd

path = "/share/pi/nigam/rag-data/results/gemini/radgraph/PROMPT2_ICL.json"

# read this json and sample 250 and make a copy and move to train data outputs -- 

output_path = "/share/pi/nigam/rag-data/training_data/gemini/mimiciii/radiology_report/PROMPT2_ICL.json"

trian_data = []

with open(path, "r") as json_file:
    data = json.load(json_file)

# Step 1: Randomly select 250 keys
selected_keys = random.sample(list(data.keys()), 250)

# Step 2: Create a new dictionary with selected key-value pairs
new_dict = {key: data[key] for key in selected_keys}

with open(
    os.path.join(output_path), "w", encoding="utf-8"
) as f:
    json.dump(new_dict, f, indent=4)
