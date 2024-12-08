import os 
import json
from ipdb import set_trace
import random
from datasets import Dataset, load_from_disk, DatasetDict
import pandas as pd

note_types = ["discharge_summary", "nursing_note", "progress_note"]


for note_type in note_types: 
    path = f"/share/pi/nigam/rag-data/medalign/{note_type}.csv"

    def extract_random_keys(input_dict, num_keys=100):
        all_keys = list(input_dict.keys())
        random_keys = random.sample(all_keys, min(num_keys, len(all_keys)))
        extracted_dict = {key: input_dict[key] for key in random_keys}
        return extracted_dict

    data = pd.read_csv(path)

    dataset_dict = {
        'id': data['note_id'].tolist(),
        'text': data['note_text'].tolist(),
        # Add more columns as needed
    }

    # Create a Hugging Face Dataset
    dataset = Dataset.from_dict(dataset_dict)
    dataset = DatasetDict({"test": dataset})
    dataset.save_to_disk(f"/share/pi/nigam/rag-data/medalign/{note_type}.hf")


