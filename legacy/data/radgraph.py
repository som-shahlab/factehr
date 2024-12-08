import os 
import json
from ipdb import set_trace
import random
#from datasets import Dataset, load_from_disk, DatasetDict
import pandas as pd

path = "/share/pi/nigam/users/monreddy/physionet.org/files/radgraph/1.0.0/train.json"
cxr_splits = "/share/pi/nigam/rag-data/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv"

# def extract_random_keys(input_dict, num_keys=100):
#     all_keys = list(input_dict.keys())
#     random_keys = random.sample(all_keys, min(num_keys, len(all_keys)))
#     extracted_dict = {key: input_dict[key] for key in random_keys}
#     return extracted_dict

train_set = []

with open(path, "r") as json_file:
    data = json.load(json_file)


extracted_dict = data
# for each datapoint


df2 = pd.read_csv(cxr_splits)

for key, data_point in extracted_dict.items():
    note = data_point['text']
    key = key[:-4]
    key = key.split("/p")[1]
    p_id = key.split("/s")[0]
    s_id = key.split("/s")[1]

    train_set.append({"study_id": s_id, "subject_id": p_id, "split": "train"})

    # if "findings" in note.lower(): 
    #     anno_note = note.lower().split('findings')[1]
    #     anno_note = "findings" + anno_note
    # elif "impression" in note.lower(): 
    #         anno_note = note.lower().split('impression')[1]
    #         anno_note = "impression" + anno_note
    # else:
    #     set_trace()

    #test_set.append({"id": key, "text": note})

df1 = pd.DataFrame(train_set)

df1['study_id'] = df1['study_id'].astype(str)
df2['study_id'] = df2['study_id'].astype(str)

df1['subject_id'] = df1['subject_id'].astype(str)
df2['subject_id'] = df2['subject_id'].astype(str)

set_trace()
merged_df = pd.merge(df1, df2, on=['study_id', 'subject_id'], suffixes=('_df1', '_df2'))


#print(len(test_set))
# test_set = Dataset.from_list(test_set)
# dataset = DatasetDict({"train": test_set})
# dataset.save_to_disk("/share/pi/nigam/rag-data/radgraph/train.hf")


