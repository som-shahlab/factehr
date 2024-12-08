import pandas as pd
import json
import os
from ipdb import set_trace
import gc 
import writer
import csv

def process_entailment_results(out_path, prompt):
    """
    Code to calcuate wiyh 
    """
    files = []
    precision_file = os.path.join(out_path, f"{prompt}.json")
    recall_file = os.path.join(out_path, f"{prompt}.json")
    if not os.path.exists(precision_file):
        files.append(precision_file)
        set_trace()
    
    if not os.path.exists(recall_file):
        files.append(recall_file)
        set_trace()
        
    return files


def process_all_models(root_path):
    """
    Process all model folders in the root path to calculate and save entailment results.

    Args:
        root_path (str): Root directory containing subfolders for each model.
    """
    full_file_list = []
    for model_folder in os.listdir(root_path):
        model_path = os.path.join(root_path, model_folder)
        if not os.path.isdir(model_path):  # Skip if not a directory
            continue
        print(f"Processing model folder: {model_folder}")
        for dataset_folder in os.listdir(model_path):
            dataset_path = os.path.join(model_path, dataset_folder)
            print(f"Processing dataset folder: {dataset_path}")
            for note_folder in os.listdir(dataset_path):
                note_path = os.path.join(dataset_path, note_folder)
                print(f"Processing note type folder: {note_path}")
            
                for filename in os.listdir(note_path):
                    # note_path = os.path.join(model_path, note_folder)
                    if filename.endswith('.json'):
                        prompt = filename.split('.json')[0] 
                        if os.path.isdir(note_path):
                            try:
                                print("Processing " + prompt)

                                files_list = process_entailment_results(note_path, prompt)
                                full_file_list.extend(files_list)
                            except FileNotFoundError as e:
                                print(f"Error processing {model_folder}: {e}")

                
    return full_file_list



def main():

    in_path = "/share/pi/nigam/rag-data/entailment_final/"
    out_path = "/share/pi/nigam/users/monreddy/rag-the-facts/status.csv"

    

    status_list = process_all_models(in_path)

    set_trace()
    writer=csv.writer(open(out_path,'wb'))
    
    for path in status_list:
        writer.writerow([path])


if __name__ == "__main__":
    main()
