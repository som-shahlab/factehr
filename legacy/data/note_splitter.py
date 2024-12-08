import pandas as pd
import json
import re
import time
import os
import argparse
import spacy
from tqdm import tqdm
import pandas as pd
from spacy.tokens import Doc
from spacy.tokens import DocBin
from tokenizer import ct_tokenizer 
from ipdb import set_trace
from sbd import ct_sentence_boundaries
from nltk.tokenize import word_tokenize
from datasets import load_from_disk


def main():

    IN_PATH="/share/pi/nigam/rag-the-facts/datasets/raw/"
    OUT_PATH="/share/pi/nigam/users/monreddy/rag-the-facts/"
    OUTPUT_PATH="/share/pi/nigam/rag-the-facts/datasets/note_sentences/"

    #source_names = ["mimiciii", "medalign", "coral"]
    source_names = ["coral"]
    # rerun coral 
    note_types = ["radiology_report", "progress_note", "nursing_note", "discharge_summary", "breastca", "pdac"]

    for source_name in source_names: 
        for note_type in note_types:
            
            in_path = os.path.join(IN_PATH, f"{source_name}/{note_type}.hf")
            if not os.path.exists(in_path):
                print(f"Path doesn't exist {in_path}, skipping")
                continue
            
            if "coral" in source_names:
                dataset = load_from_disk(in_path)['train'].select_columns(["text"])
            else:
                dataset = load_from_disk(in_path)['test'].select_columns(["text"])

            
            output_path = f"{OUTPUT_PATH}/{source_name}/{note_type}/"
            
            print("data loaded")

            full_output_path = os.path.join(output_path, "sentences.json")
            if os.path.exists(full_output_path):
                print(f"Skipping becasue {full_output_path} already exists")
                continue

            output_list = []

            for idx, data_point in enumerate(tqdm(dataset)): 

                try:
                    note = data_point['text']
                    if "coral" in source_name:
                        note = ' '.join(note)
                except:
                    note = data_point['note_text']
                    if "coral" in source_name:
                        note = ' '.join(note)


                nlp = spacy.blank("en")
                nlp.tokenizer = ct_tokenizer(nlp)
                nlp.add_pipe('clinical_text_light_sbd')
                        
                doc_tuples = nlp.pipe([note])
        

                sentences = []
                for sents in doc_tuples:
                    for sent in sents:
                        for each_sent in sent.text.split('\n'):
                            each_sent = each_sent.strip()
                            if each_sent != ' ':
                                if each_sent:
                                    sentences.append(each_sent)
                    

            
                output_list.append({
                    'idx': idx,
                    'full note': note, 
                    'sentences': sentences, 
                })
        
                
            isExist = os.path.exists(output_path)
            if not isExist:
                print("creating path")
                os.makedirs(output_path)
            with open(full_output_path, 'a') as f:
                json.dump(output_list, f)
                f.write('\n')
        

if __name__ == "__main__":
    main()