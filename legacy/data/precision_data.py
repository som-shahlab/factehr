import pandas as pd
import json
import re
import os
import spacy
import tqdm
import pandas as pd
from spacy.tokens import Doc
from spacy.tokens import DocBin
from tokenizer import ct_tokenizer 
from ipdb import set_trace
from sbd import ct_sentence_boundaries
from nltk.tokenize import word_tokenize

FILENAMES = [
    "PROMPT1_ICL.json",
    "PROMPT1.json",
    "PROMPT2_ICL.json",
    "PROMPT2.json"
]

PROMPT = """
        You are an expert on natural language entailment. Your task is to deduce whether premise statements entail hypotheses.
         " Only return a '1' if the hypothesis can be fully entailed by the premise. Return '0' if the hypothesis contains information that cannot be entailed by the premise.
         " Also generate an explanation for your answer. Generate the answer in JSON format with the following keys:
         " 'explanation': the reason why the entailment prediction is made, 
         " 'entailment_prediction': 1 or 0, whether the claim can be entailed. 
         " Only return the JSON-formatted answer and nothing else "
         "Premise: Hypothesis:
         """


def total_token_calc(note):

    len_of_words = len(word_tokenize(note))
    total_tokens = round(len_of_words*(4/3), 0)

    return int(total_tokens)

def concat_list(text):
    if isinstance(text, list):
        return ' '.join(text)
    else:
        return text

def split_facts(premise):

    if isinstance(premise, list):
        premise = " ".join(premise)
    elif isinstance(premise, str):
        premise = premise
    else:
        assert("Premise is not str")

    # Remove any leading text like "##  Independent Facts:"
    cleaned_text = re.sub(r'^\s*##\s*.+Facts+:\s*', '', premise, flags=re.IGNORECASE)

    # Split into individual facts based on numbered lists and bullet points
    facts = re.split(r'\n\s*(?:\d+\.|\*)\s*', cleaned_text)
    facts = [fact.strip() for fact in facts if fact.strip()]
    
    return facts
    # once its a single string - write code to split the facts as a numbered list and remove the starting text. 


def check_entailment(data, column1, column2, flag, id_col="ID", output_file=None):

    counter = 0 
    fullset_tokens = 0    

    for item in tqdm.tqdm(data):

        data_point = []
        premise = item[column1]
        hypothesis_full = item[column2]
        
        hypothesis_sentences = split_facts(hypothesis_full)
        sent_counter = 0 
        for hypothesis in hypothesis_sentences:
            
            if isinstance(premise, list):
                premise = " ".join(premise)
            elif isinstance(premise, str):
                premise = premise
            else:
                assert("Premise is not str")
            
            sent_counter += 1 
            data_point.append({
                'number': sent_counter, 
                'premise': premise,
                'hypothesis': hypothesis,
            })


            total_input = PROMPT + premise + hypothesis
            instance_tokens = total_token_calc(total_input)
            counter += 1 

            fullset_tokens += instance_tokens
        
        with open(output_file, 'a') as f:
            json.dump({
                'ID': item[id_col],
                'text': data_point
            }, f)
            f.write('\n')

    return fullset_tokens, counter

def calculate_precision(data, out_path, base_filename):
    """
    For Precision, 
    ground truth is the same as premise - use the original note here; which I saved as inputs.
    comparision col is same as hypothesis - generated facts and need to split these (with numbers or bullet points)
    Don't do any transformation with original note 
    """

    ground_truth_col = "inputs"
    comparison_col = "full_pred_text"
    out_file = os.path.join(out_path, f"{base_filename}_sentence_precision.json")
    print(out_file)

    print(f"Calculating precision...")
    tokens, datapoints = check_entailment(data, ground_truth_col, comparison_col, flag="precision", output_file=out_file)

    return tokens, datapoints
    


def load_json_data(json_path):
    print(json_path)
    with open(json_path, 'r') as file:
        in_data = json.load(file)
    flattened_data = []
    for key, value_list in in_data.items():
        for entry in value_list:
            entry['ID'] = key
            flattened_data.append(entry) 
    return flattened_data

def process_all_jsons(in_path, out_path, filename, mode):
    
    json_path = os.path.join(in_path, filename)
    flattened_data = load_json_data(json_path)
    base_filename = os.path.splitext(filename)[0]

    isExist = os.path.exists(out_path)
    if not isExist:
        print("creating path")
        os.makedirs(out_path)


    out_file = os.path.join(out_path, f"{base_filename}_sentence_{mode}.json")
    if os.path.exists(out_file):
        print(f"output file exits {out_file},  skipping")
        tokens = 0
        datapoints = 0
        return tokens, datapoints

    
    tokens, datapoints = calculate_precision(flattened_data, out_path, base_filename)
    return tokens, datapoints
    
                
def load_existing_entries(file_path):
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as f:
        existing_entries = set(concat_list(json.loads(line)['ID']) + concat_list(json.loads(line)['premise']) + concat_list(json.loads(line)['hypothesis']) for line in f)
    return existing_entries

def main():

    IN_PATH="/share/pi/nigam/rag-data/results_final/"
    OUT_PATH="/share/pi/nigam/users/monreddy/rag-the-facts/"
    OUTPUT_PATH="/share/pi/nigam/rag-the-facts/datasets/sentences/"

    source_names = ["mimiciii", "medalign", "coral"]
    models = ["gemini", "medlm", "GPT4", "llama3", "shc-gpt-4o"]
    note_types = ["radiology_report", "discharge_summary", "progress_note", "nursing_note", "breastca", "pdac"]

    modes = ["precision"]
    out_path = os.path.join(OUT_PATH, "information.txt")
    full_metadata = []

    for model_name in models:
        for source_name in source_names:
            for note_type in note_types:
                for mode in modes:
                    for file in FILENAMES:
                        metadata = []
                        print("Processing...")
                        in_path = os.path.join(IN_PATH, model_name, source_name, note_type) 
                        if not os.path.exists(in_path):
                            print(f"Path doesn't exist {in_path}, skipping")
                            continue
                        # outpath is only a txt file, to which we want to keep appending
                        out_path = os.path.join(OUT_PATH, "information.txt")
                        output_path = os.path.join(OUTPUT_PATH, model_name, source_name, note_type)
                        all_tokens, all_datapoints = process_all_jsons(in_path, output_path, file, mode)

                        
                        full_metadata.append(metadata)
                        metadata.append({
                            'model': model_name,
                            'source name': source_name, 
                            'note type': note_type, 
                            'mode': mode,
                            'file name': file,
                            'tokens': all_tokens,
                            'data points': all_datapoints,
                        })
                
                    if out_path:
                        with open(out_path, 'a') as f:
                            json.dump(full_metadata, f)
                            f.write('\n')
                

if __name__ == "__main__":
    main()
