import os
from datasets import load_from_disk
# from ipdb import set_trace
import prompts 
from tqdm import tqdm
import gpt4
import json
import time
from nltk.tokenize import word_tokenize
import pandas as pd
import argparse

print("Imports done")

PROMPT_MAP = {
    prompts._prompt1_icl : "PROMPT1_ICL", 
    prompts._prompt2_icl : "PROMPT2_ICL",
    prompts._prompt1 :  "PROMPT1", 
    prompts._prompt2 : "PROMPT2" 
}

root_path = "/Users/akshayswaminathan/Documents/radgraph/"
source_names = ["mimiciii"] #["medalign", "mimiciii", "coral"]
dataset_names = ["radiology_report"] #, "breastca", "pdac"
output_path_root = "/Users/akshayswaminathan/Documents/radgraph/gpt4_inference"

output_file_path = 'output.txt'

def total_token_calc(note):

    len_of_words = len(word_tokenize(note))
    total_tokens = round(len_of_words*(4/3), 0)

    return int(total_tokens)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument("model")
    args = parser.parse_args()
    model_name = args.model #"GPT4" #"shc-gpt-4o"
    for source_name in source_names: 
        for dataset_name in dataset_names:

            # data = os.path.join(root_path, f"{source_name}/{dataset_name}.csv")
            data = os.path.join(root_path, f"{source_name}/{dataset_name}.hf")
            try:
                dataset = load_from_disk(data)['test'].select_columns(["text"])
            except:
                continue
            # dataset = pd.read_csv(data)            
            
            output_path = f"{output_path_root}/{model_name}/{source_name}/{dataset_name}/"
            
            print("data loaded")

            for prompt in PROMPT_MAP.keys():

                full_output_path = os.path.join(output_path, f"{PROMPT_MAP[prompt]}.json")
                if os.path.exists(full_output_path):
                    print(f"Skipping becasue {full_output_path} already exists")
                    continue
        
                start = time.time()
                
                output_dict = {}
                # for idx, data_point in tqdm(dataset.iterrows(), total=len(dataset)):
                    # print(idx)
                    # print(data_point)
                for idx, data_point in enumerate(tqdm(dataset)): 
                    print(idx)
                    print(data_point)
                    try:
                        note = data_point['text']
                        if "coral" in source_name:
                            note = ' '.join(note)
                    except:
                        note = data_point['note_text']
                        if "coral" in source_name:
                            note = ' '.join(note)
                    # context length issue. 
                    total_tokens = total_token_calc(note)
                    if total_tokens > 32768:
                        print("Number of tokens greater than 32k")

                    #full_note = ' '.join(note)
                    time.sleep(1)
                    output = gpt4.query_gpt4(note, prompt, model_name)

                    idx = int(idx)
                    if idx not in output_dict:
                        output_dict[idx] = []
                    output_dict[idx].append(
                            {
                                "inputs": note,
                                "full_pred_text": output,
                            }
                        )


                end = time.time()
                
                isExist = os.path.exists(output_path)
                if not isExist:
                    print("creating path")
                    os.makedirs(output_path)
                with open(
                    os.path.join(output_path, f"{PROMPT_MAP[prompt]}.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(output_dict, f, indent=4)

                with open("output.txt", "a") as f:

                    print(f"Output path :", {output_path}, file=f)
                    print(f"Saved file : {PROMPT_MAP[prompt]}.json", file=f)
                    
                    print("Minutes since epoch =", (end - start) / 60, file=f )
                    print("--------\n")

                f.close()

if __name__ == "__main__":
    main()


