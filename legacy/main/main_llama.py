import os
from datasets import load_from_disk
from ipdb import set_trace
import prompts 
import statistics
from tqdm import tqdm
import llama3 as llama3
import json
import time
from nltk.tokenize import word_tokenize

print("Imports done")

PROMPT_MAP = {
    prompts._prompt1 :  "PROMPT1", 
    prompts._prompt2 : "PROMPT2", 
    prompts._prompt1_icl : "PROMPT1_ICL", 
    prompts._prompt2_icl : "PROMPT2_ICL"
}

# "discharge_summary", "nursing_note", "progress_note"
root_path = "/share/pi/nigam/rag-data/"
model_names = ["llama3"]
source_names = ["coral"]
dataset_names = ["breastca", "pdac"]

output_file_path = 'llama_output.txt'

def total_token_calc(note):

    len_of_words = len(word_tokenize(note))
    total_tokens = round(len_of_words*(4/3), 0)

    return int(total_tokens)

print("data loaded")

def main():
    model_name = "llama3"
    for source_name in source_names: 
        for dataset_name in dataset_names:

            data = os.path.join(root_path, f"{source_name}/{dataset_name}.hf")
            pred_set = load_from_disk(data)['train'].select_columns(["text"])
            #dataset = pd.read_csv(data)            
            
            output_path = f"/share/pi/nigam/rag-data/results_final/{model_name}/{source_name}/{dataset_name}/"
            
            print("data loaded")

            for prompt in PROMPT_MAP.keys():

               
                all_tokens = []
                full_output_path = os.path.join(output_path, f"{PROMPT_MAP[prompt]}.json")
                if os.path.exists(full_output_path):
                    print(f"Skipping becasue {full_output_path} already exists")
                    continue
            

                start = time.time()
                
                output_dict = {}
                for idx, data_point in enumerate(tqdm(pred_set)): 

                    note_split = data_point['text']
                    note = ' '.join(note_split)
                    # context length issue. 
                    total_tokens = total_token_calc(note)
                    all_tokens.append(total_tokens)
                    if total_tokens > 32768:
                        print("Number of tokens greater than 32k")
                        
                    
                    output = llama3.generate(note, prompt)              
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

                means = statistics.mean(all_tokens)
                sums = sum(all_tokens)

                with open("llama3_output.txt", "a") as f:

                    print(f"Output path :", {output_path}, file=f)
                    print(f"Saved file : {PROMPT_MAP[prompt]}.json", file=f)
                    print(f"Average number of tokens :", {means}, file=f)
                    print(f"Total number of tokens :", {sums}, file=f)
                    print("Minutes since epoch =", (end - start) / 60, file=f )
                    print("--------\n")

                f.close()


if __name__ == "__main__":
    main()
