import os
from datasets import load_from_disk
from ipdb import set_trace
import prompts 
from tqdm import tqdm
import vertexrun
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

root_path = "/share/pi/nigam/rag-data/"
source_names = ["opi"]
#dataset_names = ["discharge_summary", "nursing_note", "progress_note"]
dataset_names = ["radiology_report"]

output_file_path = 'output.txt'

def total_token_calc(note):

    len_of_words = len(word_tokenize(note))
    total_tokens = round(len_of_words*(4/3), 0)

    return int(total_tokens)

def main():
    model_name = "gemini"
    for source_name in source_names: 
        for dataset_name in dataset_names:

            data = os.path.join(root_path, f"{source_name}/{dataset_name}.hf")
            pred_set = load_from_disk(data)['test'].select_columns(["inputs"])
            #dataset = pd.read_csv(data)            
            
            output_path = f"/share/pi/nigam/rag-data/results_final/{model_name}/{source_name}/{dataset_name}/"
            
            print("data loaded")

            for prompt in PROMPT_MAP.keys():

                full_output_path = os.path.join(output_path, f"{PROMPT_MAP[prompt]}.json")
                if os.path.exists(full_output_path):
                    print(f"Skipping becasue {full_output_path} already exists")
                    continue
            
                print(source_name)
                print(dataset_name)
                print(PROMPT_MAP[prompt])
                
                start = time.time()
                
                output_dict = {}
                for idx, data_point in enumerate(tqdm(pred_set)): 
                    note = data_point['inputs']
                    # context length issue. 
                    total_tokens = total_token_calc(note)
                    if total_tokens > 32768:
                        print("Number of tokens greater than 32k")

                    #full_note = ' '.join(note)
                    time.sleep(3)

                    if model_name == "medlm":
                        output = vertexrun.medlm(note, prompt)
                    else:
                        output = vertexrun.gemini(note, prompt)

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

