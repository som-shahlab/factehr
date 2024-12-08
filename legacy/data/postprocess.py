import os
import json
from ipdb import set_trace
import copy
import pandas as pd

def process_json_file(file_path, new_file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        output_data = copy.deepcopy(data)

        for each_key, value in data.items():
        # Check if the "inputs" field exists
            input_text = value[0]['inputs']
            # Find the index of "CLINICAL NOTE" in the inputs and get content after it
            clinical_note_index = input_text.find("CLINICAL NOTE")
            if clinical_note_index != -1:
                clinical_note_content = input_text[clinical_note_index + len("CLINICAL NOTE"):]
                # Update the "inputs" field with content after "CLINICAL NOTE"
                output_data[each_key][0]['inputs'] = clinical_note_content


        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        # Save the modified data back to the same file
        with open(new_file_path, 'w') as modified_json_file:
            json.dump(output_data, modified_json_file, indent=4)


def process_string(input_text):
    # Example string processing function, you can modify this according to your requirements

    if "Note:" in input_text:
        clinical_note_index = input_text.find("Note:")
        if clinical_note_index != -1:
            clinical_note_content = input_text[clinical_note_index + len("Note:"):]
            # Update the "inputs" field with content after "CLINICAL NOTE"
    
    elif "numbered list:" in input_text:
        clinical_note_index = input_text.find("numbered list:")
        if clinical_note_index != -1:
            clinical_note_content = input_text[clinical_note_index + len("numbered list:"):]

    elif "repeat information):" in input_text:
        clinical_note_index = input_text.find("repeat information):")
        if clinical_note_index != -1:
            clinical_note_content = input_text[clinical_note_index + len("repeat information):"):]

    return clinical_note_content  # Convert text to uppercase
    
    
def process_csv_file(file_path, new_file_path):
    df = pd.read_csv(file_path)

    # Find the index of "CLINICAL NOTE" in the inputs and get content after it
    df['clean_prompt'] = df['clean_prompt'].apply(process_string)
    df = df.rename(columns={'clean_prompt': 'inputs'})
    df = df.rename(columns={'gpt4_output': 'full_pred_text'})



    df = df.set_index("note_id")
    df = df.drop(columns=['Unnamed: 0', 'prompt_id'])
    df.to_dict(orient="index")[0]

    new_file_path = new_file_path.replace('.csv', '.json')
    
    output_dict = {key: [value] for key, value in df.to_dict(orient="index").items()}

    set_trace()
    with open(new_file_path, "w") as f:
        json.dump(output_dict, f)

    
    print("Saved at :", new_file_path)

    # os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
    # # Save the modified data back to the same file
    # with open(new_file_path, 'w') as modified_json_file:
    #     json.dump(output_data, modified_json_file, indent=4)


def main():
    # Directory containing JSON files
    directory = "/share/pi/nigam/rag-data/results/gpt4/results"
    new_directory  = "/share/pi/nigam/rag-data/results/gpt4/pdac"

    # Iterate over all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, filename)
            # Process each JSON file
            
            process_json_file(file_path, new_file_path)
    
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(new_directory, filename)
            # Process each JSON file
            
            process_csv_file(file_path, new_file_path)

if __name__ == "__main__":
    main()


