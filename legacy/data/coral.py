import os
import pandas as pd
import nltk
from ipdb import set_trace
from nltk.tokenize import sent_tokenize
from datasets import Dataset, load_from_disk, DatasetDict

def split_text_into_chunks(text, chunk_size=10):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Group the sentences into chunks of size chunk_size
    chunks = [sentences[i:i+chunk_size] for i in range(0, len(sentences), chunk_size)]   
    # If there are any remaining sentences, append them as the last chunk
    if len(sentences) % chunk_size != 0:
        remainder = sentences[len(chunks)*chunk_size:]
        if remainder != []:
            chunks.append(remainder)

    return chunks

# Function to process each text file
def process_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
        #chunks = split_text_into_chunks(text, chunk_size=10)
        return text

# Path to the directory containing the .txt files
directory_path = '/share/pi/nigam/rag-data/physionet.org/files/curated-oncology-reports/1.0/coral/annotated/breastca'

# Initialize an empty list to store all the chunk data
chunk_data = []

# Iterate through each .txt file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt') and "15" not in filename and "16" not in filename and "17" not in filename and "18" not in filename and "19" not in filename:
        file_path = os.path.join(directory_path, filename)
        # Process each text file
        text = process_text_file(file_path)
        # Append the chunk data to the list
        chunk_data.append({"id": filename,"text": text})

        # for i, chunk in enumerate(chunks, 1):
        #     chunk_data.append({"chunk_number": i,"text": chunk})

# Save the DataFrame to a CSV file
set_trace()
print(len(chunk_data))
test_set = Dataset.from_list(chunk_data)
dataset = DatasetDict({"train": test_set})
dataset.save_to_disk("/share/pi/nigam/rag-data/coral/breast_full.hf")

# Didnt run this yet -- should be okay to run the full set