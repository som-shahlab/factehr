import os
from datasets import load_from_disk
from ipdb import set_trace
from tqdm import tqdm
import spacy
from nltk.tokenize import word_tokenize
import nltk.data
from spacy.tokens import DocBin
from tokenizer import ct_tokenizer 
from ipdb import set_trace
from sbd import ct_sentence_boundaries

root_path = "/share/pi/nigam/rag-data/"
source_names = ["mimiciii"]
#dataset_names = ["discharge_summary", "nursing_note", "progress_note"]
dataset_names = ["radiology_report"]

output_file_path = 'output.txt'

def total_token_calc(note):

    len_of_words = len(word_tokenize(note))
    total_tokens = round(len_of_words*(4/3), 0)

    return int(total_tokens)

def sent_tokenizer(text):
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sents = []
    for sent in sent_detector:
        sents.append(sent)
    return sents

def split_notes(raw_note):
# NLP pipeline
    nlp = spacy.blank("en")
    nlp.tokenizer = ct_tokenizer(nlp)
    nlp.add_pipe('clinical_text_light_sbd')
            
    doc_tuples = nlp.pipe([raw_note])
 
    sentences = [
    each_sent
    for sents in doc_tuples
    for sent in sents
    for each_sent in sent.text.split('\n')
    if each_sent
    ]
    
    return sentences

def main():
    for source_name in source_names: 
        for dataset_name in dataset_names:

            data = os.path.join(root_path, f"{source_name}/{dataset_name}.hf")
            pred_set = load_from_disk(data)['test'].select_columns(["text"])
            #dataset = pd.read_csv(data)     
        
            for idx, data_point in enumerate(tqdm(pred_set)): 
                note = data_point['text']
                if isinstance(note, list):
                    note = " ".join(note)

                # context length issue. 
                total_tokens = total_token_calc(note)
                if total_tokens > 32768:
                    print("Number of tokens greater than 32k")

                spacy = split_notes(note)
                nltk = sent_tokenizer(note)

                set_trace()
    
            with open("output.txt", "a") as f:

                print(f"Note :", {note}, file=f)
                print("--------")
                print(f"Spacy:", {spacy}, file=f)
                print("--------")
                print(f"NLTK:", {nltk}, file=f)
                print("--------\n")

            f.close()

            

if __name__ == "__main__":
    main()


