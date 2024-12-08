import json
import os
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Doc
import argparse
from langchain_core.prompts import PromptTemplate


# Import functions from your provided code
from factehr.nlp.sbd import *
from factehr.nlp.tokenizer import *

# Register the custom tokenizer
@Language.factory("ct_tokenizer")
def create_ct_tokenizer(nlp, name):
    return ct_tokenizer(nlp)

# Build the spaCy pipeline
def build_pipeline():
    # Load a blank spaCy model
    nlp = spacy.blank("en")
    
    # Add the custom tokenizer
    nlp.tokenizer = ct_tokenizer(nlp)
    
    # Add the custom sentence splitter as a component in the pipeline
    nlp.add_pipe("ct_sentence_splitter", last=True)
    
    return nlp

# Function to process clinical notes and output sentences
def process_clinical_note(note_text):
    # Load the spaCy pipeline
    nlp = build_pipeline()
    
    # Process the input clinical note
    doc = nlp(note_text)
    
    # Extract and print each sentence
    sentences = [sent.text for sent in doc.sents]
    return sentences

def get_note_dataset(dataset_name, notes_dir):
    """
    Load and return a clinical notes dataset with the correct 'note_text' column.
    This function reads a CSV file from the specified directory and automatically 
    identifies the column containing the clinical note text.
    """
    
    # Mapping dataset_name based on pattern matching
    if re.search(r"mimic-iii-1.4", dataset_name, re.I):
        dataset_name = "mimiciii"
    elif re.search(r"medalign", dataset_name, re.I):
        dataset_name = "medalign"
    elif re.search(r"mimic-cxr", dataset_name, re.I):
        dataset_name = "mimic-cxr-2.1.0"
    elif re.search(r"coral", dataset_name, re.I):
        dataset_name = "coral"
    else:
        raise ValueError(f"Dataset not found: factehr_{dataset_name}_notes.csv")

    note_dataset = pd.read_csv(os.path.join(notes_dir, f"factehr_{dataset_name}_notes.csv"))
    
    text_column_name = [
        col for col in note_dataset.columns if re.match(f"^note_text|TEXT$", col)
    ]
    if len(text_column_name) == 0:
        raise ValueError(f"Column name 'note_text|TEXT' not found in dataframe")
    
    note_dataset['note_text'] = note_dataset[text_column_name[0]]
    
    return note_dataset

def load_prompt_template(in_path):
    with open(os.path.join(in_path), "r") as f:
        text = f.read()
        template = PromptTemplate.from_template(template=text, name=in_path.split(".")[0])
    return template

def create_entailment_dataset(model_output_file, output_file, notes_dir, prompt_template_file):
    """
    Creates a JSONL dataset for entailment tasks based on model-generated facts and original notes.
    This version utilizes an external prompt template.
    """

    def process_entailment_example(example, premise, hypotheses, entailment_type):
        """Helper function to process entailment examples."""
        entailment_examples = []
        for i, hypothesis in enumerate(hypotheses):
            # Clean up extra whitespace and newlines
            hypothesis = hypothesis.strip().replace('\n', ' ')

            # Substitute placeholders in the template
            prompt = prompt_template.template.format(premise=premise, hypothesis=hypothesis)

            entailment_example = {
                "metadata": {
                    "custom_id": f"{example[2]['metadata']['uid']}|{i}|{entailment_type}",
                    "metadata": example[2]['metadata'],
                    "premise": premise,
                    "hypothesis": hypothesis
                },
                "messages": [{"role": "user", "content": prompt}]
            }
            
            entailment_examples.append(entailment_example)
        
        return entailment_examples

    # Load model output
    model_output = []
    with open(model_output_file, 'r') as f:
        for line in f:
            model_output.append(json.loads(line))

    # Load the prompt template
    prompt_template = load_prompt_template(prompt_template_file)

    # Open the output file for writing
    with open(output_file, 'w') as outfile:
        for example in model_output:
            if "choices" in example[1] and example[1]["choices"]:  # Check if choices exist
                doc_id = example[2]['metadata']["doc_id"]
                dataset_name = example[2]['metadata']["dataset_name"]
                note_dataset = get_note_dataset(dataset_name, notes_dir)
                original_note = note_dataset.loc[note_dataset['doc_id'] == doc_id]['note_text'].values[0] # Retrieve note content using doc_id

                # Process precision entailment (first case)
                premise = original_note
                hypotheses = example[1]["choices"][0]["message"]["content"].split("//")
                print(f"{len(hypotheses)} hypotheses from fact set")
                entailment_examples = process_entailment_example(example, premise, hypotheses, "precision")
                for entailment_example in entailment_examples:
                    json.dump(entailment_example, outfile)
                    outfile.write('\n')

                # Process recall entailment (second case)
                premise = example[1]["choices"][0]["message"]["content"]  # Entire generated fact set
                hypotheses = process_clinical_note(original_note)
                print(f"{len(hypotheses)} hypotheses from original note")
                entailment_examples = process_entailment_example(example, premise, hypotheses, "recall")
                for entailment_example in entailment_examples:
                    json.dump(entailment_example, outfile)
                    outfile.write('\n')


def main():
    parser = argparse.ArgumentParser(description="Create an entailment dataset based on model output and clinical notes.")
    parser.add_argument("--model_output", required=True, help="Path to the model output JSONL file")
    parser.add_argument("--output_file", required=True, help="Path to the output JSONL file")
    parser.add_argument("--notes_dir", required=True, help="Directory containing clinical notes datasets")
    parser.add_argument("--prompt_template", required=True, help="Path to the prompt template file")

    args = parser.parse_args()

    create_entailment_dataset(
        model_output_file=args.model_output,
        output_file=args.output_file,
        notes_dir=args.notes_dir,
        prompt_template_file=args.prompt_template
    )

if __name__ == "__main__":
    main()

