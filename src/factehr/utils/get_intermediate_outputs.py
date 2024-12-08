import os
import sys
import json
import pandas as pd
from glob import glob
import argparse

from factehr.evaluation.parse_nli_entailment import *
from factehr.evaluation.entailment import *
import random


# input_notes_path = "/Users/akshayswaminathan/just-the-facts/data/datasets/prompted_sampled/fact_decomposition_20241009.jsonl"
# facts_generated_path = "/Users/akshayswaminathan/just-the-facts/data/datasets/completions/test/merged_fact_decomp.jsonl"
# facts_split_path = "/Users/akshayswaminathan/just-the-facts/data/datasets/completions/test/entailment_input_.jsonl"
entailment_predictions_path = "data/datasets/prompted_sampled/entailment_for_now.jsonl"

# input_notes_df = pd.read_json(input_notes_path, lines=True)
# facts_generated_df = pd.read_json(facts_generated_path, lines=True)
# facts_split_df = pd.read_json(facts_split_path, lines=True)
entailment_predictions_df = pd.read_json(entailment_predictions_path, lines=True)

# input_notes_df['uid'] = input_notes_df['metadata'].apply(lambda x: x['uid'])
# facts_generated_df['uid'] = facts_generated_df[2].apply(lambda x: x['metadata']['uid'])
# facts_split_df['uid'] = facts_split_df['metadata'].apply(lambda x: x['metadata']['uid'])
entailment_predictions_df['uid'] = entailment_predictions_df[2].apply(lambda x: x['metadata']['metadata']['uid'])


# random_uid = random.choice(input_notes_df['uid'].tolist())

# input_note_prompt = input_notes_df.loc[input_notes_df['uid'] == random_uid]['messages'].values[0][0]['content']

# facts_generated = facts_generated_df.loc[facts_generated_df['uid'] == random_uid][1].apply(lambda x: parse_message_from_choices(x) if isinstance(x, dict) else parse_error_string(x)).values[0]
# facts_split = [x[0]['content'] for x in facts_split_df.loc[facts_split_df['uid'] == random_uid]['messages'].values]

entailment_predictions = entailment_predictions_df
# entailment_predictions = entailment_predictions_df.loc[entailment_predictions_df['uid'] == random_uid]
entailment_predictions.columns = ['messages', 'choices', 'metadata', 'uid']
entailment_predictions['premise'] = entailment_predictions['metadata'].apply(lambda x: x['metadata']["premise"] if 'metadata' in x['metadata'] else None)
entailment_predictions['hypothesis'] = entailment_predictions['metadata'].apply(lambda x: x['metadata']["hypothesis"] if 'metadata' in x['metadata'] else None)
entailment_predictions['model_output'] = entailment_predictions['choices'].apply(lambda x: parse_message_from_choices(x) if isinstance(x, dict) else parse_error_string(x)) #TODO check this works for openai
entailment_predictions['json_parsed'] = entailment_predictions['model_output'].apply(extract_json_from_string)
entailment_predictions['entailment_pred_raw'] = entailment_predictions.apply(parse_entailment, axis=1)
entailment_predictions['not_parseable'] = (~entailment_predictions['entailment_pred_raw'].isin([0, 1])) 
entailment_predictions['entailment_pred'] = entailment_predictions.apply(lambda row: 0 if row['not_parseable'] else int(row['entailment_pred_raw']), axis=1) 
entailment_predictions['entailment_pred'] = entailment_predictions['entailment_pred'].apply(int)
entailment_predictions['model_name'] = entailment_predictions['metadata'].apply(lambda x: x['metadata']["model_name"])

parsed_custom_id = entailment_predictions.apply(lambda row: parse_and_assign(row['metadata']['metadata']["custom_id"]), axis=1)

entailment_predictions = entailment_predictions.join(parsed_custom_id)

entailment_to_save = entailment_predictions[["uid", "doc_id", "dataset_name", "note_type", "prompt", "index", "entailment_type", "model_name", "not_parseable", "model_output", 'entailment_pred']]

entailment_to_save.to_csv("data/datasets/prompted_sampled/entailment_for_now.csv")

# precision_preds = entailment_predictions[entailment_predictions['precision_recall'] == 'precision']['entailment_pred'].tolist()
# recall_preds = entailment_predictions[entailment_predictions['precision_recall'] == 'recall']['entailment_pred'].tolist()

# # Get the counts
# precision_count = len(precision_preds)
# recall_count = len(recall_preds)

# # Call the entailment_proportion function for both precision and recall
# precision_proportion = entailment_proportion(precision_preds)
# recall_proportion = entailment_proportion(recall_preds)

# # Output the results
# print(f"Precision entailment proportion: {precision_proportion} (based on {precision_count} rows)")
# print(f"Recall entailment proportion: {recall_proportion} (based on {recall_count} rows)")

# # Open a text file to save the results
# with open(f"intermediate_outputs_{random_uid}.txt", "w") as outfile:
    
#     # Write the input note prompt
#     outfile.write("*******************************************\n")
#     outfile.write("*************INPUT NOTE+PROMPT*************\n")
#     outfile.write("*******************************************\n")
#     outfile.write(f"{input_note_prompt}\n\n")
    
#     # Write the generated facts
#     outfile.write("******************************************\n")
#     outfile.write("*************GENERATED FACTS*************\n")
#     outfile.write("******************************************\n")
#     outfile.write(f"{facts_generated}\n\n")
    
#     # Write how the facts were split into hypothesis/premise pairs
#     outfile.write("**************************************************\n")
#     outfile.write("*************PREMISE/HYPOTHESIS PAIRS*************\n")
#     outfile.write("**************************************************\n")
#     for i, pair in enumerate(facts_split):
#         outfile.write(f"************PAIR {i+1}**********\n")
#         outfile.write(f"{pair}\n")
    
#     outfile.write("************************************************\n")
#     outfile.write("*************ENTAILMENT PREDICTIONS*************\n")
#     outfile.write("************************************************\n")
#     # Write the entailment outputs for each pair
#     for index, row in entailment_predictions.iterrows():
#         outfile.write(f"**********ROW {index + 1}***************\n")
#         outfile.write(f"UID: {row['uid']}\n")
#         outfile.write(f"Premise: {row['premise']}\n")
#         outfile.write(f"Hypothesis: {row['hypothesis']}\n")
#         outfile.write(f"Entailment Prediction: {row['entailment_pred']}\n\n")
        
#     outfile.write("***********************************************\n")
#     outfile.write("*************FACT PRECISION/RECALL*************\n")
#     outfile.write("***********************************************\n")
    
#     outfile.write(f"Precision entailment proportion: {precision_proportion} (based on {precision_count} rows)")
#     outfile.write(f"Recall entailment proportion: {recall_proportion} (based on {recall_count} rows)")

# print("Intermediate outputs saved to 'intermediate_outputs.txt'")
    

