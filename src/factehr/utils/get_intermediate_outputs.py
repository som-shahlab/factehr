import os
import sys
import json
import pandas as pd
from glob import glob
import argparse

from factehr.evaluation.parse_nli_entailment import *
from factehr.evaluation.entailment import *
import random


entailment_predictions_path = "data/datasets/prompted_sampled/entailment_for_now.jsonl"
entailment_predictions_df = pd.read_json(entailment_predictions_path, lines=True)
entailment_predictions_df['uid'] = entailment_predictions_df[2].apply(lambda x: x['metadata']['metadata']['uid'])
entailment_predictions = entailment_predictions_df
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

    

