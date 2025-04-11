import pandas as pd 

def parse_custom_id(custom_id):
    """Splits the custom_id into its constituent parts: dataset, dataset_type, index, prompt."""
    try:
        if isinstance(custom_id, str):
            dataset, dataset_type, index, prompt = custom_id.split('|')
            return dataset, dataset_type, int(index), prompt
        else:
            return [None, None, None, None]
    except ValueError:
        # Handle any malformed custom_id values
        return None, None, None
    
in_csv = "../just-the-facts/data/datasets/output/binary_entailment_tune_sample.csv"

in_df = pd.read_csv(in_csv)

in_df[['dataset', 'split', 'uid', 'prompt']] = in_df['custom_id'].apply(lambda cid: pd.Series(parse_custom_id(cid)))

sampled_df = in_df.groupby('prompt').apply(lambda x: x.sample(50, replace=True, random_state=42)).reset_index(drop=True)
sampled_df[['prompt', "input", 'model_output', 'label']]

sampled_df[['prompt', "input", 'model_output', 'label']].to_csv("../just-the-facts/data/datasets/output/binary_entailment_tune_sample.csv")

sampled_df.loc[sampled_df['prompt'] == "entailment_binary10"]['model_output'].sample(1).reset_index(drop=True)[0]
