# For dataset format details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
from ipdb import set_trace
import torch
from torch.utils.data import Dataset
import tiktoken

context_length = 8192 

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            train_size = int(len(self.ann) * 0.8)
            self.ann = self.ann[:train_size]
        else:
            train_size = int(len(self.ann) * 0.8)
            self.ann = self.ann[train_size:]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            number_tokens_instruction = len(self.tokenizer.encode(ann["instruction"]))
            number_tokens_prompt_template = len(self.tokenizer.encode(PROMPT_DICT["prompt_input"]))
            target_ehr_length = (context_length - number_tokens_instruction - number_tokens_prompt_template)
            if target_ehr_length <= 0:
                prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
            else:
                # Do a first pass with a fast tokenizer
                fast_tokenizer = tiktoken.get_encoding("cl100k_base")
                fast_encoded = fast_tokenizer.encode(ann.get("input"))
                fast_encoded_truncated = fast_encoded[-(2 * target_ehr_length) :]
                fast_truncated_ehr = fast_tokenizer.decode(fast_encoded_truncated)
                # Then do a second pass with the actual tokenizer
                encoded_ehr = self.tokenizer.encode(fast_truncated_ehr)
                truncated_encoded_ehr = encoded_ehr[-target_ehr_length:]

                truncated_ehr = self.tokenizer.decode(truncated_encoded_ehr)
                ann['input'] = truncated_ehr
                prompt = PROMPT_DICT["prompt_input"].format_map(ann)


        if isinstance(ann['output'], list):
            full_note = ' '.join(ann['output'])
        
        else: 
            full_note = ann['output']
        example = prompt + full_note
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
