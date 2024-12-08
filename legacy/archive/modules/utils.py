import gc
import lightning as L
import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/share/pi/nigam/'
import transformers
access_token = os.getenv('HF_TOKEN')


from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from modules.data_modules.medalign import MedAlign
from modules.data_modules.radgraph import RadGraph
from modules.data_modules.mimiciii import MimicIII
# from modules.data_modules.shareclef import ShareClef
# from modules.data_modules.X import X
from ipdb import set_trace

class Seq2SeqLMInferenceModule(L.LightningModule):
    def __init__(self, model, max_length=1024, max_new_tokens=256):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        idxs = batch.pop("idx")
        # send batch keys to device:
        # for key in batch:
        #     batch[key] = batch[key].to(self.device)
        preds = self.model.generate(**batch, max_length=self.max_length)
        return (idxs, batch["input_ids"], preds)


class CausalLMInferenceModule(L.LightningModule):
    def __init__(self, model, max_length=1024, max_new_tokens=256):
        super().__init__()
        self.model = model
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        idxs = batch.pop("idx")
        # send batch keys to device:
        # for key in batch:
        #     batch[key] = batch[key].to(self.device)
        input_ids = batch["input_ids"]
        input_length = batch["input_ids"].shape[1]
        preds = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, output_scores=True)
        # also save the logits 
        preds = preds[:, input_length:]
        return (idxs, input_ids, preds, )


NAME_TO_MODULE = {
    "medalign": MedAlign,
    "radgraph": RadGraph, 
    "mimiciii": MimicIII
    # "shareclef": ShareClef,
}


def load_model_and_tokenizer(model_name_or_path: str, device="cuda", eval_type="logit"):

    print(
        f"Running model {model_name_or_path}"
    )
    if (
        "llama-2" in model_name_or_path): 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/share/pi/nigam/" 
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 4096, # may need to change this. 
            "max_new_tokens": 2048,
            "prompt_format": """
INSTRUCTION: {instruction}
CLINICAL NOTE: {text}
ANSWER: """,
        }
    
    elif (
        "flan-t5" in model_name_or_path): 
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir="/share/pi/nigam/" 
        )
        model_config = {
            "model_type": "DecoderOnly",
            "batch_size": 1,
            "max_length": 4096, # may need to change this. 
            "max_new_tokens": 2048,
            "prompt_format": """
INSTRUCTION: {instruction}
CLINICAL NOTE: {text}
ANSWER: """,
        }
    
    else:
        raise NotImplementedError(
            "Model type {} is not supported".format(model_name_or_path)
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    if model_config["model_type"] == "DecoderOnly":
        module = CausalLMInferenceModule(
            model,
            max_length=model_config["max_length"],
            max_new_tokens=model_config["max_new_tokens"],
        )
    elif model_config["model_type"] == "EncoderDecoder":
        module = Seq2SeqLMInferenceModule(
            model,
            max_length=model_config["max_length"],
            max_new_tokens=model_config["max_new_tokens"],
        )
    else:
        raise NotImplementedError("Model type is not supported")

    return module, tokenizer, model_config


def load_data_module(
    dataset_name,
    note_type,
    tokenizer,
    root_path,
    model_config,
    total_samples=None,
):
    assert (
        dataset_name in NAME_TO_MODULE.keys()
    ), f"Dataset {dataset_name} is not supported"
    return NAME_TO_MODULE[dataset_name](
        root_path=root_path,
        tokenizer=tokenizer,
        note_type=note_type, 
        model_config=model_config,
        truncation_strategy="truncate", 
        total_samples=total_samples,
    )


def compute_per_sample_loss(inputs, logits, pad_token_id):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous().double()
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1))
    loss_per_sample = loss_per_sample * (shift_labels != pad_token_id).float()
    loss_per_sample = loss_per_sample.mean(axis=1)
    return loss_per_sample

def get_num_tokens(tokenizer, inputs_label_etc):
    tokens = tokenizer(inputs_label_etc, return_tensors="pt")
    #set_trace()
    num_label_tokens =  len(tokens['input_ids'][0]) - (1 if inputs_label_etc[0] != "\n" else 3)
    return num_label_tokens, tokens


