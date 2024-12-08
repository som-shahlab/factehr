from time import time
import os
from vllm import LLM, SamplingParams
os.environ['TRANSFORMERS_CACHE'] = '/share/pi/nigam/'

access_token = "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"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Common prefix.
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

generating_prompts = [prefix + prompt for prompt in prompts]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0)

# Create an LLM.
regular_llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.4)

prefix_cached_llm = LLM(model=MODEL_ID,
                        enable_prefix_caching=True,
                        gpu_memory_utilization=0.4)
print("Results without `enable_prefix_caching`")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
start_time_regular = time()
outputs = regular_llm.generate(generating_prompts, sampling_params)
duration_regular = time() - start_time_regular

regular_generated_texts = []
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    regular_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Warmup so that the shared prompt's KV cache is computed.
prefix_cached_llm.generate(generating_prompts[0], sampling_params)

# Generate with prefix caching.
start_time_cached = time()
outputs = prefix_cached_llm.generate(generating_prompts, sampling_params)
duration_cached = time() - start_time_cached

print("Results with `enable_prefix_caching`")

cached_generated_texts = []
# Print the outputs. You should see the same outputs as before.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    cached_generated_texts.append(generated_text)
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# Compare the results and display the speedup
generated_same = all([
    regular_generated_texts[i] == cached_generated_texts[i]
    for i in range(len(prompts))
])
print(f"Generated answers are the same: {generated_same}")

speedup = round(duration_regular / duration_cached, 2)
print(f"Speed up of cached generation compared to the regular is: {speedup}")
=======
from functools import wraps
from time import time
from vllm import LLM, SamplingParams, PromptInputs
from utils import EntailmentDataset

from lmformatenforcer import CharacterLevelParser, JsonSchemaParser
from lmformatenforcer.integrations.vllm import build_vllm_logits_processor, build_vllm_token_enforcer_tokenizer_data, TokenEnforcerTokenizerData

from pydantic import BaseModel
from typing import Sequence

import uuid
from ipdb import set_trace

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 256

ListOrStrList = str | list[str]

class AnswerFormat(BaseModel):
    entailment_prediction: int
    explanation: str


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('\nfunc:%r took: %2.4f sec\n' % \
          (f.__name__, te-ts))
        return result
    return wrap


def vllm_with_character_level_parser(
    llm: LLM, 
    prompt: PromptInputs | Sequence[PromptInputs], 
    tokenizer_data: TokenEnforcerTokenizerData,
    sampling_params: SamplingParams, 
    parser: CharacterLevelParser | None = None,
) -> ListOrStrList:
    """Integrates vLLM with character level parser.
    Taken from: https://github.com/noamgat/lm-format-enforcer/blob/main/samples/colab_vllm_integration.ipynb
    """
    if parser:
        logits_processor = build_vllm_logits_processor(tokenizer_data, parser)
        sampling_params.logits_processors = [logits_processor]
    # Note on batched generation:
    # For some reason, I achieved better batch performance by manually adding a loop similar to this:
    # https://github.com/vllm-project/vllm/blob/main/examples/llm_engine_example.py,
    # I don't know why this is faster than simply calling llm.generate() with a list of prompts, but it is from my tests.
    # However, this demo focuses on simplicity, so I'm not including that here.
    results = llm.generate(prompt, sampling_params=sampling_params)
    if isinstance(prompt, str):
        return results[0].outputs[0].text
    else:
        return [result.outputs[0].text for result in results]


@timing
def generate_llm_outputs(
    llm: LLM, 
    dataset: EntailmentDataset, 
    sampling_params: SamplingParams,
    tokenizer_data: TokenEnforcerTokenizerData, 
    parser: CharacterLevelParser | None = None
) -> None:
    """Generate llm outputs with LLM"""
    outputs = []
    for prompt in dataset:
        result = vllm_with_character_level_parser(
            llm=llm,
            prompt=prompt,
            tokenizer_data=tokenizer_data,
            sampling_params=sampling_params,
            parser=parser
        )
        outputs.append(result)

    output_filename = f"llm_output_{str(uuid.uuid4())}.txt"
    output_file = open(output_filename, "w")

    for generated_text in outputs:
        output_file.write(f"{generated_text}\n")
    
    output_file.close()

    print(f"Wrote outputs to {output_filename}")

def vllm_test():
    sampling_params = SamplingParams(temperature=0.0, max_tokens=DEFAULT_MAX_NEW_TOKENS)

    llm = LLM(model=MODEL_ID, gpu_memory_utilization=0.6, dtype="float16", enable_prefix_caching=True)
    tokenizer_data = build_vllm_token_enforcer_tokenizer_data(llm)
    output_parser = JsonSchemaParser(AnswerFormat.model_json_schema())

    dataset = EntailmentDataset(file_path="/share/pi/nigam/rag-the-facts/datasets/sentences/gemini/mimiciii/nursing_note/PROMPT1_sentence_precision.json")
    print(f"Total data points: {len(dataset)}")

    print("All data points with LLM")
    generate_llm_outputs(
        llm=llm, 
        dataset=dataset, 
        sampling_params=sampling_params,
        tokenizer_data=tokenizer_data,
        parser=output_parser
    )


if __name__ == "__main__":
    vllm_test()
