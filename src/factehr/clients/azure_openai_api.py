""" 
Simple (SLOW!) client for the Azure OpenAI API

COMMENTS:
- Naive batching does not work with the chat completion endpoint
- asyncio is required to improve throughput

TODO
- azure_ad_token_provider requires Azure authorization
- impelement API_KEY vs. token provider as command line option

Installing the Azure Developer CLI provides a more secure way of setting your
API key.

See https://learn.microsoft.com/en-us/azure/developer/azure-developer-cli/get-started

On Mac with homebrew. The requires authenticating with your SHC credentials

```
brew tap azure/azd && brew install azd
azd auth login 
```

Example Useage:

export AZURE_OPENAI_API_KEY="your_value_here"
export AZURE_OPENAI_ENDPOINT="your_value_here"

python src/factehr/clients/azure_openai_api.py \
--deployment shc-gpt-4o \
--max_tokens 4096 \
--path_to_prompted_dataset data/datasets/prompted/fact_decomposition_20240821_DEBUG.jsonl \
--path_to_output_file data/datasets/completions/debug.jsonl \
--estimate_cost


python src/factehr/clients/azure_openai_api.py \
--deployment shc-gpt-4o \
--max_tokens 4096 \
--path_to_prompted_dataset data/datasets/prompted/fact_decomposition_20240821.jsonl \
--path_to_output_file data/datasets/completions/debug.jsonl \
--estimate_cost

"""

import os
import time
import json
import timeit
import openai
import asyncio
import argparse
import collections
import numpy as np
from typing import List, Set
from openai import AsyncAzureOpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from requests.exceptions import ConnectionError, Timeout
from factehr.utils import load_jsonl


########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Query Azure OpenAI API")

parser.add_argument(
    "-i",
    "--path_to_prompted_dataset",
    type=str,
    help="Path to the prompted dataset JSONL file",
    # required=True,
)

parser.add_argument(
    "-o",
    "--path_to_output_file",
    type=str,
    help="Path to the output JSONL file",
)

parser.add_argument(
    "-d",
    "--deployment",
    type=str,
    help="Deployment name for the Azure OpenAI API",
    default=None,
)

parser.add_argument(
    "-c",
    "--generation_config",
    type=str,
    help="TOML file containing generation parameters",
    default=None,
)

parser.add_argument(
    "--max_tokens",
    type=int,
    help="maximum tokens per completion",
    default=128,
)

parser.add_argument(
    "--resume",
    type=str,
    help="resume previous completion",
)

parser.add_argument(
    "--estimate_cost",
    action="store_true",
    help="Estimate cost of completions",
)


########## Data Loaders ##########


def load_jsonl_prompted_dataset(file_path: str, filter_for=None):
    """Load a JSON Lines file into memory"""
    # only include prompts that match the filter
    filter_for = filter_for if filter_for else {}

    with open(file_path, "r") as file:
        for line in file:
            item = json.loads(line)
            if filter_for:
                # Requires a metadata object in the JSONL
                if any(
                    item["metadata"].get(key) in value
                    for key, value in filter_for.items()
                ):
                    yield item
            else:
                yield item


def write_jsonl_to_file(file_path: str, array: List[str]):
    """Append JSON objects to a file in JSON Lines format"""
    with open(file_path, "a") as file:
        for item in array:
            file.write(json.dumps(item) + "\n")


########## Chat Completions ##########


def get_fresh_token_provider():
    return get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )


def run_chat_completion(client, deployment: str, messages, delay: int, **kwargs):
    """Run a chat completion using the Azure OpenAI API"""
    # Default values for common text generation parameters
    # See https://learn.microsoft.com/en-us/azure/ai-services/openai/reference
    generation_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16,  # max 2048 or 4096
        "presence_penalty": 0.0,  # [-2.0, 2.0]
        "frequency_penalty": 0.0,  # [-2.0, 2.0]
        "logit_bias": None,
        "n": 1,  # 1 to 128 WARNING - rate limits
        "logprobs": None,  # 0 - 5
        "stop": None,  # up to 4 sequences
        "stream": False,
    }

    generation_params.update(kwargs)

    try:
        completion = client.chat.completions.create(
            model=deployment, messages=messages, **generation_params
        )

    except openai.APIConnectionError as e:
        print(f"Failed to connect to OpenAI API: {e}")

        # if client._azure_ad_token_provider is None:
        #     client._azure_ad_token_provider = get_fresh_token_provider()

        return None, generation_params

    # TODO implement some smarter backoff
    except openai.RateLimitError as e:
        retry_time = delay
        if hasattr(e, "response") and hasattr(e.response, "headers"):
            retry_time = int(e.response.headers.get("Retry-After", retry_time))
        print(f"OpenAI API request exceeded rate limit: {e}")
        return run_chat_completion(client, deployment, messages, **kwargs)

    except openai.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")

        # if client._azure_ad_token_provider is None:
        #     client._azure_ad_token_provider = get_fresh_token_provider()

        return None, generation_params

    return completion, generation_params


########## Pricing ##########


def estimate_request_limits(
    prompts: List[str],
    requests_per_minute: int,
    tokens_per_minute: int,
    max_tokens: int = 16,
):
    """

    Cost Estimation:
    https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/
    """
    lens = [prompt["metadata"]["n_tokens"] for prompt in prompts]
    total_tokens = sum(lens)
    worst_case_output_tokens = len(prompts) * max_tokens
    print(worst_case_output_tokens, len(prompts), max_tokens)
    # some idealized rate math
    m = np.mean(lens) + np.std(lens)
    idealized_token_rate = total_tokens / tokens_per_minute

    # spitballing a rate
    est_completion_tokens = max_tokens + np.mean(lens) + (2 * np.std(lens))
    est_req_rate = tokens_per_minute / est_completion_tokens

    print("~" * 50)
    print(f"Total Prompts: {len(prompts)}")
    print(f"Total Input Tokens: {total_tokens}")
    print(
        f"Worst Case Output Tokens (max_tokens={max_tokens}): {worst_case_output_tokens}"
    )
    print(f"Per completion tokens: {est_completion_tokens}")
    print(f"Idealized Wall Time (Total Tokens/TPM): {idealized_token_rate:.1f} minutes")
    print(f"Est. Request Rate: {est_req_rate}/per minute )")
    print("~" * 50)

    # cost (per 1000 tokens)
    pricing_inputs = {"GPT-4": 0.03, "GPT4-32k": 0.06, "shc-gpt-4o": 0.005}
    for model_name in pricing_inputs:
        padding = " " * (15 - len(model_name))
        print(
            f"~Input Cost: {model_name}{padding}${total_tokens/1000 * pricing_inputs[model_name]:.2f} USD"
        )
    # estimate worst case output pricing
    pricing_inputs = {"GPT-4": 0.06, "GPT4-32k": 0.12, "shc-gpt-4o": 0.015}
    for model_name in pricing_inputs:
        padding = " " * (15 - len(model_name))
        print(
            f"Worst-Case Output Cost: {model_name}{padding}${worst_case_output_tokens/1000 * pricing_inputs[model_name]:.2f} USD"
        )
    return est_req_rate


def prompt_prompted_data_summary(prompts):
    summary = collections.defaultdict(collections.Counter)
    for prompt in prompts:
        summary["datset_name"][prompt["metadata"]["dataset_name"]] += 1
        summary["prompt_template_name"][prompt["metadata"]["prompt_template_name"]] += 1
    for key, value in summary.items():
        print(key, dict(value))


def load_completions(file_path: str) -> Set[str]:
    completions = load_jsonl(file_path)
    return set(prompt["prompt"]["metadata"]["uid"] for prompt in completions)


def main():

    args = parser.parse_args()

    endpoint = os.getenv("ENDPOINT_URL", "https://shcopenaisandbox.openai.azure.com/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", None)

    # SHC rate limits used to set batch sizes and requests per minute
    requests_per_minute = 480
    tokens_per_minute = 80_000

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_fresh_token_provider(),
        api_version="2024-05-01-preview",
        api_key=api_key,
    )

    # load entire dataset into memory
    filter_for = {}  # {"dataset_name": {"coral-breastca", "coral-pdac"}}
    prompts = list(
        load_jsonl_prompted_dataset(
            args.path_to_prompted_dataset, filter_for=filter_for
        )
    )

    prompt_prompted_data_summary(prompts)

    est_req_rate = estimate_request_limits(
        prompts, requests_per_minute, tokens_per_minute, max_tokens=args.max_tokens
    )

    if args.estimate_cost:
        return

    delay = 1.0  # 60.0 / est_req_rate
    print(f"Estimated Delay: {delay}")

    # load existing completions file
    completed_uids = load_completions(args.resume) if args.resume else set()
    total_uids = set()
    print(len(completed_uids))

    # query the API for each prompt
    for i, prompt in enumerate(prompts):

        uid = prompt["metadata"]["uid"]
        total_uids.add(uid)

        if uid in completed_uids:
            print(f"Skipping {uid}, alredy completed")
            continue

        completion, params = run_chat_completion(
            client,
            args.deployment,
            prompt["messages"],
            delay=delay,
            max_tokens=args.max_tokens,
        )

        if completion is None:
            print(f"Failed to generate completion {uid}")
            continue

        # add complete metadata to the completion
        completion = completion.to_dict()
        completion["prompt"] = prompt
        completion["params"] = params

        # write this completion to the output file
        write_jsonl_to_file(args.path_to_output_file, [completion])
        time.sleep(delay)

        print(f"Completed {i+1} of {len(prompts)} {uid}")
        completed_uids.add(uid)

    not_in_total = completed_uids - total_uids
    print(f"Not completed: {len(not_in_total)}")


if __name__ == "__main__":

    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    print(f"Execution time: {elapsed_time:.2f} seconds")
