#!/bin/bash

# GPT4o 
python src/factehr/clients/azure_openai_api_parallel.py \
--requests_filepath data/datasets/prompted/fact_decomposition_20240821.jsonl \
--save_filepath data/datasets/completions/debug-parallel.jsonl \
--request_url "../chat/completions?api-version=2023-03-15-preview" \
--max_requests_per_minute 480 \
--max_tokens_per_minute 80000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20 \
--api_key XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# GPT-4 
python src/factehr/clients/azure_openai_api_parallel.py \
--requests_filepath data/datasets/prompted/fact_decomposition_20240821.jsonl \
--save_filepath data/datasets/completions/debug-parallel.jsonl \
--request_url "../chat/completions?api-version=2023-03-15-preview" \
--max_requests_per_minute 480 \
--max_tokens_per_minute 80000 \
--token_encoding_name cl100k_base \
--max_attempts 5 \
--logging_level 20 \
--api_key XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
