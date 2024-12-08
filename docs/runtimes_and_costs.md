
# Experiment Runtimes

This document is updated as code is optimized to improve performance. 

## I. Summary 

### Estimated Runtimes & Cost

| Task                | Code Name    | Provider      | Model           | Est. Runtime   | Input Cost | *Output Cost |
|---------------------|--------------|---------------|-----------------|----------------|------------|-------------|
| Fact Decomposition  | `FactDecomp` | AzureOpenAI   | shc-gpt-4o       | 8.8 hours      |   $44.63         |   $471.37          |
| Fact Decomposition  | `FactDecomp` | AzureOpenAI   | gpt-4            | TBD            |            |             |
| Fact Decomposition  | `FactDecomp` | Vertex AI     | MedLM            | TBD            |            |             |
| Fact Decomposition  | `FactDecomp` | Vertex AI     | Gemini-Pro 1.5   | TBD            |            |             |
| Fact Decomposition  | `FactDecomp` | -             | Llama3-8b        | TBD            |            |             |


- Pricing based on [current Azure OpenAI rates]( https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/)
- \* Worst-case pricing based on `max_tokens`. In practice the cost will be substantially less.



## II. Sub-Task Breakdown

### A. Fact Decomposition
#### 1. Datasets

- **`FactDecomp`** (`fact_decomposition_20240821.jsonl`) 4 prompt templates × 1,841 notes = 7,364 prompts
- **`FactDecompBenchmark`** (`factdecomp_benchmark.jsonl`) 100 docs sampled from `FactDecomp` using  ```shuf fact_decomposition_20240821.jsonl | head -n 100 > factdecomp_benchmark.jsonl```

#### 2. Speed Tests 

| Model         | Approach                                                                         | #Prompts | max_tokens | Wall Time (secs) | Per-Prompt Rate (secs) | Input Cost | *Output Cost | Output Cost |
|---------------|----------------------------------------------------------------------------------|-----------|--------------|------------------|------------------------|------------|-------------|-------------------|
| shc-gpt-4o    | [`azure_openai_api_parallel.py`](src/factehr/clients/azure_openai_api_parallel.py) | 100       | 4096         | 413.55           | 4.14                   | $0.65      | $6.14       |      $1.60             |
| shc-gpt-4o    | [`azure_openai_api.py`](src/factehr/clients/azure_openai_api.py)                 | 100       | 4096         | 1643.31          | 16.4                   | $0.65      | $6.14       |      $1.60             |

*\* worst-case estimate based on max_tokens*

```bash
python src/factehr/clients/<APPROACH>.py \
--deployment shc-gpt-4o \
--max_tokens 4096 \
--path_to_prompted_dataset factdecomp_benchmark.jsonl \
--path_to_output_file shc-gpt-4o_benchmark_output.jsonl
```

### B. Entailment

