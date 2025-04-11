# Simple Benchmarking of Clinical Text Preprocessing

## I. Performance Summary

| Method                   | CPUs    | # Docs | Batch Size | Execution Time (Mean) [s] | Execution Time per Doc [s] |
|--------------------------|---------|--------|------------|---------------------------|----------------------------|
| Trove (fast)             | 1x CPU  | 2088   | -          | 45.3                       | 0.022                      |
| Trove (fast)             | 4x CPU  | 2088   | 50         | 33.0                       | 0.016                      |
| MedSpaCy                 | 1x CPU  | 2088   | -          | 285.3                      | 0.137                      |
| SpaCy (default)          | 1x CPU  | 2088   | -          | 238.1                      | 0.112                      |
| SpaCy (default)          | 4x CPU  | 2088   | 50         | 111.9                      | 0.054                      |

**System Configuration:** Apple M1 2020

## II. NLP Frameworks

### 1. [Trove](https://github.com/som-shahlab/trove)

- **`fast_ct_sentence_splitter`:** A faster, rule-based sentence splitter.
- **`ct_sentence_splitter`:** A slower, more aggressive rule-based sentence splitter that splits on two spaces.

### 2. Default SpaCy (`en_core_web_sm`)

### 3. [MedSpaCy](https://github.com/medspacy/medspacy)

- MedSpaCy includes a Python implementation of RuSH ([Ru]le-based sentence [S]egmenter using [H]ashing).
