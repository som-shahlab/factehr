<div align="center">
  <h1>📄 🧠 FactEHR</h1>
  <h4>
    <a href="https://stanford.redivis.com/datasets/bckk-15p0mwmz7">💾 Dataset</a> • 
    <a href="https://arxiv.org/abs/2412.12422">📝 Paper</a> • 
    <a href="https://github.com/som-shahlab/factehr">⚙️ Code & Docs</a>
  </h4>
  <h4>A benchmark for fact decomposition and entailment evaluation of clinical notes</h4>
  <p>
    2,168 notes • 8,665 decompositions • 987,266 entailment pairs • 1,036 human-labeled examples
  </p>
</div>

---

# 🧠 FactEHR: A Benchmark for Fact Decomposition of Clinical Notes

**FactEHR** is a benchmark dataset designed to evaluate the ability of large language models (LLMs) to perform **factual reasoning** over clinical notes. It includes:

- **2,168** deidentified notes from multiple publicly available datasets  
- **8,665** LLM-generated fact decompositions  
- **987,266** entailment pairs evaluating precision and recall of facts  
- **1,036** expert-annotated examples for evaluation

FactEHR supports LLM evaluation across tasks like **information extraction**, **entailment classification**, and **model-as-a-judge** reasoning.

> [!WARNING]  
> **Usage Restrictions:** The FactEHR dataset is subject to a Stanford Dataset DUA. Sharing data with LLM API providers is prohibited.  
> We follow PhysioNet’s responsible use principles for running LLMs on sensitive clinical data:
> 
> - ✅ Use **Azure OpenAI** (with human review opt-out)  
> - ✅ Use **Amazon Bedrock** (private copies of foundation models)  
> - ✅ Use **Google Gemini via Vertex AI** (non-training usage)  
> - ✅ Use **Anthropic Claude** (no prompt data used for training)
> - ❌ **Do not transmit data** to commercial APIs (e.g., ChatGPT, Gemini, Claude) unless HIPAA-compliant and explicitly permitted  
> - ❌ **Do not share** notes or derived outputs with third parties

---

## 📦 What's Included

| Component          | Count     | Description                                               |
|-------------------|-----------|-----------------------------------------------------------|
| Clinical Notes     | 2,168     | Deidentified clinical notes across 4 public datasets      |
| Fact Decompositions | 8,665     | Model-generated fact lists from each note                |
| Entailment Pairs   | 987,266   | Pairs evaluating if notes imply facts (and vice versa)   |
| Expert Labels      | 1,036     | Human-annotated entailment labels for benchmarking        |

See the [data summary](docs/dataset_summary.md) and [release files](docs/release_files.md) for more details.

---

## 🛠️ Installation

```bash
python -m pip install -e .
```
---

## 🧪 Running the Experiments

We support two core experiments for evaluating factual reasoning in clinical notes:

### 1. 🧩 Fact Decomposition

This task involves prompting LLMs to extract structured atomic facts from raw clinical notes.

**Inputs:**
- Notes from `combined_notes_110424.csv`
- Prompt templates in `prompts/`
- An LLM provider (e.g., OpenAI, Claude, Bedrock)

**Outputs:**
- A list of decomposed facts per note (stored in `fact_decompositions_*.csv`)

See [docs/experiments.md](docs/experiments.md#1-generating-fact-decompositions) for instructions on:
- Supported prompt formats
- Batch processing with rate-limited APIs
- Handling invalid or unparseable outputs

---

### 2. 🔍 Entailment Evaluation

FactEHR supports two entailment settings:
- **Precision (`note ⇒ fact`)** — Does a note imply a given fact?
- **Recall (`fact ⇒ sentence`)** — Does a fact imply a sentence in the note?

**Approaches:**
- Use your own classifier or fine-tuned entailment model
- Use an LLM-as-a-judge (e.g., GPT-4, Claude) to score entailment pairs

**Inputs:**
- Entailment pairs in `entailment_pairs_110424.csv`
- Fact decompositions and source notes
- Optional: human-labeled samples for evaluation

**Outputs:**
- Entailment predictions (binary labels or probabilities)
- Comparison against human annotations for calibration

See [docs/experiments.md](docs/experiments.md#2-running-llm-experiments) for:
- Prompting logic
- Suggested evaluation metrics
- Example LLM judge scripts

---


## 📚 Citation

If you use **FactEHR** in your research, please cite:

```bibtex
@misc{munnangi2025factehrdatasetevaluatingfactuality,
      title={FactEHR: A Dataset for Evaluating Factuality in Clinical Notes Using LLMs}, 
      author={Monica Munnangi and Akshay Swaminathan and Jason Alan Fries and Jenelle Jindal and Sanjana Narayanan and Ivan Lopez and Lucia Tu and Philip Chung and Jesutofunmi A. Omiye and Mehr Kashyap and Nigam Shah},
      year={2025},
      eprint={2412.12422},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.12422}, 
}
