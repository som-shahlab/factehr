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
@InProceedings{pmlr-v298-munnangi25a,
  title = 	 {Fact{EHR}: A Dataset for Evaluating Factuality in Clinical Notes Using {LLM}s},
  author =       {Munnangi, Monica and Swaminathan, Akshay and Fries, Jason Alan and Jindal, Jenelle A and Narayanan, Sanjana and Lopez, Ivan and Tu, Lucia and Chung, Philip and Omiye, Jesutofunmi and Kashyap, Mehr and Shah, Nigam},
  booktitle = 	 {Proceedings of the 10th Machine Learning for Healthcare Conference},
  year = 	 {2025},
  editor = 	 {Agrawal, Monica and Deshpande, Kaivalya and Engelhard, Matthew and Joshi, Shalmali and Tang, Shengpu and Urteaga, Iñigo},
  volume = 	 {298},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {15--16 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v298/main/assets/munnangi25a/munnangi25a.pdf},
  url = 	 {https://proceedings.mlr.press/v298/munnangi25a.html},
  abstract = 	 {Verifying and attributing factual claims is essential for the safe and effective use of large language models (LLMs) in healthcare. A core component of factuality evaluation is fact decomposition—the process of breaking down complex clinical statements into fine-grained, atomic facts for verification. Recent work has proposed fact decomposition, which uses LLMs to rewrite source text into concise sentences conveying a single piece of information, as an approach for fine-grained fact verification, in the general domain. However, clinical documentation poses unique challenges for fact decomposition due to dense terminology and diverse note types and remains understudied. To address this gap and to explore these challenges, we present FactEHR, an NLI dataset consisting of full document fact decompositions for 2,168 clinical notes spanning four types from three hospital systems resulting in 987,266 entailment pairs. We asses the generated facts on different axes, from entailment evaluation of LLMs to a qualitative analysis. Our evaluation, including review by clinicians, highlights significant variability in the performance of LLMs for fact decom- position from Gemini generating highly relevant and factually correct facts to Llama-3 generating fewer and inconsistent facts. The results underscore the need for better LLM capabilities to support factual verification in clinical text. To facilitate further research, we release anonymized code and plan to make the dataset available upon acceptance.}
}
