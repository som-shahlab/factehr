# Data Dictionary

## `combined_notes_110424.csv`
This file contains all clinical notes from which FactEHR is derived.

Rows: 2168
Columns: 5

- **`doc_id`** (String): A unique identifier for the note.
- **`note_text`** (String): The full text of the note.
- **`est_token_count`** (Integer): An estimated count of the tokens  in the `note_text`, useful for gauging note length or processing needs.
- **`note_type`** (String): The type of clinical note (e.g., progress note, discharge summary).
- **`dataset_name`** (String): The name of the dataset to which this document belongs, used for grouping or identifying data sources.

## `fact_decompositions_110424.csv`
This file contains all the fact decompositions generated by LLMs.

Rows: 8665
Columns: 6

- **`uid`** (String): A unique identifier for the row.
- **`decomp_id`** (String): A unique identifier for the fact decomposition (hash of the decomposition text).
- **`model`** (String): The name of the model used for fact decomposition.
- **`fact_decomp`** (String): The decomposed facts extracted from the clinical note, as generated by the model.
- **`doc_id`** (String): A unique identifier for the document, linking the row to a specific clinical note.


## `entailment_pairs_110424.csv`
This file contains all entailment pairs derived from fact decompositions and their source notes.

Rows: 987,266
Columns: 11

- **`uid`** (String): A unique identifier for the row.
- **`doc_id`** (String): A unique identifier for the document, linking the row to a specific clinical note.
- **`prompt`** (String): The prompt identifier for fact decomposition.
- **`index`** (Integer): The position or sequence number of the entry within the dataset, often used for indexing or sorting.
- **`entailment_type`** (String): Either "precision" or "recall."
- **`model_name`** (String): The name of the model used to generate the fact decomposition.
- **`not_parseable`** (Boolean): A flag indicating whether the model's output was not parseable in JSON.
- **`model_output`** (String): The raw output text generated by the model.
- **`entailment_pred`** (String): The model's final prediction for the entailment task.
- **`json_parsed`** (String): The model output parsed as JSON.
- **`entailment_pred_raw`** (String): The raw, unprocessed prediction for the entailment task, directly extracted from the model's output.


## `precision_hypotheses_110424.csv`
This file contains the hypotheses for precision entailment pairs. For these entailment pairs, the premise is the source note and each hypothesis is a fact from the corresponding fact decomposition.

Rows:491,663
Columns: 8

- **`doc_id`** (String): A unique identifier for the document, linking the row to a specific clinical note.
- **`dataset`** (String): The name of the dataset to which this entry belongs, used for grouping or identifying data sources.
- **`note_type`** (String): The type of clinical note, indicating its purpose (e.g., progress note, discharge summary).
- **`prompt`** (String): The input text or query presented to the model for generating predictions.
- **`index`** (Integer): The index of the fact in the fact decomposition.
- **`entailment_type`** (String): The type of entailment task being performed, such as entailment, contradiction, or neutral.
- **`model`** (String): The name of the model used to generate the fact decompositions.
- **`hypothesis`** (String): The hypothesis being evaluated in the entailment task, often paired with a prompt or premise.

## `recall_hypotheses_110424.csv`
This file contains the hypotheses for recall entailment pairs. For these entailment pairs, the premise is the fact decomposition and each hypothesis is a sentence from the corresponding source note.

Rows: 127,859
Columns: 7

- **`doc_id`** (String): A unique identifier for the document, linking the row to a specific clinical note.
- **`dataset`** (String): The name of the dataset to which this entry belongs, used for grouping or identifying data sources.
- **`note_type`** (String): The type of clinical note, indicating its purpose (e.g., progress note, discharge summary).
- **`prompt`** (String): The input text or query presented to the model for generating predictions.
- **`index`** (Integer): The index of the sentence in the source note.
- **`entailment_type`** (String): The type of entailment task being performed, such as entailment, contradiction, or neutral.
- **`hypothesis`** (String): The hypothesis being evaluated in the entailment task, often paired with a prompt or premise.

## `all_human_model_entailment_labels_120824.csv`
This file contains entailment pairs sampled from `entailment_pairs_110424.csv` that are labeled by clinical experts.

Rows: 1,036
Columns: 9

- **`doc_id`** (String): A unique identifier for the source clinical note.
- **`prompt`** (String): The prompt identifier for fact decomposition.
- **`index`** (Integer): The index of the hypothesis in its originating document (either the fact decomposition for precision or the source note for recall).
- **`entailment_type`** (String): Either "precision" or "recall".
- **`model_name`** (String): The name of the model used to generate the fact decomposition.
- **`premise`** (String): The primary statement or text used as the basis for the entailment evaluation.
- **`hypothesis`** (String): The statement being evaluated against the premise in the entailment task.
- **`human_pred`** (String): The label assigned by a human annotator for the entailment task, used as ground truth or for comparison.
- **`entailment_pred`** (String): GPT-4o's label for the entailment task.


## `factehr_dev_set.csv`
This file contains entailment pairs sampled from the same data sources as FactEHR, but not the same clinical notes as in `combined_notes_110424.csv`. This dataset is meant to be used for entailment model development. 

Rows: 2,468
Columns: 6

- **`key`** (String): An identifier of the format `model+dataset+note_type+prompt+entailment_type`.
- **`ID`** (String): When paired with `key`, this column acts as a unique identifier.
- **`premise`** (String): The primary statement or text used as the basis for the entailment evaluation.
- **`hypothesis`** (String): The statement being evaluated against the premise in the entailment task.
- **`annotator_id`** (String): A unique identifier for the human annotator who labeled or reviewed the example.
- **`human_label`** (String): The label assigned to the example by a human annotator, used as ground truth for evaluation.

## `factehr_irr_annotations_120824.csv`

This file contains the human annotations used to calculate inter-rater agreement.
