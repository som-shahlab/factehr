# FactEHR Dataset Summary

### Named Subsets

| Name        | Creation Rule                     | Size | Description                                         |
|:----------------|:--------------------------------------|:---------|:--------------------------------------------------------|
| `FactNotes`     | `note ∈ D`                            | 2,168    | A collection of notes from the dataset D.               |
| `FactDecomp`    | `note → fact-list`                    | 8,665    | Decomposition of notes into a list of associated facts. |
| `FactEntail_p`  | `I[note ⇒ fact]`                      | 491,663        | Entailment indicating if a note implies a fact.         |
| `FactEntail_r`  | `I[fact-list ⇒ sentence]`             | 495,603        | Entailment indicating if a fact list implies a sentence.|
| `FactEntail`    | `FactEntail_p ∪ FactEntail_r`         | 987,266        | Entailment pairs.         |
| `FactEntail_ann`| `x ~ U(FactEntail)`                   | 1036    | Annotated entailment data sampled from `FactEntail_p` and `FactEntail_r`. |

### Provenance & Note Types

| Note 					| MIMIC-CXR | MIMIC-III | MedAlign | Coral |
|:---------------------|-----------|-----------|----------|-------|
| progress_notes       |     -     |    250    |   250    |  172  |
| nursing_notes        |     -     |    250    |   129    |   -   |
| discharge_summaries  |     -     |    250    |   117    |   -   |
| procedure notes      |     -     |     *     |   250    |   -   |
| radiology_reports    |    250    |    250    |    *     |   -   |

### Token summary

| Tokens               | Count | Frequency |
|----------------------|-------|-----------|
| [231.00 - 606.40)    | 1062  | 49.0%     |
| [606.40 - 981.80)    | 288   | 13.3%     |
| [981.80 - 1357.20)   | 114   | 5.3%      |
| [1357.20 - 1732.60)  | 125   | 5.8%      |
| [1732.60 - 2108.00)  | 94    | 4.3%      |
| [2108.00 - 2483.40)  | 119   | 5.5%      |
| [2483.40 - 2858.80)  | 116   | 5.4%      |
| [2858.80 - 3234.20)  | 99    | 4.6%      |
| [3234.20 - 3609.60)  | 78    | 3.6%      |
| [3609.60 - 3985.00)  | 73    | 3.4%      |
