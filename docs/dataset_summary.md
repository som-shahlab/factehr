# FactEHR Dataset Summary

### Named Subsets

| Name        | Creation Rule                     | Size | Description                                         |
|:----------------|:--------------------------------------|:---------|:--------------------------------------------------------|
| `FactNotes`     | `note ∈ D`                            | 2,172    | A collection of notes from the dataset D.               |
| `FactDecomp`    | `note → fact-list`                    | TBD    | Decomposition of notes into a list of associated facts. |
| `FactEntail_p`  | `I[note ⇒ fact]`                      |          | Entailment indicating if a note implies a fact.         |
| `FactEntail_r`  | `I[fact-list ⇒ sentence]`             |          | Entailment indicating if a fact list implies a sentence.|
| `FactEntail`    | `FactEntail_p ∪ FactEntail_r`         |          | Entailment pairs.         |
| `FactEntail_ann`| `x ~ U(FactEntail)`                   | TBD    | Annotated entailment data sampled from `FactEntail_p` and `FactEntail_r`. |

### Provenance & Note Types

| Note 					| MIMIC-CXR | MIMIC-III | MedAlign | Coral |
|:---------------------|-----------|-----------|----------|-------|
| progress_notes       |     -     |    250    |   250    |  172  |
| nursing_notes        |     -     |    250    |   137    |   -   |
| discharge_summaries  |     -     |    250    |   113    |   -   |
| procedure notes      |     -     |     *     |   250    |   -   |
| radiology_reports    |    250    |    250    |    *     |   -   |
