import pandas as pd
from pybigquery import BigQuery

# Define your BigQuery project ID
project_id = "som-nero-phi-nigam-starr"

# Establish connections to BigQuery datasets
bq = BigQuery(project_id=project_id)

medalign_note_tbl = bq.query_to_pandas('SELECT * FROM medalign_release_phi.note')
person_tbl = bq.query_to_pandas('SELECT * FROM shahlab_omop_cdm5_subset_2023_03_05.person WHERE person_id IN {}'.format(tuple(medalign_note_tbl['person_id'])))
provider_tbl = bq.query_to_pandas('SELECT * FROM shahlab_omop_cdm5_subset_2023_03_05.provider WHERE provider_id IN {}'.format(tuple(medalign_note_tbl['provider_id'])))

# Function to get unique concept IDs from a DataFrame
def get_unique_concepts_from_tbl(in_df):
    concept_cols = [col for col in in_df.columns if col.endswith("_concept_id")]
    unique_concepts = set()
    for col in concept_cols:
        unique_concepts.update(in_df[col].dropna().unique())
    return list(unique_concepts)

# Function to get concept value from a specified concept_id column
def get_concept_value_from_col(in_df, col_name, concept_df):
    var_name = col_name.replace("_concept_id", "")
    out_df = pd.merge(in_df, concept_df[['concept_id', var_name + '_concept_name']], left_on=col_name, right_on='concept_id', how='left')
    return out_df.drop(columns=['concept_id'])

# Query concept table and get unique concepts
concepts = []
for df in [medalign_note_tbl, person_tbl, provider_tbl]:
    concepts.extend(get_unique_concepts_from_tbl(df))
concepts = list(set(concepts))

concept_tbl = bq.query_to_pandas('SELECT * FROM shahlab_omop_cdm5_subset_2023_03_05.concept WHERE concept_id IN {}'.format(tuple(concepts)))

# Function to create concept value columns in DataFrame
def create_concept_value_cols(in_df, concept_df):
    concept_cols = [col for col in in_df.columns if col.endswith("_concept_id")]
    out_df = in_df.copy()
    for col in concept_cols:
        out_df = get_concept_value_from_col(out_df, col, concept_df)
    return out_df

# Clean person_tbl, provider_tbl, and medalign_note_tbl
person_clean = create_concept_value_cols(person_tbl, concept_tbl)
person_clean = person_clean[['person_id', 'year_of_birth', 'care_site_id', 'location_id', 'gender', 'race', 'ethnicity']]

provider_clean = create_concept_value_cols(provider_tbl, concept_tbl)
provider_clean = provider_clean[['provider_id', 'year_of_birth', 'care_site_id', 'specialty']]

note_clean = create_concept_value_cols(medalign_note_tbl, concept_tbl)
note_clean = note_clean.rename(columns=str.lower)
note_clean = note_clean[['note_id', 'person_id', 'note_date', 'note_datetime', 'note_title', 'note_text', 'provider_id', 'visit_occurrence_id', 'visit_detail_id', 'note_type', 'note_class', 'language']]

# Merge cleaned DataFrames and compute additional columns
final_df = pd.merge(note_clean, provider_clean, on='provider_id', how='left')
final_df = pd.merge(final_df, person_clean, on='person_id', how='left')
final_df['note_length'] = final_df['note_text'].apply(len)
final_df['note_year'] = pd.to_datetime(final_df['note_date']).dt.year

# Define functions for categorizing note_title, note_class, and specialty
def categorize_note_title(title):
    # Implement your categorization logic here
    pass

def categorize_note_class(note_class):
    # Implement your categorization logic here
    pass

def categorize_specialty(specialty):
    # Implement your categorization logic here
    pass

final_df['note_title_cat'] = final_df['note_title'].apply(categorize_note_title)
final_df['note_class_cat'] = final_df['note_class'].apply(categorize_note_class)
final_df['specialty_cat'] = final_df['specialty'].apply(categorize_specialty)
final_df['patient_age'] = final_df['note_year'] - final_df['year_of_birth']

# Generate summary table using pandas' describe method
summary_tbl = final_df[['gender', 'patient_age', 'race', 'ethnicity', 'language', 'note_type', 'note_length', 'note_year', 'note_title_cat', 'note_class_cat', 'specialty_cat']].describe()

# Assuming you want to print the summary table
print(summary_tbl)
