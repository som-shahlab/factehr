import pandas as pd
from ipdb import set_trace
from datasets import Dataset, load_from_disk, DatasetDict

# Path to your CSV file
csv_file_path = '/share/pi/nigam/rag-data/physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv'  # Update with your actual file path

# Read CSV file into a DataFrame
mimic_tbl = pd.read_csv(csv_file_path)

# Convert to tibble (not necessary in pandas, as DataFrame is similar to tibble)
# mimic_tbl = mimic_tbl.astype('object')  # Optionally convert to object type
# Count by CATEGORY and DESCRIPTION and filter by 'progress' in DESCRIPTION

progress_notes = mimic_tbl[mimic_tbl['DESCRIPTION'].str.lower().str.contains('progress', na=False)]
progress_counts = progress_notes.groupby(['CATEGORY', 'DESCRIPTION']).size().reset_index(name='count')
progress_counts_sorted = progress_counts.sort_values(by='count', ascending=False)


print("Filtered Progress Notes:")
#print(progress_counts_sorted)

#discharge = mimic_tbl[mimic_tbl.CATEGORY.isin(['Discharge summary'])][:250]

#discharge_train = mimic_tbl[mimic_tbl.CATEGORY.isin(['Nursing/other'])][-250:]

#discharge_test = mimic_tbl[mimic_tbl.CATEGORY.isin(['Nursing/other'])][:250]

discharge = progress_notes[:235]
#save this csv as a json 


# dataset_dict_train = {
#     'id': discharge_train['SUBJECT_ID'].tolist(),
#     'text': discharge_train['TEXT'].tolist(),
#     # Add more columns as needed
# }

dataset_dict_test = {
    'id': discharge['SUBJECT_ID'].tolist(),
    'text': discharge['TEXT'].tolist(),
    # Add more columns as needed
}


# Create a Hugging Face Dataset
# dataset_train = Dataset.from_dict(dataset_dict_train)
# dataset = DatasetDict({"train": dataset_train})

dataset_test = Dataset.from_dict(dataset_dict_test)
dataset = DatasetDict({"test": dataset_test})

set_trace()
dataset.save_to_disk(f"/share/pi/nigam/rag-data/mimiciii/progress_note.hf")

# # Count by CATEGORY
# category_counts = mimic_tbl.groupby('CATEGORY').size().reset_index(name='count')
# category_counts_sorted = category_counts.sort_values(by='count', ascending=False)


# # sample 10 from each note type: Progress Note, Discharge summary, nursing notes

# print("\nCategory Counts:")
# print(category_counts_sorted)


