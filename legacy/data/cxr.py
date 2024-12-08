import os
import shutil
import csv
import pandas as pd
from typing import Dict, Tuple, Union
from functools import lru_cache
from pathlib import Path
import shutil
from tqdm import tqdm


REPORTS_DIR = Path("/share/pi/nigam/rag-data/physionet.org/files/mimic-cxr/2.0.0/files"  )
SPLITS_CSV_PATH = Path("/share/pi/nigam/rag-data/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv"  )
OUTPUT_DIR = Path("/share/pi/nigam/rag-data/cxr_splits/")


@lru_cache
def get_subject_id_part_map(parts_dir: Path) -> Dict[str, str]:
    """Get the mapping of p<subject_id>: p<part>"""
    output_dict = {}

    for part in os.listdir(parts_dir):
        for subject_dir in os.listdir(parts_dir / part):
            output_dict[subject_dir] = part

    return output_dict


def copy_file_to_split(subject_id: str, study_id: str, split: str) -> Tuple[bool, Union[str, None]]:
    """Copies files from REPORTS_DIR to OUTPUT_DIR based on split"""
    subject_id_part_map = get_subject_id_part_map(REPORTS_DIR)

    part_dir = subject_id_part_map[f"p{subject_id}"]
    study_file = REPORTS_DIR / f"{part_dir}/p{subject_id}/s{study_id}.txt"

    if not os.path.exists(study_file):
        return False, f"not found {study_file}"
    
    destination_file = OUTPUT_DIR / f"{split}/s{study_id}.txt"
    destination_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copyfile(study_file, destination_file)
    except Exception as exc:
        return False, exc.__traceback__
    
    return True, None


def copy_data_to_split_dir():
    """Copies data to directories organizied by splits."""

    print(f"Copying data from {REPORTS_DIR} to {OUTPUT_DIR}")

    og_splits_df = pd.read_csv(SPLITS_CSV_PATH)
    splits_df = og_splits_df.drop("dicom_id", axis=1).drop_duplicates("study_id").reset_index(drop=True).astype(str)

    # get train, val and test splits in to a list
    
    tqdm.pandas(desc="Copying files: ")
    splits_df[["is_success", "err_msg"]] = splits_df.progress_apply(
        lambda row: copy_file_to_split(subject_id=row["subject_id"], study_id=row["study_id"], split=row["split"]), 
        result_type='expand', 
        axis="columns"
    )

    unsuccessful_copies = splits_df[~splits_df["is_success"]]

    print(unsuccessful_copies)

    if len(unsuccessful_copies) > 0:
        unsuccessful_copies_file = OUTPUT_DIR / "unsuccessful_copies.csv"
        unsuccessful_copies_file.parent.mkdir(exist_ok=True)
        unsuccessful_copies.to_csv(unsuccessful_copies_file)
        print(f"Saved unsuccessful copies to {unsuccessful_copies_file}")


if __name__ == "__main__":
    copy_data_to_split_dir()
