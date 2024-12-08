""" 
Load legacy FactEHR document samples and JOIN with the original datasets to  
establish provenance of the documents. Guiding principles:

- Use the original dataset distribution files to sample the documents.
- Preserve all row information from the original dataset.
- Use the `note_id` as the primary key to JOIN the datasets.

NOTE / TODO The Dask code is a memory hog and could be replaced for simplicity.

Example usage:

python scripts/hotfixes/get_note_provenance.py \
--path_to_legacy /Users/jfries/Desktop/just-the-facts-doc/raw_20240812 \
--path_to_input data/datasets/raw/ \
--path_to_output data/datasets/


python scripts/hotfixes/get_note_provenance.py \
--path_to_legacy /Users/jfries/Desktop/hotfix/raw_20240812 \
--path_to_input data/datasets/raw/ \
--path_to_output data/datasets/


"""

import gc
import os
import re
import sys
import glob
import timeit
import logging
import argparse
import zipfile
from pathlib import Path
import collections
import numpy as np
import pandas as pd
from typing import Iterable, List, TypeVar, Set, Any, Dict
import dask.dataframe as dd
from dask.distributed import Client
from factehr.utils import hash_text
from datasets import Dataset, load_from_disk, DatasetDict


########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("note_summary_stats.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Generate corpus from entailemnt pairs")
parser.add_argument(
    "-i",
    "--path_to_input",
    type=str,
    help="Path to the input directory containing files to process",
    # required=True,
)
parser.add_argument(
    "-l",
    "--path_to_legacy",
    type=str,
    help="Path to the input directory containing legacy sampled MIMIC data",
    # required=True,
)
parser.add_argument(
    "-o",
    "--path_to_output",
    type=str,
    help="Path to the output JSON file",
    # required=True,
)


def join_on_text_hash(
    file_path: str,
    hash_keys: Set[str],
    join_on: str = "TEXT",
    dtype: Dict[str, Any] = None,
    chunksize: int = 100000,
):
    """Recover the original MIMIC-III note rows using text hashes."""
    joined = []
    for df in pd.read_csv(
        file_path, compression="infer", dtype=dtype or {}, chunksize=chunksize
    ):
        # create text hash of note text
        df["doc_id"] = df[join_on].apply(hash_text)
        # find exact note text matches
        joined.append(df[df.doc_id.isin(hash_keys)])
    return pd.concat(joined)


def dump_mimic_cxr_notes(path_to_input: str, path_to_output: str):
    """
    Load MIMIC-CXR data and dump to disk.
    """

    def text_files_in_zip(zip_file_path):
        """
        Generator that yields the contents of each text file in the given ZIP archive.

        :param zip_file_path: Path to the ZIP file.
        :yield: The content of each text file as a string.
        """
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            for file_name in zip_file.namelist():
                with zip_file.open(file_name) as file:
                    # Assuming the files are text files encoded in UTF-8
                    yield file_name, file.read().decode("utf-8")

    mimic_cxr_map = {}
    for name, text in text_files_in_zip(path_to_input):
        if not name.endswith(".txt"):
            continue

        # files/p11/p11449781/s54204103.txt
        # subject_id / study_id
        fields = name.split("/")
        subject_id = re.search(r"p(\d{3,})[/]", name)
        study_id = re.search(r"s(\d{3,})\.txt", name)
        mimic_cxr_map[(subject_id.group(1), study_id.group(1), name)] = text

    # Save to disk
    data = []
    for key in mimic_cxr_map:
        data.append(
            {
                "DATASET": "mimic-cxr-2.1.0",
                "SUBJECT_ID": key[0],
                "STUDY_ID": key[1],
                "FILE_NAME": key[2],
                "TEXT": mimic_cxr_map[key],
            }
        )

    pd.DataFrame(data).to_csv(path_to_output, index=False)


def load_legacy_hf_datasets(path_to_legacy: str):
    """Load legacy dcoument data (stored as Hugging Face datasets) from disk.

    Parameters
    ----------
    path_to_legacy : str
        The path to the legacy data directory.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the loaded data.
    """
    data = []
    for file_path in glob.glob(f"{path_to_legacy}/*.hf"):
        dataset = load_from_disk(file_path)
        dataset_name = file_path.split("/")[-1].split(".")[0]

        for split in dataset:
            for item in dataset[split]:
                hash_key = hash_text(item["text"])
                data.append(
                    {
                        "id": item["id"],
                        "dataset": dataset_name,
                        "doc_id": hash_key,
                        "split": split,
                        "text": item["text"],
                    }
                )

    return pd.DataFrame(data)


def dask_large_merge(
    input_path: str,
    merge_df: pd.DataFrame,
    n_workers: int = 1,
    memory_limit: str = "12GB",
):
    """Join a small DataFrame with a large Dask DataFrame in parallel.
    We use Dask to parallelize the merge operation across multiple CPU cores.

    TODO key names across tables is a hacky solution

    Parameters
    ----------
    medalign_input_path : str
        file path to the medalign data
    legacy_df : pd.DataFrame
        legacy data
    n_workers : int, optional
        number of CPUs, by default 4

    Returns
    -------
    pd.DataFrame
        joined data
    """
    config = {
        # Spill to disk when 60% memory is used
        "distributed.worker.memory.target": 0.6,
        # Start spilling to disk when 80% memory is used
        "distributed.worker.memory.spill": 0.8,
    }
    # Specify the number of workers (CPU cores) Dask should use
    client = Client(n_workers=n_workers, memory_limit=memory_limit, config=config)

    try:
        # Read the Dask DataFrame
        dask_df = dd.read_parquet(input_path)

        # Define the merge function to apply on each partition
        def merge_function(chunk):
            # only retrain the source chunk's rows
            return chunk[chunk["note_id"].isin(merge_df["id"])]

        # Apply the merge function to each partition in parallel
        merged_df = dask_df.map_partitions(merge_function)

        # Compute and return the final result
        return merged_df.compute()

    finally:
        # Ensure the Dask client is closed after computation
        client.close()


def brat_to_df(path_to_input_dir: str) -> pd.DataFrame:
    """Convert BRAT annotated text files to a pandas DataFrame."""
    data = []
    for file_path in glob.glob(f"{path_to_input_dir}/*.txt"):
        with open(file_path, "r") as file:
            text = file.read()
            data.append({"note_text": text})
    return pd.DataFrame(data)


def main():

    args = parser.parse_args()

    n_legacy_docs = 0

    # validate we have all legacy datasets
    for name in ["mimiciii", "coral", "medalign"]:
        if not os.path.exists(Path(args.path_to_legacy) / name):
            logger.error(f"Missing legacy dataset: {name}")
            sys.exit(1)

    # =========================================================================
    # 1. Load legacy MIMIC document samples
    # =========================================================================
    legacy_df = load_legacy_hf_datasets(Path(args.path_to_legacy) / "mimiciii")
    n_legacy_docs += len(legacy_df)
    logger.info(f"Loaded {len(legacy_df)} legacy MIMIC document samples")

    # Source MIMIC-III, MIMIC-CXR data, and CORAL data+
    mimic_iii_input_path = "physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
    mimic_cxr_input_path = "physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
    mimic_cxr_input_path = Path(args.path_to_input) / mimic_cxr_input_path
    mimic_iii_input_path = Path(args.path_to_input) / mimic_iii_input_path

    # Dump MIMIC-CXR to CSV
    mimic_cxr_output_path = Path(args.path_to_output) / "mimic-cxr-2.1.0_notes.csv"
    if not os.path.exists(mimic_cxr_output_path):
        dump_mimic_cxr_notes(mimic_cxr_input_path, mimic_cxr_output_path)

    mimic_cxr_dtypes = {
        "DATASET": "object",
        "SUBJECT_ID": "Int64",
        "STUDY_ID": "Int64",
        "FILE_NAME": "object",
        "TEXT": "object",
        "doc_id": "object",
    }
    # TODO We can add parse_dates=["CHARTDATE", "CHARTTIME", "STORETIME"]
    mimiciii_dtypes = {
        "ROW_ID": "Int64",
        "SUBJECT_ID": "Int64",
        "HADM_ID": "Int64",
        "CHARTDATE": "object",
        "CHARTTIME": "object",
        "STORETIME": "object",
        "CATEGORY": "object",
        "DESCRIPTION": "object",
        "CGID": "Int64",
        "ISERROR": "boolean",
        "TEXT": "object",
        "doc_id": "object",
    }

    # Join MIMIC-CXR and MIMIC-III with legacy MIMIC data
    hash_keys = set(legacy_df.doc_id.to_list())
    mimic_cxr_df = join_on_text_hash(
        mimic_cxr_output_path, hash_keys, dtype=mimic_cxr_dtypes
    )
    mimic_iii_df = join_on_text_hash(
        mimic_iii_input_path, hash_keys, dtype=mimiciii_dtypes
    )

    # Save to disk
    if not os.path.exists(Path(args.path_to_output) / "corpora"):
        os.makedirs(Path(args.path_to_output) / "corpora")

    mimic_cxr_df.to_csv(
        Path(args.path_to_output) / "corpora" / "mimic-cxr_notes.csv", index=False
    )
    mimic_iii_df.to_csv(
        Path(args.path_to_output) / "corpora" / "mimiciii_notes.csv", index=False
    )
    logger.info(
        f"Matched {len(mimic_cxr_df) + len(mimic_iii_df)}/{len(legacy_df)} "
        f"documents MIMIC-CXR={len(mimic_cxr_df)} MIMIC-III={len(mimic_iii_df)}"
    )

    # free up memory
    del mimic_cxr_df
    del mimic_iii_df
    gc.collect()

    # =========================================================================
    # 2. Load legacy CORAL document samples
    # =========================================================================
    # annoatated and unannotated datasets are DISJOINT

    legacy_df = load_legacy_hf_datasets(Path(args.path_to_legacy) / "coral")
    n_legacy_docs += len(legacy_df)
    logger.info(f"Loaded {len(legacy_df)} legacy CORAL document samples")

    # coral_idx,Sex,UCSFDerivedRaceEthnicity_X,BirthDate,note_text,doc_id
    coral_dtypes = {
        "coral_idx": "Int64",
        "Sex": "object",
        "UCSFDerivedRaceEthnicity_X": "object",
        "BirthDate": "object",
        "note_text": "object",
        "doc_id": "object",
    }

    # annotated notes
    coral_annotated_input_dir = (
        "physionet.org/files/curated-oncology-reports/1.0/coral/annotated/breastca/"
    )
    coral_breastca_ann_df = brat_to_df(
        Path(args.path_to_input) / coral_annotated_input_dir
    )
    coral_breastca_ann_df["doc_id"] = coral_breastca_ann_df["note_text"].apply(
        hash_text
    )
    coral_breastca_ann_df.to_csv(
        Path(args.path_to_input) / "coral_breastca_ann.csv", index=False
    )
    coral_annotated_input_dir = (
        "physionet.org/files/curated-oncology-reports/1.0/coral/annotated/pdac/"
    )
    coral_pdac_ann_df = brat_to_df(Path(args.path_to_input) / coral_annotated_input_dir)
    coral_pdac_ann_df["doc_id"] = coral_breastca_ann_df["note_text"].apply(hash_text)
    coral_pdac_ann_df.to_csv(
        Path(args.path_to_input) / "coral_pdac_ann.csv", index=False
    )

    # TODO: add back unannotated notes
    # unannotated notes
    # coral_breastca_input_path = "physionet.org/files/curated-oncology-reports/1.0/coral/unannotated/data/breastca_unannotated.csv"
    # coral_pdac_input_path = "physionet.org/files/curated-oncology-reports/1.0/coral/unannotated/data/pdac_unannotated.csv"
    # coral_breastca_input_path = Path(args.path_to_input) / coral_breastca_input_path
    # coral_pdac_input_path = Path(args.path_to_input) / coral_pdac_input_path

    # unannotated notes
    # coral_breastca_df = pd.read_csv(coral_breastca_input_path)
    # coral_breastca_df["doc_id"] = coral_breastca_df["note_text"].apply(hash_text)
    # coral_pdac_df = pd.read_csv(coral_pdac_input_path)
    # coral_pdac_df["doc_id"] = coral_pdac_df["note_text"].apply(hash_text)

    # filter to original sampled docs
    hash_keys = set(legacy_df.doc_id.to_list())

    coral_breastca_df = join_on_text_hash(
        Path(args.path_to_input) / "coral_breastca_ann.csv",
        hash_keys,
        join_on="note_text",
        dtype=coral_dtypes,
    )
    coral_pdac_df = join_on_text_hash(
        Path(args.path_to_input) / "coral_pdac_ann.csv",
        hash_keys,
        join_on="note_text",
        dtype=coral_dtypes,
    )

    # Save to disk
    if not os.path.exists(Path(args.path_to_output) / "corpora"):
        os.makedirs(Path(args.path_to_output) / "corpora")

    coral_breastca_df.to_csv(
        Path(args.path_to_output) / "corpora" / "coral-breastca_ann_notes.csv",
        index=False,
    )
    coral_pdac_df.to_csv(
        Path(args.path_to_output) / "corpora" / "coral-pdac_ann_notes.csv", index=False
    )
    logger.info("Saved CORAL breastca and pdac notes to disk")

    # =========================================================================
    # 3. Load legacy MedAlign document samples
    # =========================================================================
    # In MedAlign, the ID is the original `note_id` so we can do a standard join
    legacy_df = load_legacy_hf_datasets(Path(args.path_to_legacy) / "medalign")
    n_legacy_docs += len(legacy_df)
    logger.info(f"Loaded {len(legacy_df)} legacy MedAlign document samples")

    medalign_input_path = "medalign-aaai_confidential_notes"
    medalign_input_path = Path(args.path_to_input) / medalign_input_path

    # NOTE 16GB machines have memeory issues with 2+ workers
    # 2 workers - Execution time: 14.79 seconds
    medalign_df = dask_large_merge(medalign_input_path, legacy_df, n_workers=1)

    # dump medalign to disk
    medalign_df["doc_id"] = medalign_df["note_text"].apply(hash_text)
    medalign_output_path = (
        Path(args.path_to_output) / "corpora" / "medalign-aaai_release-notes.csv"
    )
    medalign_df.to_csv(medalign_output_path, index=False)
    logger.info(f"Matched {len(medalign_df)}/{len(legacy_df)} documents from MedAlign")

    print(f"Total legacy documents: {n_legacy_docs}")


if __name__ == "__main__":

    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")
