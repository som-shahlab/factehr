""" 
This script defines our sampling regieme for source clinical note datasets.

|                      | MIMIC-CXR | MIMIC-III | MedAlign | Coral |
|----------------------|-----------|-----------|----------|-------|
| progress_notes       |      -    |     250   |    250   |  172  |
| nursing_notes        |      -    |     250   |    137   |   -   |
| discharge_summaries  |      -    |     250   |    113   |   -   |
| procedure notes      |      -    |      *    |    250   |   -   |
| - radiology_reports  |     250   |     250   |     *    |   -   |


Example usage:

python scripts/sample_source_datasets.py \
--path_to_input data/datasets/raw/ \
--path_to_output data/datasets/corpora/v2/ \
--tokenizer tiktoken \
--min_doc_length 64 \
--max_doc_length 3840

"""

import os
import re
import glob
import timeit
import logging
import argparse
import zipfile
import tiktoken
import functools
import collections
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, List, TypeVar, Set, Any, Dict, Callable
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
    required=True,
)

parser.add_argument(
    "-o",
    "--path_to_output",
    type=str,
    help="Path to the output JSON file",
    required=True,
)

parser.add_argument(
    "--min_doc_length",
    type=int,
    help="minimum document length in white space tokens",
    default=None,
)

parser.add_argument(
    "--max_doc_length",
    type=int,
    help="maximum document length in white space tokens",
    default=None,
)

parser.add_argument(
    "-s", "--seed", type=int, help="Random seed for reproducibility", default=123456
)

parser.add_argument(
    "--tokenizer",
    type=str,
    help="Tokenizer to use for tokenization",
    default="tiktoken",
)

parser.add_argument(
    "-n",
    "--file_name_prefix",
    type=str,
    help="Prefix name of prompted dataset file",
    default="factehr",
)

parser.add_argument(
    "--max_strata_size",
    type=int,
    help="Max sample size to sample per strata",
    default=250,
)


########## Dataloaders ##########


def load_mimic_cxr_notes(path_to_input: str):
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

    # convert to dataframe
    data = []
    for key in mimic_cxr_map:
        data.append(
            {
                "DATASET": "mimic-cxr-2.1.0",
                "SUBJECT_ID": int(key[0]),
                "STUDY_ID": int(key[1]),
                "FILE_NAME": key[2],
                "TEXT": mimic_cxr_map[key],
            }
        )

    return pd.DataFrame(data)


def brat_to_df(path_to_input_dir: str) -> pd.DataFrame:
    """Convert BRAT annotated text files to a pandas DataFrame."""
    data = []
    for file_path in glob.glob(f"{path_to_input_dir}/*.txt"):
        with open(file_path, "r") as file:
            text = file.read()
            data.append({"note_text": text})
    return pd.DataFrame(data)


def get_mimic_note_type_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Get the note type summary from a MIMIC DataFrame."""
    note_types = collections.Counter()
    for row in df.itertuples():
        note_type = (row.CATEGORY, row.DESCRIPTION)
        note_types[note_type] += 1
    return dict(note_types)


def tiktoken_count(text, encoder=None) -> int:
    """Token count for a given text using TikToken."""
    if not encoder:
        encoder = tiktoken.encoding_for_model("gpt-4o")
    return len(encoder.encode(text))


def estimate_token_count(text) -> int:
    """Fast (but inaccurate for clinical text) token count estimation."""
    word_count = len(text.split())
    # Estimate using 4/3 ratio
    token_count = word_count * 4 / 3
    return token_count


def filter_by_token_length(
    df,
    text_column: str,
    min_token_length: int = 0,
    max_token_length: int = 100000,
    tokenizer: str = "tiktoken",
    drop_est_token_count: bool = False,
):
    """Filters a DataFrame based on the est. token length of `text_column`."""
    if tokenizer == "tiktoken":
        enc = tiktoken.encoding_for_model("gpt-4o")
        token_count_func = functools.partial(tiktoken_count, encoder=enc)
    else:
        token_count_func = estimate_token_count

    # use dask for parallel processing
    # meta is required to know the output typing
    meta = pd.Series(dtype="int64")
    ddf = dd.from_pandas(df, npartitions=4)
    # Estimate token count for each row in the text column
    ddf["est_token_count"] = ddf[text_column].map_partitions(
        lambda x: x.apply(token_count_func), meta=meta
    )
    df = ddf.compute()

    # min/max length constraints
    constraints = (df["est_token_count"] > min_token_length) & (
        df["est_token_count"] < max_token_length
    )
    # filter the DataFrame based on the estimated token count
    filtered_df = df[constraints]
    # drop the 'est_token_count' column after filtering
    if drop_est_token_count:
        filtered_df.drop(columns=["est_token_count"])
    return filtered_df


def sample_dataframe(
    dataframe: pd.DataFrame,
    sample_n: int,
    rng: np.random.Generator,
    min_token_len: int = 0,
    max_token_len: int = 100000,
    text_column: str = "TEXT",
    tokenizer: str = "tiktoken",
):
    if min_token_len is not None or max_token_len is not None:
        sample_df = filter_by_token_length(
            dataframe.copy(),
            text_column=text_column,
            min_token_length=min_token_len,
            max_token_length=max_token_len,
            tokenizer=tokenizer,
        )

    sample_size = min(len(sample_df), sample_n)
    sample_idxs = rng.choice(sample_df.index, size=sample_size, replace=False)
    sample_df = sample_df.loc[sample_idxs]

    return sample_df


########## Main ##########


def main():

    args = parser.parse_args()

    # Create output folder
    if not os.path.exists(Path(args.path_to_output)):
        os.makedirs(Path(args.path_to_output))

    max_strata_sample_n = args.max_strata_size

    # =========================================================================
    # 1. Sample MIMIC-CXR
    #  - Radiology reports (chest radiographs)
    # =========================================================================
    # Load MIMIC-CXR notes from a nested directory structured zip file
    mimic_cxr_input_path = (
        Path(args.path_to_input)
        / "physionet.org/files/mimic-cxr/2.1.0/mimic-cxr-reports.zip"
    )
    mimic_cxr_df = load_mimic_cxr_notes(mimic_cxr_input_path)

    # load canononical split definitions
    mimic_cxr_splits_input_path = (
        Path(args.path_to_input)
        / "physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz"
    )
    splits_df = pd.read_csv(
        mimic_cxr_splits_input_path,
        compression="infer",
        dtype={"subject_id": "Int64", "study_id": "Int64"},
    )
    # merge splits with mimic_cxr
    mimic_cxr_df = pd.merge(
        mimic_cxr_df,
        splits_df,
        left_on=["SUBJECT_ID", "STUDY_ID"],
        right_on=["subject_id", "study_id"],
    )

    # sample from the test set
    mimic_cxr_test_df = mimic_cxr_df.query("split == 'test'")

    # force unique note text
    mimic_cxr_test_df = mimic_cxr_test_df.drop_duplicates(subset=["TEXT"])

    # sample notes
    strata_df = sample_dataframe(
        mimic_cxr_test_df,
        sample_n=max_strata_sample_n,
        rng=np.random.default_rng(args.seed),
        min_token_len=args.min_doc_length,
        max_token_len=args.max_doc_length,
        tokenizer=args.tokenizer,
    )
    strata_df["note_type"] = "radiology_report"
    strata_df["doc_id"] = strata_df["TEXT"].apply(hash_text)
    strata_df["dataset_name"] = "mimic-cxr-2.1.0"

    # Write MIMIC-CXR to CSV
    mimic_cxr_output_path = (
        Path(args.path_to_output) / f"{args.file_name_prefix}_mimic-cxr-2.1.0_notes.csv"
    )

    strata_df.to_csv(mimic_cxr_output_path, index=False)
    logger.info(f"Sampled {len(strata_df)} from MIMIC-CXR[test]:radiology_report")

    # =========================================================================
    # 2. Sample MIMIC-III
    #  - Discharge summaries
    #  - Progress notes
    #  - Radiology reports
    #  - Nursing notes
    # =========================================================================

    # Load MIMIC-III
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
    mimic_iii_input_path = "physionet.org/files/mimiciii/1.4/NOTEEVENTS.csv.gz"
    mimic_iii_input_path = Path(args.path_to_input) / mimic_iii_input_path
    mimiciii_df = pd.read_csv(
        mimic_iii_input_path, compression="infer", dtype=mimiciii_dtypes
    )

    # definitions for stratifying notes
    strata_defs = {
        "discharge_summary": "CATEGORY == 'Discharge summary' and DESCRIPTION == 'Report'",
        "progress_note": "DESCRIPTION.str.contains('Progress Note', case=False)",
        "radiology_report": "CATEGORY == 'Radiology'",
        "nursing_note": "CATEGORY == 'Nursing/other' and DESCRIPTION == 'Report'",
    }

    # reset our random number generator
    rng = np.random.default_rng(args.seed)
    stratified = {}
    sample_set = set()

    # stratified sample by note type
    for strata, query in strata_defs.items():
        strata_df = mimiciii_df.query(query)
        # make certain this strata is disjoint from all others
        if set(strata_df.ROW_ID).intersection(sample_set):
            dups = set(strata_df.ROW_ID).intersection(sample_set)
            logger.error(f"Duplicate samples found in {strata}: {len(dups)}")
            continue
        sample_set.update(set(strata_df.ROW_ID))

        stratified[strata] = sample_dataframe(
            strata_df,
            sample_n=max_strata_sample_n,
            rng=np.random.default_rng(args.seed),
            min_token_len=args.min_doc_length,
            max_token_len=args.max_doc_length,
            tokenizer=args.tokenizer,
        )

        stratified[strata]["note_type"] = strata
        stratified[strata]["doc_id"] = strata_df["TEXT"].apply(hash_text)
        stratified[strata]["dataset_name"] = "mimic-iii-1.4"

        logger.info(f"Sampled {len(stratified[strata])} from MIMIC-III:{strata}")

    # clean up memory
    del mimiciii_df

    # write sample to disk
    stratified_df = pd.concat(stratified.values()).reset_index(drop=True)
    stratified_df.to_csv(
        Path(args.path_to_output) / f"{args.file_name_prefix}_mimiciii_notes.csv",
        index=False,
    )

    # =========================================================================
    # 2. Sample CORAL
    #  - Progress Notes
    # =========================================================================

    coral_data_root = (
        Path(args.path_to_input)
        / "physionet.org/files/curated-oncology-reports/1.0/coral/"
    )

    # annotated and unannotated notes
    coral_breastca_ann_df = brat_to_df(coral_data_root / "annotated/breastca/")
    coral_pdac_ann_df = brat_to_df(coral_data_root / "annotated/pdac/")
    coral_pdac_df = pd.read_csv(
        coral_data_root / "unannotated/data/pdac_unannotated.csv"
    )
    coral_breastca_df = pd.read_csv(
        coral_data_root / "unannotated/data/breastca_unannotated.csv"
    )

    coral_df = pd.concat(
        [coral_breastca_ann_df, coral_pdac_ann_df, coral_pdac_df, coral_breastca_df]
    ).reset_index(drop=True)

    coral_df = sample_dataframe(
        coral_df,
        sample_n=max_strata_sample_n,
        rng=np.random.default_rng(args.seed),
        min_token_len=args.min_doc_length,
        max_token_len=args.max_doc_length,
        text_column="note_text",
        tokenizer=args.tokenizer,
    )

    # add our metadata
    coral_df["note_type"] = "progress_note"
    coral_df["dataset_name"] = "coral"
    coral_df["doc_id"] = coral_df["note_text"].apply(hash_text)

    coral_df.to_csv(
        Path(args.path_to_output) / f"{args.file_name_prefix}_coral_notes.csv",
        index=False,
    )
    logger.info(f"Sampled {len(coral_df)} from CORAL:progress_note")

    # =========================================================================
    # 3. Sample MedAlign
    #  - Discharge summaries
    #  - Progress notes
    #  - Procedure notes
    #  - Nursing notes
    # =========================================================================

    medalign_input_path = (
        Path(args.path_to_input) / "medalign-aaai_release_notes.parquet"
    )
    medalign_df = pd.read_parquet(medalign_input_path, engine="pyarrow")

    strata_defs = {
        "discharge_summary": "note_title == 'discharge summary'",
        "progress_note": "note_title == 'progress notes'",
        "procedures": "note_title == 'procedures'",
        "nursing_note": "note_title == 'nursing note'",
    }

    stratified = {}
    for strata, query in strata_defs.items():
        strata_df = medalign_df.query(query)
        # force unique note text
        strata_df = strata_df.drop_duplicates(subset=["note_text"])
        # filter based on estimated token length
        stratified[strata] = sample_dataframe(
            strata_df,
            sample_n=max_strata_sample_n,
            rng=np.random.default_rng(args.seed),
            min_token_len=args.min_doc_length,
            max_token_len=args.max_doc_length,
            text_column="note_text",
            tokenizer=args.tokenizer,
        )
        # add our metadata
        stratified[strata]["note_type"] = strata
        stratified[strata]["doc_id"] = stratified[strata]["note_text"].apply(hash_text)
        stratified[strata]["dataset_name"] = "medalign_aaai_pre_v1"

        logger.info(f"Sampled {len(stratified[strata])} from MedAlign[test]:{strata}")

    stratified_df = pd.concat(stratified.values()).reset_index(drop=True)
    stratified_df.to_csv(
        Path(args.path_to_output) / f"{args.file_name_prefix}_medalign_notes.csv",
        index=False,
    )

    # =========================================================================
    #  Print Summary Statistics
    # =========================================================================
    dataframes = {
        file_path.split("/")[-1].split(".")[0]: pd.read_csv(file_path)
        for file_path in glob.glob(f"{args.path_to_output}/*.csv")
    }

    n_docs = 0
    for name, df in dataframes.items():
        logger.info(
            f"Summary statistics for {name}: N={len(df)} unique N={len(set(df.doc_id))}"
        )
        n_docs += len(df)
    logger.info(f"Total documents sampled: {n_docs}")


if __name__ == "__main__":

    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")
