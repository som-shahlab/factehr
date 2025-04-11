""" 
NLP Prepocessing of Clinical Text Data

Fact decomposition of clinical text data involves sentence splitting to
generate entailment pairs. This script preprocesses clinical text data 
using clinical text tokenization and sentence splitting.

Example Usage:

python scripts/build_docbin_dataset.py \
--path_to_input data/datasets/corpora/v2/ \
--path_to_output data/datasets/ \
--n_procs 4 \
--batch_size 100 \
--nlp_framework trove \
--file_name_prefix factehr_v2_{ts}

"""

import re
import glob
import timeit
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Generator, DefaultDict, Dict, Any, Tuple, List, NamedTuple

import spacy
from spacy.util import is_package, get_package_path
from spacy.tokens import Doc, DocBin
from factehr.nlp.tokenizer import ct_tokenizer
from factehr.nlp.sbd import ct_sentence_splitter, ct_fast_sentence_splitter

########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler("nlp_preprocessing.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


########## Argparse Setup ##########

parser = argparse.ArgumentParser(
    description="Batch NLP preprocessing of clinical documents"
)
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
    help="Path to the output file",
    required=True,
)

parser.add_argument(
    "--primary_key",
    type=str,
    help="Primary key column name in the input files",
    default="doc_id",
)

parser.add_argument(
    "--text_column_name",
    type=str,
    help="Text column name in the input CSV files",
    default="note_text|TEXT",
)

parser.add_argument(
    "-n",
    "--n_procs",
    type=int,
    help="Number of processes to use for multiprocessing",
    default=1,
)

parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    help="Batch size for processing documents",
    default=100,
)

parser.add_argument(
    "-f",
    "--nlp_framework",
    type=str,
    choices=["spacy", "trove", "medspacy"],
    help="Specify the tokenizer + sentence splitter to use ('spacy', 'trove' or 'medspacy')",
    default="trove",
)

parser.add_argument(
    "--file_name_prefix",
    type=str,
    help="Prefix name of prompted dataset file",
    default="factehr_{ts}",
)

########## Dataloader and NLP Framework Setup ##########


def dataloader(
    path_to_input: str,
    file_ext: str,
    primary_key: str,
    note_text: str,
    include_all_cols: bool = True,
) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
    """Load clinical text data from tab delimitted files"""
    file_list = glob.glob(path_to_input + f"*.{file_ext}")
    assert len(file_list) > 0, "No files found in the input directory"

    for file_path in file_list:
        # dataset = file_path.split("/")[-1].split(".")[0].split("_")[0]
        df = pd.read_csv(file_path)
        # match note text column key name
        text_column_name = [
            col for col in df.columns if re.match(f"^{note_text}$", col)
        ]
        if len(text_column_name) == 0:
            raise ValueError(f"Column name {note_text} not found in dataframe")

        text_column_name = text_column_name[0]

        for row in df.itertuples():
            doc_id = getattr(row, primary_key)
            text = getattr(row, text_column_name)
            # include all columns in metadata or not
            metadata = row._asdict() if include_all_cols else {}
            metadata.update({"doc_id": doc_id})

            yield (text, metadata)


def get_nlp_pipeline(nlp_framework: str):
    """Get the NLP pipeline based on the specified framework"""
    if nlp_framework == "spacy":
        if not is_package("en_core_web_sm"):
            spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    elif nlp_framework == "trove":
        nlp = spacy.blank("en")
        nlp.tokenizer = ct_tokenizer(nlp)
        nlp.add_pipe("fast_ct_sentence_splitter")

    elif nlp_framework == "medspacy":
        # load here to avoid breaking multiprocessing
        from medspacy.custom_tokenizer import create_medspacy_tokenizer
        from medspacy.sentence_splitting import PyRuSHSentencizer

        nlp = spacy.blank("en")
        nlp.tokenizer = create_medspacy_tokenizer(nlp)
        nlp.add_pipe("medspacy_pyrush")

    else:
        raise ValueError("Invalid NLP framework specified")

    return nlp


########## Main ##########


def main():

    args = parser.parse_args()

    # check for multiprocessing support
    if args.nlp_framework == "medspacy" and args.n_procs > 1:
        raise ValueError(
            "MedSpaCy does not support multiprocessing for batch processing"
        )

    # generator for clinical text data (directly from tab-delimitted files)
    corpus = dataloader(
        args.path_to_input, "csv", args.primary_key, args.text_column_name
    )
    # load the NLP pipeline
    nlp = get_nlp_pipeline(args.nlp_framework)

    # batch processing with multiprocessing
    documents = []
    doc_tuples = nlp.pipe(
        corpus,
        n_process=args.n_procs,
        batch_size=args.batch_size,
        as_tuples=True,
    )
    # generate spacy documents
    for doc, metadata in doc_tuples:
        doc.user_data = metadata
        documents.append(doc)
    logger.info(f"Processed {len(documents)} documents")

    # output to docbin
    doc_bin = DocBin(store_user_data=True)
    for doc in documents:
        doc_bin.add(doc)

    # save the processed documents to disk with versioned filename
    if "{ts}" in args.file_name_prefix:
        version_str = datetime.now().strftime("%Y%m%d")
        args.file_name_prefix = args.file_name_prefix.format(ts=version_str)

    output_file_path = f"{args.path_to_output}/{args.file_name_prefix}.docbin"
    bytes_data = doc_bin.to_bytes()
    with open(output_file_path, "wb") as fp:
        fp.write(bytes_data)

    logger.info(f"Saved processed documents to {output_file_path}")


if __name__ == "__main__":
    # change this for getting run statistics
    num_runs = 1
    wall_times = np.array(
        [
            timeit.timeit("main()", setup="from __main__ import main", number=1)
            for _ in range(num_runs)
        ]
    )
    logger.info(
        f"Execution time: Mean (SD) = {np.mean(wall_times):.1f} ({np.std(wall_times):.1f}) seconds"
    )
