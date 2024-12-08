""" 
Score entailment pairs for fact decomposition of documents. Results for 
"Just the Facts, Doc: Factual Decomposition of Clinical Notes" (Munnangi et al. 2024)

XXX entailment pairs generated from 2000 Notes X 5 LLMs X 4 Prompts

USAGE EXAMPLE:

python src/factehr/evaluation/score_entailment_pairs.py \
--path_to_input /Users/jfries/Desktop/just-the-facts-doc/entailment_final/ \
--path_to_output data/manuscript/entailment_pair_scores.tsv \
--map_unparseable_json 0 \
--filter_on note_type:discharge_summary

"""

import re
import glob
import json
import timeit
import hashlib
import logging
import argparse
import collections
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import Generator, DefaultDict, Dict, Any, Tuple, List, NamedTuple
from factehr.evaluation.entailment import entailment_proportion

########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("entailment_scores.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


########## Argparse Setup ##########


def parse_key_value_pair(s):
    """Parse key/value pairs delimitted by a colon"""
    try:
        key, value = s.split(":", 1)
        return key, value
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid key-value pair: '{s}'. Expected format is key:value."
        )


parser = argparse.ArgumentParser(
    description="Entailment evaluation using fact precision + fact recall"
)
parser.add_argument(
    "-i",
    "--path_to_input",
    type=str,
    help="Path to the input directory containing files to process",
    # required=True,
)
parser.add_argument(
    "-o",
    "--path_to_output",
    type=str,
    help="Path to the output CSV file",
    # required=True,
)
parser.add_argument(
    "--map_unparseable_json",
    choices=[-1, 0, 1],
    help="Default value for unparsable JSON output (-1 means ignore)",
    default=-1,
    type=int,
)

parser.add_argument(
    "--filter_on",
    nargs="+",  # allows multiple key-value pairs
    type=parse_key_value_pair,
    help="Key-value pairs to filter on, e.g., note_type:discharge_summary",
)

parser.add_argument(
    "--disable_validation_checks",
    action="store_true",
    help="Disable validation checks (WARNING do NOT use this by default)",
)

########## Trial Metadata Object Definition ##########


class Trial(NamedTuple):
    key: tuple
    prompt: str
    metric: str
    llm: str
    dataset: str
    note_type: str
    note_subtype: str
    file_path: str

    def __str__(self):
        key_str = ", ".join(str(k) for k in self.key)
        return f"Trial[key=({key_str}) metric='{self.metric}')]"


########## Data Loaders ##########


def load_jsonl(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """Load JSON LLM outputs

    Parameters
    ----------
    file_path : str
        Path to the JSONL file containing LLM outputs.

    Yields
    ------
    Dict[str, Any]
        A dictionary representing the JSON object from each line in the file.
    """
    with open(file_path, "r") as file:
        for line in file:
            yield json.loads(line)


def get_trial_metadata(file_path: str) -> Trial:
    """Use directory nesting as implicit metadata definition for
    experimental variables.

    Parameters
    ----------
    file_path : str
        file path name of experiment JSON

    Returns
    -------
    Trial
        Trial named tuple containing all experiment named fields
    """
    remap_note_types = {"pdac": "progress_note", "breastca": "progress_note"}
    llm, dataset, note_type, fname = file_path.split("/")
    prompt = re.search(r"(PROMPT[12](\_ICL)*)", fname).group()
    metric = re.search(r"(recall|precision)", fname).group()

    # standardize note type names
    note_subtype = None
    if note_type in remap_note_types:
        note_subtype = note_type
        note_type = remap_note_types[note_type]

    key = (prompt, llm, dataset, note_type, note_subtype)
    return Trial(key, prompt, metric, llm, dataset, note_type, note_subtype, file_path)


def load_experiments(
    path_to_data: str, filter_on: Dict[str, List[str]] = None, validate: bool = True
) -> Dict[Tuple[str], List[Trial]]:
    """Load experiment metadata and paths to results JSONL files.

    Parameters
    ----------
    dir_name : str
        root directory of experiment result files
    filter_on : Dict[str, List[str]], optional
        Remove experiment trials that match the filter_on criteria, by default None
    validate : bool, optional
        read the result files and confirm all doc_ids are joinable across
        paired entailment experiments, by default True

    Returns
    -------
    Dict[Tuple[str], List[Trial]]
        experiment metadata definitions
    """
    filter_on = {} if filter_on is None else filter_on

    # 1. Load all experiment files
    filelist = glob.glob(f"{path_to_data}/*/*/*/*.json")
    experiments = collections.defaultdict(list)

    for file_path in filelist:
        relative_path = file_path.replace(path_to_data, "")
        try:
            trial = get_trial_metadata(relative_path)
            # filter out trials based on critera
            if not any(
                getattr(trial, key) in values for key, values in filter_on.items()
            ):
                experiments[trial.key].append(trial)
            else:
                logger.warning(f"SKIPPING {trial}")

        except Exception as e:
            logger.error(f"Error loading: {file_path}: {e}")

    if not validate:
        return experiments

    # 2. Validation checks
    # 2a. Do we have all our file pairs (precision and recall)
    rm = []
    for key, trials in experiments.items():
        entailment_metrics = [t.metric for t in trials]
        if len(entailment_metrics) != 2:
            rm.append(key)

    for key in rm:
        del experiments[key]
    logger.warning(f"Removed {len(rm)} experiments due to missing files..")

    # 2.b Do we have paired entailment results for each doc_id?
    # note, we do not *remove* incomplete entailment pairs here
    n_total, n_missing = 0, 0
    details = []
    for key, experiment in experiments.items():

        pairs = collections.defaultdict(set)
        for trial in experiment:
            file_path = f"{path_to_data}/{trial.file_path}"
            for item in load_jsonl(file_path):
                pairs[trial.metric].add(item["ID"])

        n_total += len(pairs["precision"].union(pairs["recall"]))
        # symmetric_difference
        missing = pairs["precision"] ^ pairs["recall"]
        n_missing += len(missing)

        if len(missing) > 0:
            details.append(key)
            logger.warning(f"Missing IDs for {key} n={len(missing)} IDs={missing}")

    logger.info(
        "Validation check complete: missing "
        f"{n_missing}/{n_total} ({n_missing/n_total*100:.1f})% entailment pairs"
    )

    return experiments


def get_doc_id_to_hash_map(file_path: str) -> Dict[int, str]:
    """Use document text to build MD5 hash for use as primary key.

    Parameters
    ----------
    file_path : str
        file path to experiment file containing full note text

    Returns
    -------
    Dict[int, str]
        mapping of integer id to a MD5 hash hex string
    """
    doc_index = {}
    for item in load_jsonl(file_path):
        hash_key = hashlib.md5(item["premise"].encode("utf-8")).hexdigest()
        doc_index[item["ID"]] = hash_key

    return doc_index


def load_entailment_results(
    file_path: str, map_unparseable_json: int = 0
) -> Dict[str, List[int]]:
    """Extract entailment scores and bin scores by document ID

    Parameters
    ----------
    file_path : str
        _description_
    map_unparseable_json : int, optional
        if the LLM did not return parsable JSON, this picks the default value
        we assign the entailment prediction. None means the item is skipped
        and not included in the scores, by default 0

    Returns
    -------
    Dict[str, List[int]]
        id mapped to list of entailment predictions
    """
    results_by_doc_id = collections.defaultdict(list)

    for item in load_jsonl(file_path):
        unparsable_json = "verdict" in item

        # skip unparsable JSON and do not include in calculation
        if unparsable_json and map_unparseable_json == -1:
            logger.warning(f"Skipping due to unparsable JSON mapping rule: {item}")
            continue

        score = map_unparseable_json if unparsable_json else item["entailment_pred"]
        results_by_doc_id[item["ID"]].append(score)

    return dict(results_by_doc_id)


########## Scoring ##########


def score_experiment(
    experiment: List[Trial], data_root: str, map_unparseable_json: int
):
    """Evaluate LLM entailment predictions using the LLM-as-a-judge paradigm.
    Given LLM generations for entailment predictions, compute fact precision
    and fact recall.

    Parameters
    ----------
    experiment : List[Trial]
        list of experiment/per-trial metadata to describe experiment run
    data_root : str
        root directory of experiment data

    Returns
    -------
    _type_
        _description_
    """
    errs = []
    results = collections.defaultdict(dict)

    for trial in experiment:

        file_path = f"{data_root}/{trial.file_path}"
        # LLM-as-a-judge predictions for entailment pairs
        judgements = load_entailment_results(
            file_path, map_unparseable_json=map_unparseable_json
        )

        if trial.metric == "precision":
            # build a primary key from the full note text
            doc_id_to_hash = get_doc_id_to_hash_map(file_path)

            for doc_id, hash_key in doc_id_to_hash.items():
                results[doc_id]["note_text_hash"] = hash_key

            for doc_id, llm_preds in judgements.items():
                # TODO add logic to load sample_weights when they exist
                sample_weights = np.ones(len(llm_preds))
                results[doc_id]["fact_precision"] = entailment_proportion(
                    llm_preds, sample_weights
                )
                results[doc_id]["fact_fp"] = llm_preds.count(0)
                results[doc_id]["n_facts"] = len(llm_preds)

        elif trial.metric == "recall":
            for doc_id, llm_preds in judgements.items():
                results[doc_id]["fact_recall"] = entailment_proportion(llm_preds)
                results[doc_id]["fact_fn"] = llm_preds.count(0)
                results[doc_id]["n_sents"] = len(llm_preds)

    for doc_id in results:
        try:
            # HACK integrity check
            if not all(
                key in results[doc_id] for key in ["fact_precision", "fact_recall"]
            ):
                # logger.warning(
                #     f"Skipping ID={doc_id} due to incomplete data {results[doc_id].keys()}"
                # )
                continue

            # compute f1 measure
            p = results[doc_id]["fact_precision"]
            r = results[doc_id]["fact_recall"]
            # if f1 is not defined, assign 0
            if np.isnan(p) or np.isnan(r) or (p + r) == 0:
                results[doc_id]["fact_f1"] = 0.0
            else:
                results[doc_id]["fact_f1"] = 2 * p * r / (p + r)
            # add our experiment metadata
            for key in ["prompt", "dataset", "note_type", "note_subtype", "llm"]:
                results[doc_id][key] = getattr(trial, key)
            # add experiment doc_id (NOT a unique doc key)
            results[doc_id]["doc_id"] = doc_id

        except Exception as e:
            # missing a paired_doc_id
            errs.append((doc_id, e))

    return results, errs


def main():

    args = parser.parse_args()

    # optionally filter out experiments based on key/value pairs
    filter_on = collections.defaultdict(list)
    if args.filter_on is not None:
        for key, value in args.filter_on:
            filter_on[key].append(value)
        filter_on = dict(filter_on)

    # 1. Load and integrity check for data issues including:
    # - Missing entailment files, need fact precision and fact recall
    # - Partial joins due to missing doc_ids across paired files
    experiments = load_experiments(
        args.path_to_input,
        filter_on=filter_on,
        validate=not args.disable_validation_checks,
    )

    # 2. Load experiments and score entailment pairs
    output = []
    for key, metadata in experiments.items():
        # compute entailment scores
        results, errs = score_experiment(
            metadata, args.path_to_input, args.map_unparseable_json
        )
        output.append(pd.DataFrame(data=results.values()))

    # 3. Save output results to dataframe
    output = pd.concat(output, ignore_index=True)
    output.to_csv(args.path_to_output, sep="\t", index=False)


if __name__ == "__main__":

    elapsed_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    logger.info(f"Execution time: {elapsed_time:.2f} seconds")
