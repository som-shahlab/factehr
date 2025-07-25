import os
import argparse
import logging
from datasets import load_dataset

########## Logger Setup ##########

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

########## Argparse Setup ##########

parser = argparse.ArgumentParser(description="Load and save dataset to disk")
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of the dataset to load (SciTail, MedNLI, etc.)",
    required=True,
)

parser.add_argument(
    "--path_to_save_dir",
    type=str,
    help="Path to the directory where the dataset will be saved",
    required=True,
)

########## Function Definitions ##########

def load_jsonl(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]

def create_mednli_dataset_dict(data_dir):
    train_path = os.path.join(data_dir, "mli_train_v1.jsonl")
    dev_path   = os.path.join(data_dir, "mli_dev_v1.jsonl")
    test_path  = os.path.join(data_dir, "mli_test_v1.jsonl")

    def simplify(e):
        return {
            "pairID": e["pairID"],
            "sentence1": e["sentence1"],
            "sentence2": e["sentence2"],
            "label": e["gold_label"]
        }

    dataset = DatasetDict({
        "train": Dataset.from_list([simplify(e) for e in load_jsonl(train_path)]),
        "validation": Dataset.from_list([simplify(e) for e in load_jsonl(dev_path)]),
        "test": Dataset.from_list([simplify(e) for e in load_jsonl(test_path)]),
    })

    return dataset

def load_and_save_dataset(dataset_name: str, save_dir: str):
    """Load a dataset and save it to disk."""
    
    # Loading specific datasets based on user input
    if dataset_name == "scitail":
        dataset = load_dataset("allenai/scitail", "snli_format")
    elif dataset_name == "mednli":
        base_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(os.path.dirname(base_dir))
        data_dir = os.path.join(parent_dir, "data", "datasets", "raw", "physionet.org", "content", "mednli", "get-zip", "1.0.0")
        dataset = load_dataset(
            path="bigbio/mednli",
            data_dir=data_dir,
            # local_files_only=True,
            trust_remote_code=True
            # token=True
        )
    elif dataset_name == "multinli":
        dataset = load_dataset("nyu-mll/multi_nli")
    elif dataset_name == "snli":
        dataset = load_dataset("stanfordnlp/snli")
    else:
        logger.error(f"Dataset {dataset_name} is not supported")
        return

    # Create directory to save the dataset if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the dataset to the specified directory
    dataset.save_to_disk(save_dir)
    logger.info(f"Saved dataset {dataset_name} to {save_dir}")

########## Main Function ##########

def main():
    args = parser.parse_args()
    load_and_save_dataset(args.dataset_name, args.path_to_save_dir)

if __name__ == "__main__":
    main()
