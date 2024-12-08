from functools import wraps
import json
from time import time
from torch.utils.data import Dataset
from typing import TypedDict


class EntailmentPair(TypedDict):
    """Single data point."""
    doc_id: str
    sent_id: int
    premise: str
    hypothesis: str


class Message(TypedDict):
    role: str
    content: str


class EntailmentDatasetInstance(TypedDict):
    """A single instance of the EntailmentDataset."""
    entailment_pair: EntailmentPair
    message: list[Message]

class BatchedEntailmentDatasetInstance(TypedDict):
    """A batch of EntailMentDatasetInstance as returned by the PyTorch DataLoader."""
    message: list[list[Message]]
    entailment_pair: list[EntailmentPair]

class ModelOutput(TypedDict):
    verdict: int
    explanation: str


class Output(TypedDict):
    ID: str
    premise: str
    hypothesis: str
    entailment_pred: int
    explanation: str


CONTENT_PREFIX = (
    "You are an expert on natural language entailment. Your task is to deduce whether premise statements entail hypotheses."
    " Only return a '1' if the hypothesis can be fully entailed by the premise. Return '0' if the hypothesis contains information that cannot be entailed by the premise."
    " Also generate an explanation for your answer. Generate the answer in JSON format with the following keys:"
    " 'explanation': the reason why the entailment prediction is made,"
    " 'entailment_prediction': 1 or 0, whether the claim can be entailed."
    " Only return the JSON-formatted answer and nothing else\n"
)

DYNAMIC_CONTENT = "Premise: {premise}\nHypothesis: {hypothesis}"


def data_loader_collate_fn(original_batch):
    """Collate function for the dataloader used for EntailmentDataset."""
    batched_messages = []
    batched_entailment_pairs = []

    for instance in original_batch:
        batched_messages.append(instance["message"])
        batched_entailment_pairs.append(instance["entailment_pair"])

    return BatchedEntailmentDatasetInstance(
        message=batched_messages,
        entailment_pair=batched_entailment_pairs
    )


class EntailmentDataset(Dataset):
    """Dataset object for entailment data"""

    def __init__(self, file_path: str, ids_to_discard: set[str] | None = None) -> None:
        """
        :param file_path: Path to the dataset file
        """
        self._data = self.read_data(file_path=file_path, ids_to_discard=ids_to_discard)

    @staticmethod
    def read_data(file_path: str, ids_to_discard: set[str] | None = None) -> list[EntailmentPair]:
        """Yields a data point from a file"""
        if ids_to_discard is None:
            ids_to_discard = set()

        print(f"Skipping {len(ids_to_discard)} ids")

        all_data = []
        doc_ids = set()

        with open(file_path, "r") as file:
            for line in file:
                data = json.loads(line)
                
                for entailment_pair in data["text"]:
                    if data["ID"] in ids_to_discard:
                        continue
                    
                    doc_ids.add(data["ID"])
                    all_data.append(
                        EntailmentPair(
                            doc_id=data["ID"],
                            sent_id=entailment_pair["number"],
                            premise=entailment_pair["premise"],
                            hypothesis=entailment_pair["hypothesis"]
                        )
                    )

        print(f"{len(doc_ids)} doc IDs in the dataset")
        return all_data
    

    def create_message(self, entailment_pair: EntailmentPair) -> list[Message]:
        """Create a message to send to the model."""
        return [
            Message(
                role="user",
                content=CONTENT_PREFIX + DYNAMIC_CONTENT.format(premise=entailment_pair["premise"], hypothesis=entailment_pair["hypothesis"])
            )
        ]

    def __len__(self):
        return len(self._data)
    

    def __getitem__(self, index) -> EntailmentDatasetInstance:
        return EntailmentDatasetInstance(
            entailment_pair=self._data[index],
            message=self.create_message(self._data[index])
        )


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap


def extract_json_from_string(string):
    """
    Extracts the first JSON object found within a given string.

    Args:
        string (str): The string to search for JSON objects.

    Returns:
        dict: The extracted JSON object if found, otherwise None.
    """
    try:
        return json.loads(string)
    except json.JSONDecodeError:
        try:
            start_idx = string.index('{')
            end_idx = string.rindex('}') + 1
            return json.loads(string[start_idx:end_idx])
        except (ValueError, json.JSONDecodeError):
            return None
