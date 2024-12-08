""" 

"""

import re
import json
import hashlib
import pandas as pd
from typing import Generator, Dict, List, Any


def hash_text(text: str) -> str:
    """Get MD5 hex hash string for doc"""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


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


def split_facts(text: str, exclude: List[str] = None) -> List[str]:
    """Modifed from Monica 8-14-2024"""
    # skip some pathological fact generations
    exclude = {":", ""} if exclude is None else exclude
    # Remove any leading text like "##  Independent Facts:"
    text = re.sub(r"^\s*##\s*.+Facts+:\s*", "", text, flags=re.I)

    # Split into individual facts based on numbered lists and bullet points
    facts = re.split(r"\n\s*(?:\d+\.|\*)\s*", text)
    facts = [fact.strip() for fact in facts if fact.strip() not in exclude]

    return facts
