import numpy as np
from typing import List


def entailment_proportion(preds: List[int], sample_weights: List[int] = None) -> float:
    """Entailment proportion evaluation. Use this to compute fact precision and
    fact recall as defined in (Xie et al 2024). All sample_weights must be >= 1

    Parameters
    ----------
    preds : List[int]
        list of indication function values 1:= entailed 0:= not entailed
    sample_weights : List[int], optional
        per-instance sample weight (useful for upweighting by freq of exact
        entailment pair duplicates pairs), by default None

    Returns
    -------
    float
        proportion of total entailed pairs
    """
    # if no entailemnt predictions are provided, score as 0
    if len(preds) == 0:
        return 0.0
    # type casting for numpy math conveniance
    preds = np.array(preds).astype(int)
    if sample_weights is not None:
        sample_weights = np.array(sample_weights).astype(int)

        if np.any(sample_weights == 0):
            raise ValueError("sample_weights cannot contain zero")
    # total number of entailment pairs to evaluate
    N = len(preds) if sample_weights is None else np.sum(sample_weights)
    n = np.sum(preds) if sample_weights is None else np.sum(sample_weights * preds)
    return n / N
