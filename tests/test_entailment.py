import numpy as np
import pytest
from factehr.evaluation.entailment import entailment_proportion


def test_no_sample_weights_all_entailed():
    preds = [1, 1, 1, 1]
    result = entailment_proportion(preds)
    assert pytest.approx(result) == 1.0


def test_no_sample_weights_none_entailed():
    preds = [0, 0, 0, 0]
    result = entailment_proportion(preds)
    assert pytest.approx(result) == 0.0


def test_no_sample_weights_mixed():
    preds = [1, 0, 1, 0]
    result = entailment_proportion(preds)
    assert pytest.approx(result) == 0.5


def test_with_sample_weights_all_entailed():
    preds = [1, 1, 1]
    sample_weights = [2, 2, 2]
    result = entailment_proportion(preds, sample_weights)
    assert pytest.approx(result) == 1.0


def test_with_sample_weights_none_entailed():
    preds = [0, 0, 0]
    sample_weights = [3, 3, 3]
    result = entailment_proportion(preds, sample_weights)
    assert pytest.approx(result) == 0.0


def test_with_sample_weights_mixed():
    preds = [1, 0, 1, 0]
    sample_weights = [1, 2, 1, 2]
    result = entailment_proportion(preds, sample_weights)
    assert pytest.approx(result, rel=1e-5) == 0.33333


def test_with_sample_weights_varying_values():
    preds = [1, 1, 0, 1]
    sample_weights = [3, 1, 4, 2]
    result = entailment_proportion(preds, sample_weights)
    assert pytest.approx(result) == 0.6


def test_empty_preds():
    preds = []
    result = entailment_proportion(preds)
    assert result == 0.0


def test_empty_preds_with_sample_weights():
    preds = []
    sample_weights = []
    result = entailment_proportion(preds, sample_weights)
    assert result == 0.0


def test_sample_weights_contain_zero():
    preds = [1, 0, 1]
    sample_weights = [1, 0, 2]  # Contains a zero
    with pytest.raises(ValueError):
        entailment_proportion(preds, sample_weights)
