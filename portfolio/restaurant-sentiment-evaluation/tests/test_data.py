from pathlib import Path
import pytest
from sentiment_portfolio.data import load_labelled_sentences, validate_text_labels


def test_parser_splits_on_final_tab(tmp_path: Path):
    p = tmp_path / "reviews.txt"
    p.write_text("great food\t1\nbad service\t0\n", encoding="utf-8")
    text, y = load_labelled_sentences(p)
    assert text.tolist() == ["great food", "bad service"]
    assert y.tolist() == [1, 0]


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        validate_text_labels(["good"], [2])
