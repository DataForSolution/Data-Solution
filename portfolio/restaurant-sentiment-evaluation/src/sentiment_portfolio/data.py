"""Parsing and validation for UCI-style labelled review sentences."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def validate_text_labels(texts, labels) -> tuple[pd.Series, pd.Series]:
    text = pd.Series(texts, dtype="string").reset_index(drop=True)
    y = pd.to_numeric(pd.Series(labels), errors="raise").astype(int).reset_index(drop=True)
    if len(text) == 0 or len(text) != len(y):
        raise ValueError("texts and labels must be aligned and non-empty")
    if text.isna().any() or text.str.strip().eq("").any():
        raise ValueError("review text must be non-empty")
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("labels must be binary 0/1")
    return text, y


def load_labelled_sentences(path) -> tuple[pd.Series, pd.Series]:
    """Parse `sentence<TAB>label`, splitting on the final tab only."""
    rows = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            sentence, label = raw.rsplit("\t", 1)
        except ValueError as exc:
            raise ValueError(f"malformed line {lineno}") from exc
        rows.append((sentence, label))
    if not rows:
        raise ValueError("no labelled sentences found")
    return validate_text_labels([r[0] for r in rows], [r[1] for r in rows])
