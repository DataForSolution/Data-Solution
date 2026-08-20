"""Sparse text-model pipelines with fold-local vocabulary learning."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def model_names():
    return ("multinomial_nb", "logistic_regression", "linear_svm")


def build_model_pipeline(name: str) -> Pipeline:
    if name not in model_names():
        raise ValueError(f"unknown model: {name!r}")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    if name == "multinomial_nb":
        classifier = MultinomialNB(alpha=1.0)
    elif name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    else:
        classifier = LinearSVC(C=1.0, random_state=42)
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
