import pytest
from sentiment_portfolio.modeling import build_model_pipeline, model_names


def test_all_declared_models_have_vectorizer_inside_pipeline():
    for name in model_names():
        pipe = build_model_pipeline(name)
        assert list(pipe.named_steps) == ["tfidf", "classifier"]


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model_pipeline("random_model_zoo")
