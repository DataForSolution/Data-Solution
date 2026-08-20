"""Adapter for an injected VADER-compatible analyzer."""


def vader_predict(texts, analyzer, *, positive_threshold: float = 0.0):
    """Map analyzer compound scores to binary labels.

    The analyzer is injected so core tests do not need an NLTK data download.
    """
    predictions = []
    for text in texts:
        result = analyzer.polarity_scores(str(text))
        if "compound" not in result:
            raise ValueError("analyzer result is missing compound score")
        predictions.append(int(float(result["compound"]) > positive_threshold))
    return predictions
