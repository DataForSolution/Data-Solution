from sentiment_portfolio.vader import vader_predict


class FakeAnalyzer:
    def polarity_scores(self, text):
        return {"compound": 0.5 if "good" in text else -0.2 if "bad" in text else 0.0}


def test_vader_adapter_threshold_semantics():
    assert vader_predict(["good", "bad", "neutral"], FakeAnalyzer()) == [1, 0, 0]
