import pandas as pd
from fraud_portfolio.modeling import build_logistic_pipeline, candidate_configurations


def test_pipeline_handles_unseen_categories():
    train = pd.DataFrame({"Age":[20,30,40,50], "Make":["A","B","A","B"]})
    y = [0,1,0,1]
    model = build_logistic_pipeline(train)
    model.fit(train, y)
    proba = model.predict_proba(pd.DataFrame({"Age":[35], "Make":["NEW"]}))
    assert proba.shape == (1, 2)


def test_candidate_space_is_small_and_valid():
    configs = candidate_configurations()
    assert len(configs) == 8
    assert all(c["C"] > 0 for c in configs)
