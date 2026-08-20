from petquant_portfolio.safety import apply_reliable_score_gate, false_reliable_rate


def test_false_reliable_rate():
    result = false_reliable_rate(
        ["unreliable", "unreliable", "caution", "reliable"],
        ["reliable", "caution", "reliable", "reliable"],
    )
    assert result["unreliable_total"] == 2
    assert result["false_reliable_count"] == 1
    assert result["false_reliable_rate"] == 0.5


def test_no_unreliable_reference_fails_closed():
    result = false_reliable_rate(["caution"], ["reliable"])
    assert result["unreliable_total"] == 0
    assert result["false_reliable_rate"] is None


def test_reliable_score_gate():
    assert apply_reliable_score_gate(
        ["reliable", "reliable", "unreliable"],
        [89.9, 90.0, 40.0],
    ) == ["caution", "reliable", "unreliable"]
