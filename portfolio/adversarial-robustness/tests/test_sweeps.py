import numpy as np

from adversarial_portfolio.sweeps import run_defence_sweep


def test_sweep_uses_same_predictor_for_clean_and_adversarial_paths():
    clean = np.array([[2.0, 0.0], [0.0, 2.0]])
    adv = np.array([[0.0, 2.0], [0.0, 2.0]])
    calls = []

    def predictor(x):
        calls.append(np.array(x, copy=True))
        z = np.asarray(x, dtype=float)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def factory(scale):
        return lambda x: np.asarray(x) * scale

    result = run_defence_sweep(
        parameters=[1.0, 0.5],
        clean_inputs=clean,
        adversarial_inputs=adv,
        true_labels=[0, 1],
        predictor=predictor,
        transform_factory=factory,
        parameter_name="scale",
    )

    assert result["scale"].tolist() == [1.0, 0.5]
    assert len(calls) == 6  # 2 baselines + 2 predictor calls for each parameter
    assert "clean_accuracy_delta" in result.columns
    assert "robust_accuracy_gain" in result.columns
