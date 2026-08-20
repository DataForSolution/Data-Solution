import torch

from adversarial_portfolio.attacks import pgd_linf


def test_pgd_respects_linf_budget_and_input_bounds():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    with torch.no_grad():
        model[1].weight.copy_(torch.tensor([[1., 1., -1., -1.], [-1., -1., 1., 1.]]))
        model[1].bias.zero_()

    x = torch.tensor([[[[0.8, 0.8], [0.2, 0.2]]]], dtype=torch.float32)
    y = torch.tensor([0])
    adv = pgd_linf(model, x, y, epsilon=0.1, step_size=0.04, steps=5)

    assert torch.max(torch.abs(adv - x)).item() <= 0.100001
    assert torch.all(adv >= 0.0)
    assert torch.all(adv <= 1.0)


def test_pgd_rejects_invalid_config():
    model = torch.nn.Linear(2, 2)
    x = torch.zeros((1, 2))
    y = torch.tensor([0])
    try:
        pgd_linf(model, x, y, epsilon=-1, step_size=0.1, steps=1)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
