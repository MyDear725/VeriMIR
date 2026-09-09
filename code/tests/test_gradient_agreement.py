import torch

from verimir.gradient_agreement import (
    retrieval_interface_conflict_objective,
    retrieval_interface_parameters,
)


def test_ricf_positive_agreement_exactly_recovers_equal_view_mean():
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    clean = parameter.square().sum()
    augmented = (2.0 * parameter).square().sum()
    objective, details = retrieval_interface_conflict_objective(
        clean, augmented, [parameter], policy="adaptive",
    )
    assert torch.equal(details["alpha"], torch.tensor(1.0))
    assert torch.allclose(objective, 0.5 * (clean + augmented))


def test_ricf_negative_conflict_only_attenuates_augmented_coefficient():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    clean = parameter.sum()
    augmented = -parameter.sum()
    objective, details = retrieval_interface_conflict_objective(
        clean, augmented, [parameter], policy="adaptive",
    )
    assert torch.allclose(details["cosine"], torch.tensor(-1.0))
    assert torch.allclose(details["alpha"], torch.tensor(0.0))
    objective.backward()
    assert torch.allclose(parameter.grad, torch.tensor([0.5]))


def test_ricf_observe_only_keeps_v21_objective_under_negative_conflict():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    clean = parameter.sum()
    augmented = -parameter.sum()
    objective, details = retrieval_interface_conflict_objective(
        clean, augmented, [parameter], policy="observe_only",
    )
    assert torch.allclose(details["cosine"], torch.tensor(-1.0))
    assert torch.equal(details["alpha"], torch.tensor(1.0))
    assert torch.allclose(objective, 0.5 * (clean + augmented))


def test_ricf_orthogonal_gradients_are_not_attenuated():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    clean = parameter[0]
    augmented = parameter[1]
    _, details = retrieval_interface_conflict_objective(
        clean, augmented, [parameter], policy="adaptive",
    )
    assert torch.allclose(details["cosine"], torch.tensor(0.0))
    assert torch.equal(details["alpha"], torch.tensor(1.0))


def test_ricf_zero_norm_falls_back_to_no_attenuation():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    clean = parameter.sum() * 0.0
    augmented = parameter.sum()
    _, details = retrieval_interface_conflict_objective(
        clean, augmented, [parameter], policy="adaptive",
    )
    assert bool(details["zero_norm"])
    assert torch.equal(details["alpha"], torch.tensor(1.0))


def test_ricf_rejects_unknown_policy_and_nonpositive_epsilon():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    clean = parameter.sum()
    augmented = parameter.square().sum()
    try:
        retrieval_interface_conflict_objective(clean, augmented, [parameter], policy="unknown")
    except ValueError as error:
        assert "adaptive or observe_only" in str(error)
    else:
        raise AssertionError("unknown RICF policy was accepted")
    try:
        retrieval_interface_conflict_objective(clean, augmented, [parameter], eps=0.0)
    except ValueError as error:
        assert "eps must be positive" in str(error)
    else:
        raise AssertionError("non-positive RICF epsilon was accepted")


class _DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_projection = torch.nn.Linear(3, 2)
        self.body = torch.nn.Linear(3, 3)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _DummyBackbone()
        self.appearance_head = torch.nn.Linear(2, 2)
        self.semantic_head = torch.nn.Linear(2, 2)
        self.appearance_classifier = torch.nn.Linear(2, 4)


def test_retrieval_interface_parameter_selection_excludes_body_and_classifier():
    model = _DummyModel()
    selected = {id(parameter) for parameter in retrieval_interface_parameters(model)}
    expected = {
        id(parameter)
        for module in (
            model.backbone.visual_projection,
            model.appearance_head,
            model.semantic_head,
        )
        for parameter in module.parameters()
    }
    excluded = {
        id(parameter)
        for module in (model.backbone.body, model.appearance_classifier)
        for parameter in module.parameters()
    }
    assert selected == expected
    assert selected.isdisjoint(excluded)

