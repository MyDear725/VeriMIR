import torch
import torch.nn.functional as F

from verimir.losses import (
    verified_augmentation_allocation,
    verified_augmentation_gate,
    verified_two_view_objective,
    weighted_mean,
)


def test_verified_gate_uses_conservative_branch_minimum_and_caption_weight_once():
    clean_appearance = F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=-1)
    augmented_appearance = clean_appearance.clone()
    clean_semantic = clean_appearance.clone()
    augmented_semantic = F.normalize(
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=-1,
    )
    caption_weights = torch.tensor([0.4, 0.7], requires_grad=True)

    gate, details = verified_augmentation_gate(
        clean_appearance,
        clean_semantic,
        augmented_appearance,
        augmented_semantic,
        caption_weights,
    )

    assert torch.allclose(details["appearance_agreement"], torch.ones(2))
    assert torch.allclose(details["semantic_agreement"], torch.tensor([0.5, 1.0]))
    assert torch.allclose(gate, torch.tensor([0.2, 0.7]))
    assert not gate.requires_grad
    assert torch.all((0.0 <= gate) & (gate <= 1.0))


def test_verified_gate_rejects_view_when_either_branch_fully_disagrees():
    clean = F.normalize(torch.tensor([[1.0, 0.0]]), dim=-1)
    opposite = -clean
    gate, _ = verified_augmentation_gate(
        clean,
        clean,
        clean,
        opposite,
        torch.ones(1),
    )
    assert torch.equal(gate, torch.zeros(1))


def test_verified_two_view_objective_applies_absolute_budget():
    clean_loss = torch.tensor(2.0, requires_grad=True)
    augmented_loss = torch.tensor(4.0, requires_grad=True)
    gate = torch.full((3,), 0.25)
    objective = verified_two_view_objective(clean_loss, augmented_loss, gate)
    assert torch.allclose(objective, torch.tensor(1.5))
    objective.backward()
    assert torch.allclose(clean_loss.grad, torch.tensor(0.5))
    assert torch.allclose(augmented_loss.grad, torch.tensor(0.125))


def test_verified_two_view_objective_exactly_recovers_equal_view_mean_at_true_all_one_gate():
    clean_loss = torch.tensor(2.0)
    augmented_loss = torch.tensor(4.0)
    objective = verified_two_view_objective(
        clean_loss,
        augmented_loss,
        torch.ones(5),
    )
    assert torch.equal(objective, torch.stack([clean_loss, augmented_loss]).mean())


def test_verified_two_view_relative_mode_keeps_scalar_outer_coefficients_fixed():
    clean_loss = torch.tensor(2.0, requires_grad=True)
    augmented_loss = torch.tensor(4.0, requires_grad=True)
    objective = verified_two_view_objective(
        clean_loss,
        augmented_loss,
        torch.full((3,), 0.25),
        mode="relative",
    )
    assert torch.allclose(objective, torch.tensor(3.0))
    objective.backward()
    assert torch.allclose(clean_loss.grad, torch.tensor(0.5))
    assert torch.allclose(augmented_loss.grad, torch.tensor(0.5))


def test_verified_two_view_residual_mode_keeps_scalar_outer_coefficients_fixed():
    clean_loss = torch.tensor(2.0, requires_grad=True)
    augmented_loss = torch.tensor(4.0, requires_grad=True)
    objective = verified_two_view_objective(
        clean_loss,
        augmented_loss,
        torch.tensor([0.1, 0.7]),
        mode="residual",
    )
    assert torch.allclose(objective, torch.tensor(3.0))
    objective.backward()
    assert torch.allclose(clean_loss.grad, torch.tensor(0.5))
    assert torch.allclose(augmented_loss.grad, torch.tensor(0.5))


def test_verified_residual_allocation_has_unit_mean_and_preserves_order():
    gate = torch.tensor([0.1, 0.3, 0.8], requires_grad=True)
    allocation = verified_augmentation_allocation(gate, mode="residual")
    assert torch.allclose(allocation.mean(), torch.tensor(1.0))
    assert torch.all(allocation > 0.0)
    assert torch.equal(torch.argsort(allocation), torch.argsort(gate))
    assert not allocation.requires_grad


def test_verified_residual_allocation_exactly_recovers_all_one_fallback():
    allocation = verified_augmentation_allocation(
        torch.ones(5),
        mode="residual",
    )
    assert torch.equal(allocation, torch.ones(5))


def test_verified_residual_allocation_compresses_gate_ratio():
    gate = torch.tensor([0.2, 0.8])
    allocation = verified_augmentation_allocation(gate, mode="residual")
    assert allocation.max() / allocation.min() < gate.max() / gate.min()


def test_verified_residual_allocation_temperately_reweights_augmented_loss():
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    allocation = verified_augmentation_allocation(
        torch.tensor([0.25, 0.75]),
        mode="residual",
    )
    loss = weighted_mean(values, allocation)
    loss.backward()
    assert torch.allclose(loss, torch.tensor(2.25))
    assert torch.allclose(values.grad, torch.tensor([0.375, 0.625]))


def test_verified_relative_weights_reallocate_heterogeneous_samples():
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    gate = torch.tensor([0.25, 0.75])
    loss = weighted_mean(values, gate)
    loss.backward()
    assert torch.allclose(loss, torch.tensor(2.5))
    assert torch.allclose(values.grad, gate)


def test_verified_two_view_objective_rejects_unknown_mode():
    try:
        verified_two_view_objective(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.ones(2),
            mode="unknown",
        )
    except ValueError as error:
        assert "absolute, relative, or residual" in str(error)
    else:
        raise AssertionError("unknown verified augmentation mode was accepted")


def test_verified_gate_rejects_nonfinite_input():
    clean = torch.tensor([[1.0, 0.0]])
    bad = torch.tensor([[float("nan"), 0.0]])
    try:
        verified_augmentation_gate(clean, clean, bad, clean, torch.ones(1))
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("nonfinite verified augmentation input was accepted")
