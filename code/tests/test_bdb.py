import pytest
import torch
import torch.nn.functional as F
from verimir.schedule import _sv_bt5sd_effective_weight
from verimir.losses import build_bidirectional_top5_teacher, self_verified_bidirectional_top5_support_loss
from verimir.retrieval import fused_score


def _bank(*, requires_grad=False):
    generator = torch.Generator().manual_seed(140)
    appearance = F.normalize(torch.randn(12, 6, generator=generator), dim=-1)
    semantic = F.normalize(torch.randn(12, 5, generator=generator), dim=-1)
    appearance.requires_grad_(requires_grad)
    semantic.requires_grad_(requires_grad)
    return {
        "appearance": appearance,
        "semantic": semantic,
        "labels": torch.tensor([0] * 4 + [1] * 4 + [2] * 4),
        "sample_ids": torch.arange(100, 112),
        "calibration": {
            "appearance_mean": 0.07,
            "appearance_std": 0.61,
            "semantic_mean": -0.03,
            "semantic_std": 0.73,
        },
        "snapshot_version": 0,
        "calibration_version": 0,
    }


def _teacher(bank):
    return build_bidirectional_top5_teacher(
        bank,
        beta=0.2,
        appearance_weight=0.6,
        top_k=5,
        chunk_size=4,
    )


def test_teacher_caches_exact_top5_and_rank6_to10_without_self():
    bank = _bank()
    teacher = _teacher(bank)
    appearance = F.normalize(bank["appearance"], dim=-1)
    semantic = F.normalize(bank["semantic"], dim=-1)
    score_a = (appearance @ appearance.T - 0.07) / 0.61
    score_s = (semantic @ semantic.T + 0.03) / 0.73
    score = fused_score(score_a, score_s, 0.2, 0.6)
    score.fill_diagonal_(-float("inf"))
    exact = torch.topk(score, 10, dim=1, sorted=True)
    assert torch.equal(teacher["top_indices"], exact.indices[:, :5])
    assert torch.equal(teacher["boundary_indices"], exact.indices[:, 5:10])
    assert torch.allclose(teacher["fused_cutoff"], exact.values[:, 4])
    assert not torch.any(
        teacher["top_indices"][:, :, None].eq(
            teacher["boundary_indices"][:, None, :]
        )
    )
    rows = torch.arange(12)[:, None]
    assert not teacher["top_indices"].eq(rows).any()
    assert not teacher["boundary_indices"].eq(rows).any()


def test_full_loss_is_finite_fp32_and_backpropagates_both_image_branches():
    bank = _bank(requires_grad=True)
    teacher = _teacher(bank)
    indices = torch.tensor([0, 4, 8])
    clean_a = bank["appearance"].detach()[indices].clone().requires_grad_(True)
    clean_s = bank["semantic"].detach()[indices].clone().requires_grad_(True)
    aug_a = (-clean_a.detach()).clone().requires_grad_(True)
    aug_s = (-clean_s.detach()).clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, details = self_verified_bidirectional_top5_support_loss(
            clean_a,
            clean_s,
            aug_a,
            aug_s,
            bank["labels"][indices],
            bank["sample_ids"][indices],
            indices,
            bank,
            teacher,
            beta=0.2,
            appearance_weight=0.6,
            margin=10.0,
            include_incoming=True,
        )
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss) and loss > 0.0
    assert details["include_incoming"].item() == 1.0
    loss.backward()
    appearance_grad = sum(
        value.grad.abs().sum() for value in (clean_a, aug_a)
        if value.grad is not None
    )
    semantic_grad = sum(
        value.grad.abs().sum() for value in (clean_s, aug_s)
        if value.grad is not None
    )
    assert appearance_grad > 0.0
    assert semantic_grad > 0.0
    assert bank["appearance"].grad is None
    assert bank["semantic"].grad is None


def test_v141_incoming_weight_matches_auditable_weighted_mean():
    bank = _bank()
    teacher = _teacher(bank)
    indices = torch.tensor([0, 4, 8])
    clean_a = bank["appearance"][indices].clone()
    clean_s = bank["semantic"][indices].clone()
    aug_a = -clean_a
    aug_s = -clean_s
    common = dict(
        clean_appearance=clean_a,
        clean_semantic=clean_s,
        augmented_appearance=aug_a,
        augmented_semantic=aug_s,
        labels=bank["labels"][indices],
        sample_ids=bank["sample_ids"][indices],
        bank_indices=indices,
        bank=bank,
        teacher=teacher,
        beta=0.2,
        appearance_weight=0.6,
        margin=10.0,
    )
    outgoing, _ = self_verified_bidirectional_top5_support_loss(
        **common, include_incoming=False,
    )
    calibrated, details = self_verified_bidirectional_top5_support_loss(
        **common, include_incoming=True, incoming_weight=0.05,
    )
    expected = (
        outgoing + 0.05 * details["incoming_loss"]
    ) / 1.05
    assert calibrated.item() == pytest.approx(expected.item(), rel=1e-6)
    assert details["incoming_weight"].item() == pytest.approx(0.05)


def test_v141_rejects_invalid_incoming_weight():
    bank = _bank()
    teacher = _teacher(bank)
    indices = torch.tensor([0, 4, 8])
    with pytest.raises(ValueError, match="incoming_weight"):
        self_verified_bidirectional_top5_support_loss(
            bank["appearance"][indices],
            bank["semantic"][indices],
            bank["appearance"][indices],
            bank["semantic"][indices],
            bank["labels"][indices],
            bank["sample_ids"][indices],
            indices,
            bank,
            teacher,
            beta=0.2,
            appearance_weight=0.6,
            incoming_weight=-0.1,
        )


def test_outgoing_only_has_no_incoming_loss_and_teacher_is_version_locked():
    bank = _bank()
    teacher = _teacher(bank)
    indices = torch.tensor([0, 4, 8])
    clean_a = bank["appearance"][indices].clone().requires_grad_(True)
    clean_s = bank["semantic"][indices].clone().requires_grad_(True)
    loss, details = self_verified_bidirectional_top5_support_loss(
        clean_a,
        clean_s,
        clean_a.detach().clone().requires_grad_(True),
        clean_s.detach().clone().requires_grad_(True),
        bank["labels"][indices],
        bank["sample_ids"][indices],
        indices,
        bank,
        teacher,
        beta=0.2,
        appearance_weight=0.6,
        include_incoming=False,
    )
    assert torch.isfinite(loss)
    assert details["incoming_loss"].item() == 0.0
    assert details["include_incoming"].item() == 0.0
    stale = dict(teacher)
    stale["snapshot_version"] = 1
    with pytest.raises(ValueError, match="versions differ"):
        self_verified_bidirectional_top5_support_loss(
            clean_a,
            clean_s,
            clean_a,
            clean_s,
            bank["labels"][indices],
            bank["sample_ids"][indices],
            indices,
            bank,
            stale,
            beta=0.2,
            appearance_weight=0.6,
        )


def test_incoming_zero_edge_case_is_differentiable_exact_zero():
    bank = _bank()
    teacher = _teacher(bank)
    # Candidate bank index 0 is deliberately absent from every frozen sparse
    # support/boundary set, exercising the explicitly frozen no-fallback case.
    teacher["top_indices"] = torch.ones_like(teacher["top_indices"])
    teacher["boundary_indices"] = torch.full_like(
        teacher["boundary_indices"], 2,
    )
    index = torch.tensor([0])
    clean_a = bank["appearance"][index].clone().requires_grad_(True)
    clean_s = bank["semantic"][index].clone().requires_grad_(True)
    loss, details = self_verified_bidirectional_top5_support_loss(
        clean_a,
        clean_s,
        clean_a.detach().clone().requires_grad_(True),
        clean_s.detach().clone().requires_grad_(True),
        bank["labels"][index],
        bank["sample_ids"][index],
        index,
        bank,
        teacher,
        beta=0.2,
        appearance_weight=0.6,
        include_incoming=True,
    )
    assert details["incoming_loss"].item() == pytest.approx(0.0, abs=1e-12)
    assert torch.isfinite(loss)
    loss.backward()
    assert clean_a.grad is not None
    assert clean_s.grad is not None


def test_teacher_rejects_duplicate_ids_and_too_small_bank():
    duplicate = _bank()
    duplicate["sample_ids"][1] = duplicate["sample_ids"][0]
    with pytest.raises(ValueError, match="unique"):
        _teacher(duplicate)
    small = _bank()
    for key in ("appearance", "semantic", "labels", "sample_ids"):
        small[key] = small[key][:10]
    with pytest.raises(ValueError, match=r"more than 2\*top_k"):
        _teacher(small)


def test_v140_pulse_schedule_is_frozen():
    assert [_sv_bt5sd_effective_weight(0.025, epoch) for epoch in range(1, 5)] == [
        0.025,
        0.0125,
        0.0,
        0.0,
    ]
    with pytest.raises(ValueError, match="epoch"):
        _sv_bt5sd_effective_weight(0.025, 0)
