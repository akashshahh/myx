"""Tests for the PANNs-based perception model.

Most tests use a randomly-initialized model (fast, no checkpoint needed). The
pretrained-load test is skipped when the ~312 MB checkpoint isn't present.
"""
from __future__ import annotations

import os

import pytest
import torch

from data.degradation import ISSUE_DIMENSIONS
from perception.model import (
    NUM_ISSUES,
    SAMPLE_RATE,
    PerceptionModel,
    build_perception_model,
)

CHECKPOINT = "checkpoints/Cnn14_mAP=0.431.pth"


@pytest.fixture(scope="module")
def model() -> PerceptionModel:
    m = build_perception_model(checkpoint_path=None, freeze_backbone=True)
    m.eval()
    return m


def test_num_issues_matches_dimensions():
    assert NUM_ISSUES == len(ISSUE_DIMENSIONS) == 8


def test_forward_shape_and_range(model):
    # 6 seconds of mono audio @ 32 kHz
    x = torch.randn(2, 6 * SAMPLE_RATE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, NUM_ISSUES)
    assert torch.all(y >= 0) and torch.all(y <= 1)


def test_variable_length_input(model):
    # A 3-second clip should also work (PANNs pools over time).
    x = torch.randn(1, 3 * SAMPLE_RATE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, NUM_ISSUES)


def test_freeze_leaves_only_head_trainable():
    m = build_perception_model(checkpoint_path=None, freeze_backbone=True)
    trainable = {n for n, p in m.backbone.named_parameters() if p.requires_grad}
    assert trainable, "expected some trainable params"
    assert all(n.startswith(("fc1", "fc_audioset")) for n in trainable)
    # conv blocks must be frozen
    assert not any(n.startswith("conv_block") for n in trainable)


def test_unfreeze_makes_all_trainable():
    m = build_perception_model(checkpoint_path=None, freeze_backbone=True)
    m.unfreeze_backbone()
    assert all(p.requires_grad for p in m.backbone.parameters())


def test_head_width_is_eight(model):
    assert model.backbone.fc_audioset.out_features == NUM_ISSUES


@pytest.mark.skipif(not os.path.exists(CHECKPOINT), reason="pretrained checkpoint not downloaded")
def test_pretrained_loads_and_runs():
    m = build_perception_model(checkpoint_path=CHECKPOINT, freeze_backbone=True)
    m.eval()
    x = torch.randn(2, 6 * SAMPLE_RATE)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, NUM_ISSUES)
    assert torch.all(y >= 0) and torch.all(y <= 1)
