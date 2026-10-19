"""Numerical contracts for pixelwise regression probes."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.config.schema import SegmentationConfig
from torchgeo_bench.pixelwise_regression_task import (
    PixelwiseRegressionSolver,
    build_pixelwise_regression_probe_and_solver,
)
from torchgeo_bench.results import bootstrap_mean


class _FixedProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Identity()
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images[:, :1] + self.bias


class _SpatialBackbone(nn.Module):
    num_channels = 1

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(1, 2, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layer(images)


def test_evaluation_averages_per_image_rmse_in_reported_units() -> None:
    images = torch.zeros(2, 1, 4, 4)
    targets = torch.stack([torch.ones(4, 4), torch.full((4, 4), 2.0)])
    solver = PixelwiseRegressionSolver(
        _FixedProbe(), target_scale=10.0, device="cpu", lr_scheduler="none"
    )

    metrics, per_image = solver.evaluate(DataLoader(TensorDataset(images, targets), batch_size=2))

    torch.testing.assert_close(per_image, torch.tensor([10.0, 20.0]))
    assert metrics["RMSE"] == pytest.approx(15.0)


def test_cached_targets_remain_float_for_regression() -> None:
    config = SegmentationConfig(layers=["layer"], head="linear", scheduler="none")
    probe, solver = build_pixelwise_regression_probe_and_solver(
        _SpatialBackbone(), config, torch.device("cpu"), target_scale=10.0
    )
    for parameter in probe.head.parameters():
        nn.init.zeros_(parameter)
    images = torch.zeros(2, 1, 8, 8)
    targets = torch.stack([torch.ones(8, 8), torch.full((8, 8), 2.0)])
    cache = probe.extract_segmentation_features(
        DataLoader(TensorDataset(images, targets), batch_size=2),
        cache_dtype=torch.float32,
        target_dtype=torch.float32,
    )

    assert cache.masks.dtype == torch.float32
    metrics, per_image = solver.evaluate_cached(cache, batch_size=2)
    torch.testing.assert_close(per_image, torch.tensor([10.0, 20.0]))
    assert metrics["RMSE"] == pytest.approx(15.0)


def test_bootstrap_mean_resamples_held_out_images() -> None:
    lower, upper = bootstrap_mean(torch.tensor([10.0, 20.0]), n_boot=200, seed=7)
    assert (lower, upper) == pytest.approx((10.0, 20.0))


def test_bootstrap_mean_can_be_disabled() -> None:
    lower, upper = bootstrap_mean(torch.tensor([10.0]), n_boot=0)
    assert torch.isnan(torch.tensor([lower, upper])).all()
