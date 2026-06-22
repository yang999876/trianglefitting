from __future__ import annotations

import time
import warnings
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from .config import LossWeights
from .model import EllipseParameters, ROLE_ACCENT, ROLE_BASE, ROLE_TEXTURE
from .utils import normalized_triangle_area, rgb_rmse

try:
    import lpips  # type: ignore
except ImportError:  # pragma: no cover
    lpips = None


def _sync_if_needed(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _gaussian_kernel(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size // 2)
    kernel = torch.exp(-(coords**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def _avg_blur(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return image
    padding = kernel_size // 2
    padded = F.pad(image, (padding, padding, padding, padding), mode="reflect")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


class MultiScaleSSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5) -> None:
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer("weights", torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32))

    def _ssim(self, first: torch.Tensor, second: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        channel = first.shape[1]
        kernel_2d = _gaussian_kernel(self.window_size, self.sigma, first.device, first.dtype)
        window = kernel_2d.view(1, 1, self.window_size, self.window_size).expand(channel, 1, self.window_size, self.window_size)
        mu_first = F.conv2d(first, window, padding=self.window_size // 2, groups=channel)
        mu_second = F.conv2d(second, window, padding=self.window_size // 2, groups=channel)
        mu_first_sq = mu_first.pow(2)
        mu_second_sq = mu_second.pow(2)
        mu_first_second = mu_first * mu_second
        sigma_first_sq = F.conv2d(first * first, window, padding=self.window_size // 2, groups=channel) - mu_first_sq
        sigma_second_sq = F.conv2d(second * second, window, padding=self.window_size // 2, groups=channel) - mu_second_sq
        sigma_first_second = F.conv2d(first * second, window, padding=self.window_size // 2, groups=channel) - mu_first_second
        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2.0 * mu_first_second + c1) * (2.0 * sigma_first_second + c2)
        denominator = (mu_first_sq + mu_second_sq + c1) * (sigma_first_sq + sigma_second_sq + c2)
        ssim_map = numerator / denominator.clamp_min(1e-6)
        cs_map = (2.0 * sigma_first_second + c2) / (sigma_first_sq + sigma_second_sq + c2).clamp_min(1e-6)
        return ssim_map.mean(dim=(1, 2, 3)), cs_map.mean(dim=(1, 2, 3))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        levels = min(int(self.weights.shape[0]), int(torch.floor(torch.log2(torch.tensor(min(first.shape[-2:]), dtype=torch.float32))).item()))
        levels = max(1, levels)
        weights = self.weights[:levels].to(device=first.device, dtype=first.dtype)
        mssim_values = []
        mcs_values = []
        current_first = first
        current_second = second
        for _ in range(levels):
            ssim_value, cs_value = self._ssim(current_first, current_second)
            mssim_values.append(ssim_value.clamp(0.0, 1.0))
            mcs_values.append(cs_value.clamp(0.0, 1.0))
            if min(current_first.shape[-2:]) <= 1:
                break
            current_first = F.avg_pool2d(current_first, kernel_size=2, stride=2, ceil_mode=True)
            current_second = F.avg_pool2d(current_second, kernel_size=2, stride=2, ceil_mode=True)
        if len(mssim_values) == 1:
            return mssim_values[0].mean()
        weights = weights[: len(mssim_values)]
        value = torch.ones_like(mssim_values[0])
        for index in range(len(mssim_values) - 1):
            value = value * (mcs_values[index] ** weights[index])
        value = value * (mssim_values[-1] ** weights[-1])
        return value.mean()


class SobelEdgeMagnitude(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_x = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
        kernel_y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        grayscale = (0.299 * image[:, 0:1]) + (0.587 * image[:, 1:2]) + (0.114 * image[:, 2:3])
        kernel_x = self.kernel_x.to(device=image.device, dtype=image.dtype)
        kernel_y = self.kernel_y.to(device=image.device, dtype=image.dtype)
        edge_x = F.conv2d(grayscale, kernel_x, padding=1)
        edge_y = F.conv2d(grayscale, kernel_y, padding=1)
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


class BandPassDifference(nn.Module):
    def __init__(self, small_kernel: int = 3, large_kernel: int = 9) -> None:
        super().__init__()
        self.small_kernel = small_kernel
        self.large_kernel = large_kernel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        low_small = _avg_blur(image, self.small_kernel)
        low_large = _avg_blur(image, self.large_kernel)
        return low_small - low_large


class VGGFeatureDistance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        weights = None
        weights_mode = "weights=None"
        try:
            weights = models.VGG16_Weights.IMAGENET1K_V1
            weights_mode = "imagenet"
        except AttributeError:  # pragma: no cover
            weights = None
        try:
            features = models.vgg16(weights=weights).features.eval()
        except Exception as exc:  # pragma: no cover
            warnings.warn(
                "Falling back to untrained VGG16 perceptual distance because pretrained weights were unavailable: %s" % exc,
                RuntimeWarning,
            )
            features = models.vgg16(weights=None).features.eval()
            weights_mode = "untrained-fallback"
        for parameter in features.parameters():
            parameter.requires_grad = False
        self.mode = "vgg16-feature-distance:%s" % weights_mode
        self.blocks = nn.ModuleList([features[:4], features[4:9], features[9:16], features[16:23]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))
        self.layer_weights = (0.1, 0.2, 0.3, 0.4)
        self._cached_target: Optional[Tuple[torch.Tensor, ...]] = None

    def set_target(self, target: torch.Tensor) -> None:
        with torch.no_grad():
            self._cached_target = self.encode(target)

    def encode(self, image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        normalized = (image - self.mean) / self.std
        current = normalized
        features = []
        for block in self.blocks:
            current = block(current)
            features.append(current)
        return tuple(features)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction_features = self.encode(prediction)
        target_features = self._cached_target if self._cached_target is not None else self.encode(target)
        losses = []
        for weight, pred_features, target_features_layer in zip(self.layer_weights, prediction_features, target_features):
            losses.append(weight * F.l1_loss(pred_features, target_features_layer))
        return sum(losses)


class LPIPSVGG(nn.Module):
    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.mode = "disabled"
        self.enabled = enabled
        if not enabled:
            self.model = None
            return
        if lpips is not None:
            self.model = lpips.LPIPS(net="vgg").eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.mode = "lpips-vgg"
        else:
            self.model = VGGFeatureDistance()
            self.mode = self.model.mode

    def set_target(self, target: torch.Tensor) -> None:
        if not self.enabled:
            return
        if hasattr(self.model, "set_target"):
            self.model.set_target(target)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return prediction.new_tensor(0.0)
        if lpips is not None and self.mode == "lpips-vgg":
            return self.model((prediction * 2.0) - 1.0, (target * 2.0) - 1.0).mean()
        return self.model(prediction, target)


class CompositeEllipseLoss(nn.Module):
    def __init__(
        self,
        area_regularization_weight: float,
        area_regularizer_power: float = 1.0,
        large_area_threshold: float = 0.03,
        large_area_penalty_scale: float = 2.0,
        base_area_penalty_scale: float = 0.35,
        texture_area_penalty_scale: float = 1.0,
        accent_area_penalty_scale: float = 1.2,
        perceptual_enabled: bool = True,
        edge_loss_enabled: bool = True,
        band_loss_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.area_regularization_weight = area_regularization_weight
        self.area_regularizer_power = area_regularizer_power
        self.large_area_threshold = large_area_threshold
        self.large_area_penalty_scale = large_area_penalty_scale
        self.base_area_penalty_scale = base_area_penalty_scale
        self.texture_area_penalty_scale = texture_area_penalty_scale
        self.accent_area_penalty_scale = accent_area_penalty_scale
        self.edge_loss_enabled = edge_loss_enabled
        self.band_loss_enabled = band_loss_enabled
        self.ms_ssim = MultiScaleSSIM()
        self.edge_detector = SobelEdgeMagnitude()
        self.band_detector = BandPassDifference()
        self.perceptual = LPIPSVGG(enabled=perceptual_enabled)
        self._cached_target_edges: Optional[torch.Tensor] = None
        self._cached_normalized_target_edges: Optional[torch.Tensor] = None
        self._cached_target_band: Optional[torch.Tensor] = None
        self._cached_target_blur: Optional[torch.Tensor] = None

    @property
    def perceptual_mode(self) -> str:
        return self.perceptual.mode

    def set_target(self, target: torch.Tensor) -> None:
        with torch.no_grad():
            if self.edge_loss_enabled:
                self._cached_target_edges = self.edge_detector(target)
                edge_mean = self._cached_target_edges.mean(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                self._cached_normalized_target_edges = self._cached_target_edges / edge_mean
            else:
                self._cached_target_edges = None
                self._cached_normalized_target_edges = None
            if self.band_loss_enabled:
                self._cached_target_band = self.band_detector(target)
            else:
                self._cached_target_band = None
            self._cached_target_blur = _avg_blur(target, kernel_size=9)
        self.perceptual.set_target(target)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        decoded: EllipseParameters,
        loss_weights: LossWeights,
        weighted_l1_edge_gain: float,
        enable_lpips: bool,
        collect_timing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        timings: Dict[str, float] = {}

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if weighted_l1_edge_gain > 0.0 and self._cached_normalized_target_edges is not None:
            l1_weights = (1.0 + (weighted_l1_edge_gain * self._cached_normalized_target_edges)).expand(-1, prediction.shape[1], -1, -1)
            l1_loss = (torch.abs(prediction - target) * l1_weights).mean()
        else:
            l1_loss = F.l1_loss(prediction, target)
        if collect_timing:
            _sync_if_needed(l1_loss)
            timings["loss_l1_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if loss_weights.blur_l1 > 0.0:
            target_blur = self._cached_target_blur if self._cached_target_blur is not None else _avg_blur(target, kernel_size=9)
            prediction_blur = _avg_blur(prediction, kernel_size=9)
            blur_l1_loss = F.l1_loss(prediction_blur, target_blur)
        else:
            blur_l1_loss = prediction.new_tensor(0.0)
        if collect_timing:
            _sync_if_needed(blur_l1_loss)
            timings["loss_blur_l1_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if loss_weights.band_l1 > 0.0:
            target_band = self._cached_target_band if self._cached_target_band is not None else self.band_detector(target)
            prediction_band = self.band_detector(prediction)
            band_loss = F.l1_loss(prediction_band, target_band)
            bandpass_rmse = rgb_rmse(prediction_band, target_band)
        else:
            band_loss = prediction.new_tensor(0.0)
            bandpass_rmse = prediction.new_tensor(0.0)
        if collect_timing:
            _sync_if_needed(band_loss)
            timings["loss_band_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if loss_weights.ms_ssim > 0.0:
            ms_ssim_score = self.ms_ssim(prediction, target)
        else:
            ms_ssim_score = prediction.new_tensor(1.0)
        if collect_timing:
            _sync_if_needed(ms_ssim_score)
            timings["loss_ms_ssim_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        perceptual_loss = self.perceptual(prediction, target) if enable_lpips else prediction.new_tensor(0.0)
        if collect_timing:
            _sync_if_needed(perceptual_loss)
            timings["loss_lpips_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if loss_weights.edge_l1 > 0.0:
            target_edges = self._cached_target_edges if self._cached_target_edges is not None else self.edge_detector(target)
            edge_loss = F.l1_loss(self.edge_detector(prediction), target_edges)
        else:
            edge_loss = prediction.new_tensor(0.0)
        if collect_timing:
            _sync_if_needed(edge_loss)
            timings["loss_edge_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        if self.area_regularization_weight > 0.0:
            triangle_area = normalized_triangle_area(decoded.sizes[:, 0], decoded.sizes[:, 1]).clamp_min(1e-8)
            large_area_multiplier = torch.where(
                triangle_area > self.large_area_threshold,
                torch.full_like(triangle_area, self.large_area_penalty_scale),
                torch.ones_like(triangle_area),
            )
            role_scale = torch.full_like(triangle_area, self.accent_area_penalty_scale)
            role_scale = torch.where(decoded.role_ids == ROLE_BASE, torch.full_like(role_scale, self.base_area_penalty_scale), role_scale)
            role_scale = torch.where(decoded.role_ids == ROLE_TEXTURE, torch.full_like(role_scale, self.texture_area_penalty_scale), role_scale)
            area_regularizer = torch.mean((triangle_area**self.area_regularizer_power) * large_area_multiplier * role_scale)
        else:
            area_regularizer = prediction.new_tensor(0.0)
        if collect_timing:
            _sync_if_needed(area_regularizer)
            timings["loss_regularizer_seconds"] = time.perf_counter() - start

        if collect_timing:
            _sync_if_needed(prediction)
        start = time.perf_counter()
        total = (
            loss_weights.l1 * l1_loss
            + loss_weights.blur_l1 * blur_l1_loss
            + loss_weights.band_l1 * band_loss
            + loss_weights.ms_ssim * (1.0 - ms_ssim_score)
            + loss_weights.lpips * perceptual_loss
            + loss_weights.edge_l1 * edge_loss
            + self.area_regularization_weight * area_regularizer
        )
        if collect_timing:
            _sync_if_needed(total)
            timings["loss_assembly_seconds"] = time.perf_counter() - start

        metrics = {
            "loss": total,
            "l1": l1_loss,
            "blur_l1": blur_l1_loss,
            "band_l1": band_loss,
            "ms_ssim": ms_ssim_score,
            "lpips": perceptual_loss,
            "edge_l1": edge_loss,
            "area_reg": area_regularizer,
            "rgb_rmse": rgb_rmse(prediction, target),
            "bandpass_rmse": bandpass_rmse,
        }
        return total, metrics, timings


