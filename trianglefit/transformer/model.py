from __future__ import annotations

import math
import warnings
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

from trianglefit.direct.model import EllipseParameters, ROLE_ACCENT, ROLE_BASE, ROLE_TEXTURE


def _load_resnet18(pretrained: bool) -> nn.Module:
    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        except AttributeError:  # pragma: no cover
            weights = None
    try:
        resnet = models.resnet18(weights=weights)
    except Exception as exc:  # pragma: no cover
        warnings.warn("Falling back to randomly initialized ResNet18: %s" % exc, RuntimeWarning)
        resnet = models.resnet18(weights=None)
    return nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)


def _build_2d_sincos_position(height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for 2D sine-cosine position encoding.")
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1, (dim // 4) - 1)))
    out_x = grid_x.reshape(-1, 1) * omega.view(1, -1)
    out_y = grid_y.reshape(-1, 1) * omega.view(1, -1)
    return torch.cat((torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)), dim=1)


def _make_anchor_grid(num_slots: int) -> torch.Tensor:
    """Generate normalized slot anchors that cover the canvas uniformly."""
    columns = int(math.ceil(math.sqrt(float(num_slots))))
    rows = int(math.ceil(float(num_slots) / float(columns)))
    ys = torch.linspace(0.5 / rows, 1.0 - (0.5 / rows), rows)
    xs = torch.linspace(0.5 / columns, 1.0 - (0.5 / columns), columns)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    anchors = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return anchors[:num_slots]


class TriangleTransformerGenerator(nn.Module):
    def __init__(
        self,
        num_triangles: int,
        base_count: int,
        texture_count: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        pretrained_backbone: bool = True,
        freeze_backbone: bool = True,
        min_size: float = 0.002,
        base_max_size: float = 0.34,
        texture_max_size: float = 0.16,
    ) -> None:
        super().__init__()
        self.num_triangles = int(num_triangles)
        self.base_count = int(base_count)
        self.texture_count = int(texture_count)
        self.d_model = int(d_model)
        self.min_size = float(min_size)
        self.base_max_size = float(base_max_size)
        self.texture_max_size = float(texture_max_size)

        self.backbone = _load_resnet18(pretrained=pretrained_backbone)
        if freeze_backbone:
            self.backbone.eval()
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        # ResNet18 layer4 杈撳嚭 512 閫氶亾锛?x1 鍗风Н鎶婂畠鎶曞奖鎴?Transformer 鐨?d_model銆?        self.input_projection = nn.Conv2d(512, self.d_model, kernel_size=1)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(num_decoder_layers))

        # 姣忎釜 query 灏辨槸涓€涓彲瀛︿範鐨勪笁瑙掑舰 slot锛岃礋璐ｅ拰鍏朵粬 slot 鍗忓晢鍒嗗伐銆?        self.triangle_queries = nn.Parameter(torch.randn(self.num_triangles, self.d_model) * 0.02)
        self.anchor_projection = nn.Linear(2, self.d_model)
        self.register_buffer("anchors", _make_anchor_grid(self.num_triangles))
        self.param_head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 8),
        )
        self._reset_parameter_head()

    def _reset_parameter_head(self) -> None:
        final_layer = self.param_head[-1]
        if not isinstance(final_layer, nn.Linear):
            return
        with torch.no_grad():
            final_layer.weight[0:2].zero_()
            final_layer.bias[0:2].zero_()

    def train(self, mode: bool = True) -> "TriangleTransformerGenerator":
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def _role_ids(self, device: torch.device) -> torch.Tensor:
        role_ids = torch.full((self.num_triangles,), ROLE_ACCENT, device=device, dtype=torch.long)
        role_ids[: self.base_count] = ROLE_BASE
        role_ids[self.base_count : self.base_count + self.texture_count] = ROLE_TEXTURE
        return role_ids

    def _size_limits(self, role_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        limits = torch.full((self.num_triangles, 2), self.texture_max_size, device=role_ids.device, dtype=dtype)
        limits = torch.where((role_ids == ROLE_BASE).view(-1, 1), torch.full_like(limits, self.base_max_size), limits)
        limits = torch.where((role_ids == ROLE_ACCENT).view(-1, 1), torch.full_like(limits, self.texture_max_size * 0.6), limits)
        return limits

    def forward(self, target_image: torch.Tensor) -> EllipseParameters:
        # Frozen ResNet 璐熻矗鎶婂浘鍍忚浆鎴愯涔?feature map锛涜缁冧富瑕佸彂鐢熷湪鍚庨潰鐨?query 鍜?decoder銆?        if any(parameter.requires_grad for parameter in self.backbone.parameters()):
            features = self.backbone(target_image)
        else:
            with torch.no_grad():
                features = self.backbone(target_image)
        projected = self.input_projection(features)
        batch_size, _, height, width = projected.shape

        # memory tokens 鏄浘鍍?patch/鍖哄煙鐗瑰緛锛宲osition encoding 璁?Transformer 鐭ラ亾姣忎釜 token 鐨勭┖闂翠綅缃€?        memory = projected.flatten(2).transpose(1, 2)
        position = _build_2d_sincos_position(height, width, self.d_model, projected.device, projected.dtype)
        memory = memory + position.unsqueeze(0)

        anchors = self.anchors.to(device=target_image.device, dtype=target_image.dtype)
        anchor_queries = self.anchor_projection(anchors).unsqueeze(0)
        queries = self.triangle_queries.unsqueeze(0).expand(batch_size, -1, -1) + anchor_queries
        decoded_queries = self.decoder(tgt=queries, memory=memory)
        raw = self.param_head(decoded_queries)[0]

        role_ids = self._role_ids(target_image.device)
        size_limits = self._size_limits(role_ids, target_image.dtype)
        center_offsets = 0.22 * torch.tanh(raw[:, 0:2])
        centers = (anchors + center_offsets).clamp(0.0, 1.0)
        sizes = self.min_size + (torch.sigmoid(raw[:, 2:4]) * size_limits)
        theta = math.pi * torch.tanh(raw[:, 4:5])
        rgb = torch.sigmoid(raw[:, 5:8])
        alpha = torch.ones((self.num_triangles, 1), device=target_image.device, dtype=target_image.dtype)
        background = target_image.mean(dim=(2, 3), keepdim=True).clamp(0.0, 1.0)

        return EllipseParameters(
            centers=centers,
            sizes=sizes,
            theta=theta,
            rgb=rgb,
            alpha=alpha,
            role_ids=role_ids,
            background_grid_rgb=background,
        )

    def role_counts(self) -> Tuple[int, int, int]:
        accent_count = max(0, self.num_triangles - self.base_count - self.texture_count)
        return self.base_count, self.texture_count, accent_count


