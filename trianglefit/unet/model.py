from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F

from trianglefit.direct.model import EllipseParameters, ROLE_ACCENT, ROLE_BASE, ROLE_TEXTURE


def _make_anchor_grid(num_slots: int) -> torch.Tensor:
    """鐢熸垚鍧囧寑瑕嗙洊鐢婚潰鐨?slot anchor锛屾瘡涓?anchor 瀵瑰簲涓€涓笁瑙掑舰鐨勫垵濮嬭礋璐ｅ尯鍩熴€?""
    columns = int(math.ceil(math.sqrt(float(num_slots))))
    rows = int(math.ceil(float(num_slots) / float(columns)))
    ys = torch.linspace(0.5 / rows, 1.0 - (0.5 / rows), rows)
    xs = torch.linspace(0.5 / columns, 1.0 - (0.5 / columns), columns)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    anchors = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
    return anchors[:num_slots]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 涓ゅ眰 3x3 鍗风Н璐熻矗鎻愬彇灞€閮ㄩ鑹层€佽竟缂樺拰鍧楅潰缁撴瀯銆?        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TriangleUNetGenerator(nn.Module):
    def __init__(
        self,
        num_triangles: int,
        base_count: int,
        texture_count: int,
        hidden_channels: int = 32,
        min_size: float = 0.002,
        base_max_size: float = 0.34,
        texture_max_size: float = 0.16,
    ) -> None:
        super().__init__()
        self.num_triangles = int(num_triangles)
        self.base_count = int(base_count)
        self.texture_count = int(texture_count)
        self.min_size = float(min_size)
        self.base_max_size = float(base_max_size)
        self.texture_max_size = float(texture_max_size)

        c = int(hidden_channels)
        self.enc1 = ConvBlock(3, c)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ConvBlock(c * 2, c * 2)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ConvBlock(c * 4, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)

        # 姣忎釜涓夎褰細鎷垮埌锛氬眬閮?U-Net 鐗瑰緛 + 鍏ㄥ眬鍥惧儚鐗瑰緛 + 鑷繁鐨?anchor 鍧愭爣銆?        self.slot_head = nn.Sequential(
            nn.Linear((c * 2) + 2, c * 2),
            nn.SiLU(inplace=True),
            nn.Linear(c * 2, 8),
        )
        self.register_buffer("anchors", _make_anchor_grid(self.num_triangles))

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

    def _sample_slots(self, feature_map: torch.Tensor) -> torch.Tensor:
        batch_size = feature_map.shape[0]
        anchors = self.anchors.to(device=feature_map.device, dtype=feature_map.dtype)
        # grid_sample 浣跨敤 [-1, 1] 鍧愭爣锛岃繖閲屾妸褰掍竴鍖栧浘鍍忓潗鏍囪浆鎹㈣繃鍘汇€?        sample_grid = ((anchors * 2.0) - 1.0).view(1, self.num_triangles, 1, 2)
        sample_grid = sample_grid.expand(batch_size, -1, -1, -1)
        sampled = F.grid_sample(feature_map, sample_grid, mode="bilinear", align_corners=False)
        return sampled.squeeze(-1).permute(0, 2, 1)

    def forward(self, target_image: torch.Tensor) -> EllipseParameters:
        # 缂栫爜鍣細閫愭闄嶄綆鍒嗚鲸鐜囷紝璁╃綉缁滅湅鍒版洿澶х殑涓婁笅鏂囥€?        enc1 = self.enc1(target_image)
        enc2 = self.enc2(self.down1(enc1))
        bottleneck = self.bottleneck(self.down2(enc2))

        # 瑙ｇ爜鍣細閫愭鎭㈠鍒嗚鲸鐜囷紝骞堕€氳繃 skip connection 鎷垮洖娴呭眰缁嗚妭銆?        up2 = self.up2(bottleneck)
        if up2.shape[-2:] != enc2.shape[-2:]:
            up2 = F.interpolate(up2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))

        up1 = self.up1(dec2)
        if up1.shape[-2:] != enc1.shape[-2:]:
            up1 = F.interpolate(up1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))

        local_features = self._sample_slots(dec1)
        global_feature = dec1.mean(dim=(2, 3)).unsqueeze(1).expand(-1, self.num_triangles, -1)
        anchors = self.anchors.to(device=target_image.device, dtype=target_image.dtype)
        anchor_features = anchors.unsqueeze(0).expand(target_image.shape[0], -1, -1)
        raw = self.slot_head(torch.cat((local_features, global_feature, anchor_features), dim=-1))

        # 褰撳墠瀹為獙鍙仛鍗曞浘璁粌锛屾墍浠?batch 鍥哄畾浣跨敤绗?0 寮犲浘銆?        raw = raw[0]
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



