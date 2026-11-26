"""モデル定義。事前学習済み重みは一切使わない（CLAUDE.md 3 章）。

model 設定（exp の YAML）:
  name: timm のモデル名
  pool: "avg"（既定、timm 標準の global average pooling）/ "center"（最終特徴マップの中央だけを平均）
  center_frac: pool=center のとき、特徴マップ一辺に対する中央領域の割合（m=256 の 3×3 タイル入力なら 1/3）
  kwargs: timm.create_model へ渡す追加引数
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn


class CenterPoolNet(nn.Module):
    """backbone の最終特徴マップ (B,C,H,W) の中央 center_frac 四方だけを平均して 1 ロジットを出す。
    384 入力の ResNet18 なら 12×12 の中央 4×4（= 中央タイル相当）。"""

    def __init__(self, name: str, center_frac: float, **kwargs):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=False, num_classes=0, global_pool="", **kwargs)
        self.center_frac = float(center_frac)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone.forward_features(x)
        H, W = f.shape[-2:]
        hs = int(round(H * (1 - self.center_frac) / 2))
        ws = int(round(W * (1 - self.center_frac) / 2))
        f = f[:, :, hs:H - hs, ws:W - ws].mean((2, 3))
        return self.fc(f)


def build_model(name: str, **kwargs) -> nn.Module:
    kwargs.pop("pretrained", None)
    return timm.create_model(name, pretrained=False, num_classes=1, **kwargs)


def build_from_cfg(mcfg: dict) -> nn.Module:
    kw = dict(mcfg.get("kwargs", {}) or {})
    if mcfg.get("pool", "avg") == "center":
        return CenterPoolNet(mcfg["name"], mcfg["center_frac"], **kw)
    return build_model(mcfg["name"], **kw)
