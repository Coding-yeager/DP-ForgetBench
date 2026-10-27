"""Architecture tier ladder for DP-ForgetBench utility experiments.

Implements the 5-tier model ladder recommended by the AI agent review.
Each tier trades trainable parameter count for better utility under tight
client-level DP, while keeping the benchmark's federated protocol intact.

Tier 0 : SmallGroupNormCNN from scratch (~60k params)   [existing baseline]
Tier 1 : ScatterNet fixed wavelet + linear head          [no external data]
Tier 2 : Frozen pretrained backbone + linear head        [self-supervised]
Tier 3 : Frozen backbone + nearest-class-mean (NCM)     [one-round, exact unlearn]
Tier 4 : Frozen backbone + LoRA / BiTFiT head           [limited fine-tuning]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Tier 0: SmallGroupNormCNN (existing, imported from federated.py)
# ---------------------------------------------------------------------------
# Just re-exported here for uniform naming in the experiment runner.
from dp_forgetbench.federated import SmallGroupNormCNN  # noqa: F401


# ---------------------------------------------------------------------------
# Tier 1: Fixed Morlet / Haar ScatterNet features + GroupNorm linear head
# ---------------------------------------------------------------------------

class HaarScatterLayer(nn.Module):
    """Single-level 2D Haar wavelet scattering (no learned params).

    Applies the Haar wavelet in horizontal, vertical, and diagonal
    directions at two scales, then takes the modulus. Produces
    fixed translation-invariant features without external data.

    Output: (N, 3 * 4 * 4 * 4) = (N, 192) for a 32x32 image.
    """

    def __init__(self) -> None:
        super().__init__()
        # Haar wavelets: LL (avg), LH (horizontal), HL (vertical), HH (diagonal)
        _h = torch.tensor([[1., 1.], [1., 1.]]) / 2.0          # LL
        _lh = torch.tensor([[1., 1.], [-1., -1.]]) / 2.0       # LH
        _hl = torch.tensor([[1., -1.], [1., -1.]]) / 2.0       # HL
        _hh = torch.tensor([[1., -1.], [-1., 1.]]) / 2.0       # HH

        kernels_2d = torch.stack([_h, _lh, _hl, _hh])          # (4, 2, 2)
        # Expand to (4, 1, 2, 2) for depthwise conv over 3 channels
        kernels = kernels_2d.unsqueeze(1)                       # (4, 1, 2, 2)
        # Register as buffer (not a parameter)
        self.register_buffer("kernels", kernels)
        self.out_channels = 4 * 4 * 4  # 4 filters * 4 sub-bands * stride-2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernels = self.kernels.to(x.device)
        # Level-1: depthwise conv over C input channels, producing C*4 channels
        k_level1 = kernels.repeat(C, 1, 1, 1)      # (C*4, 1, 2, 2)
        out = F.conv2d(x, k_level1, stride=2, padding=0, groups=C)  # (B, C*4, H/2, W/2)
        # Level-2: depthwise conv over C*4 channels from level-1
        C2 = C * 4
        k_level2 = kernels.repeat(C2, 1, 1, 1)     # (C2*4, 1, 2, 2)
        out2 = F.conv2d(out, k_level2, stride=2, padding=0, groups=C2)  # (B, C2*4, H/4, W/4)
        # Modulus + global avg pool for translation invariance
        feat1 = F.adaptive_avg_pool2d(out.abs(), (1, 1)).flatten(1)
        feat2 = F.adaptive_avg_pool2d(out2.abs(), (1, 1)).flatten(1)
        return torch.cat([feat1, feat2], dim=1)


class Tier1ScatterLinear(nn.Module):
    """ScatterNet features → small GroupNorm linear head.

    All scattering is fixed (no gradient). Only the linear classifier is
    trained under DP, keeping d small (≈ C * 4 + C * 4 features).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.scatter = HaarScatterLayer()
        # Feature dim: C=3, 2 levels, 4 filters each → 3*4*2 = 24 per pool
        # After global avg pool: 3*4 + 3*4 = 24 per level × 2 = 48? Let's calculate dynamically.
        dummy = torch.zeros(1, 3, 32, 32)
        with torch.no_grad():
            feat_dim = self.scatter(dummy).shape[1]
        n_groups = min(8, feat_dim)
        while feat_dim % n_groups != 0:
            n_groups -= 1
        self.norm = nn.GroupNorm(n_groups, feat_dim)
        self.head = nn.Linear(feat_dim, num_classes)
        self._feat_dim = feat_dim

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            with torch.no_grad():  # Scatter is always frozen
                feat = self.scatter(x)
        else:
            feat = x
        feat = self.norm(feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Tier 2: Frozen pretrained backbone + GroupNorm linear head
# ---------------------------------------------------------------------------

class FrozenBackboneLinear(nn.Module):
    """A frozen pretrained encoder with a DP-trained GroupNorm linear head.

    The backbone is loaded once, features are computed in eval mode with
    no gradients (saving both memory and privacy budget). Only the GroupNorm
    scale/bias and the linear head are trainable.

    By default uses a small MobileNetV3-Small from torchvision, which:
      - Has no CIFAR-10 classes in its training objective (ImageNet).
      - Produces 576-d features.
      - Fits comfortably in 8 GB VRAM even at batch_size=256.
    """

    # Public statistics computed on ImageNet (used by MobileNetV3 training)
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        backbone_name: str = "dinov2_small",
        num_classes: int = 10,
        pretrained: bool = True,
        feature_dim: int | None = None,
        upsample_size: tuple[int, int] | None = (224, 224),
        use_cosine_norm: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.upsample_size = upsample_size if any(k in backbone_name for k in ("dinov2", "upsample", "bit", "vit", "resnet50")) else None
        self.use_cosine_norm = use_cosine_norm if any(k in backbone_name for k in ("dinov2", "upsample", "bit", "vit", "resnet50")) else False
        self._build_backbone(backbone_name, pretrained)

        # Determine feature dimension
        if feature_dim is None:
            dummy = torch.zeros(1, 3, 32, 32)
            with torch.no_grad():
                feat = self._encode(dummy)
            feature_dim = feat.shape[1]
        self.feature_dim = feature_dim

        # If cosine norm is active, only the linear head is trained under DP
        if self.use_cosine_norm:
            self.norm = nn.Identity()
        else:
            n_groups = min(8, feature_dim)
            while feature_dim % n_groups != 0:
                n_groups -= 1
            self.norm = nn.GroupNorm(n_groups, feature_dim)

        self.head = nn.Linear(feature_dim, num_classes)
        print(f"[Tier2] Backbone: {backbone_name}, Feature dim: {feature_dim}, "
              f"Trainable params: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def _build_backbone(self, name: str, pretrained: bool) -> None:
        try:
            if name in ("dinov2_small", "dinov2_vits14"):
                import torch
                self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
                self.upsample_size = (224, 224)
                self.use_cosine_norm = True
            elif "bit" in name:
                try:
                    import timm
                    self.backbone = timm.create_model("resnetv2_50x1_bit", pretrained=pretrained, num_classes=0)
                except Exception:
                    import torchvision.models as tvm
                    weights = "DEFAULT" if pretrained else None
                    model = tvm.wide_resnet50_2(weights=weights)
                    self.backbone = nn.Sequential(*list(model.children())[:-1])
                self.upsample_size = (224, 224)
                self.use_cosine_norm = True
            elif "vit" in name:
                import torchvision.models as tvm
                weights = "DEFAULT" if pretrained else None
                model = tvm.vit_b_16(weights=weights)
                self.backbone = nn.Sequential(*list(model.children())[:-1])
                self.upsample_size = (224, 224)
                self.use_cosine_norm = True
            elif name == "mobilenet_v3_small":
                import torchvision.models as tvm
                weights = "DEFAULT" if pretrained else None
                model = tvm.mobilenet_v3_small(weights=weights)
                self.backbone = nn.Sequential(*list(model.children())[:-1])
                self.upsample_size = None
                self.use_cosine_norm = False
            elif name == "resnet18":
                import torchvision.models as tvm
                weights = "DEFAULT" if pretrained else None
                model = tvm.resnet18(weights=weights)
                self.backbone = nn.Sequential(*list(model.children())[:-1])
                self.upsample_size = None
                self.use_cosine_norm = False
            elif name in ("resnet50", "resnet50_upsample"):
                import torchvision.models as tvm
                weights = "DEFAULT" if pretrained else None
                model = tvm.resnet50(weights=weights)
                self.backbone = nn.Sequential(*list(model.children())[:-1])
                self.upsample_size = (224, 224)
                self.use_cosine_norm = True
            else:
                raise ValueError(f"Unsupported backbone: {name}")
        except Exception as e:
            raise RuntimeError(f"Failed to load backbone '{name}': {e}") from e

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without gradients. Always runs in eval mode."""
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            if getattr(self, "upsample_size", None) is not None and x.shape[-2:] != self.upsample_size:
                x = F.interpolate(x, size=self.upsample_size, mode="bilinear", align_corners=False)
            feat = self.backbone(x)
            result = feat.flatten(1)
            if getattr(self, "use_cosine_norm", False):
                result = F.normalize(result, p=2, dim=-1)
        if was_training:
            self.backbone.train()
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            feat = self._encode(x)
        else:
            feat = x
            if getattr(self, "use_cosine_norm", False):
                feat = F.normalize(feat, p=2, dim=-1)
        if not getattr(self, "use_cosine_norm", False):
            feat = self.norm(feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            feat = self.norm(feat)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Tier 3: Frozen backbone + Nearest-Class-Mean (NCM) classifier
# ---------------------------------------------------------------------------

class NearestClassMeanClassifier(nn.Module):
    """One-round Nearest Class Mean (FedNCM) classifier on frozen features.

    Each client sends: per-class feature sums and per-class counts.
    Server aggregates (with DP noise) and computes class prototypes.
    Classification is nearest L2 distance to prototypes.

    Key advantage: Only ONE aggregation round is needed, so the RDP
    composition is over a single mechanism, not T rounds. This gives
    dramatically lower ε for the same noise.

    For unlearning: exact unlearning = subtract client's sufficient stats.
    This also hits the SecAgg barrier if we want to subtract exactly,
    but in the non-SecAgg arm it gives certified deletion.
    """

    def __init__(self, feature_dim: int, num_classes: int = 10) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        # Prototypes are stored as a buffer, not a trained parameter
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(
        self,
        class_feature_sums: dict[int, torch.Tensor],  # class_id -> sum of features
        class_counts: dict[int, int],
    ) -> None:
        """Fit prototypes from aggregated sufficient statistics."""
        for cls in range(self.num_classes):
            if cls in class_feature_sums and class_counts.get(cls, 0) > 0:
                self.prototypes[cls] = class_feature_sums[cls] / class_counts[cls]
        self.fitted = torch.tensor(True)

    def exact_unlearn(
        self,
        removed_class_sums: dict[int, torch.Tensor],
        removed_counts: dict[int, int],
        original_sums: dict[int, torch.Tensor],
        original_counts: dict[int, int],
    ) -> None:
        """Exact unlearning: subtract the forgotten client's sufficient stats."""
        for cls in range(self.num_classes):
            orig_sum = original_sums.get(cls, torch.zeros(self.feature_dim))
            rem_sum = removed_class_sums.get(cls, torch.zeros(self.feature_dim))
            orig_count = original_counts.get(cls, 0)
            rem_count = removed_counts.get(cls, 0)
            new_count = orig_count - rem_count
            if new_count > 0:
                new_sum = orig_sum - rem_sum
                self.prototypes[cls] = new_sum / new_count
            else:
                # Client was sole contributor to this class; reset to zero proto
                self.prototypes[cls] = torch.zeros(self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns negative squared distances to each prototype as logits."""
        # x: (N, d), prototypes: (C, d)
        # dist²(x_i, p_c) = ||x_i||² - 2 x_i·p_c + ||p_c||²
        dists = (
            (x ** 2).sum(dim=1, keepdim=True)
            - 2 * x @ self.prototypes.T
            + (self.prototypes ** 2).sum(dim=1, keepdim=True).T
        )
        return -dists  # Negative dists as logits (argmax = nearest)


class Tier3FrozenNCM(nn.Module):
    """Frozen backbone + NCM head. Full inference model for Tier 3."""

    def __init__(
        self,
        backbone_name: str = "dinov2_small",
        num_classes: int = 10,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self._tier2 = FrozenBackboneLinear(backbone_name, num_classes, pretrained)
        self.backbone = self._tier2.backbone
        self.ncm = NearestClassMeanClassifier(self._tier2.feature_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._tier2._encode(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            feat = self.encode(x)
        else:
            feat = x
        return self.ncm(feat)


# ---------------------------------------------------------------------------
# Tier 4: Frozen backbone + LoRA adapter on head
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation on a linear layer.

    W_eff = W_frozen + (B @ A) * scale
    A: (r, in_features), B: (out_features, r)
    Trainable params: r * (in + out) instead of in * out.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scale = alpha / rank

        # Frozen base weight (initialized to zero, actual weights loaded externally)
        self.register_buffer("weight", torch.zeros(out_features, in_features))
        self.register_buffer("bias_vec", torch.zeros(out_features))

        # LoRA factors (trainable)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias_vec)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    @property
    def trainable_params(self) -> int:
        return self.rank * (self.in_features + self.out_features)


class Tier4FrozenLoRA(nn.Module):
    """Frozen backbone + LoRA-adapted linear head (Tier 4).

    Trainable: LoRA A, LoRA B.
    Frozen: backbone weights.
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_small",
        num_classes: int = 10,
        lora_rank: int = 16,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        base = FrozenBackboneLinear(backbone_name, num_classes, pretrained)
        self.backbone = base.backbone
        self.upsample_size = base.upsample_size
        self.use_cosine_norm = base.use_cosine_norm
        feat_dim = base.feature_dim
        self.feature_dim = feat_dim
        self.norm = base.norm

        # LoRA head replacing the linear head
        self.head = LoRALinear(feat_dim, num_classes, rank=lora_rank)
        with torch.no_grad():
            self.head.weight.copy_(base.head.weight.data)
            self.head.bias_vec.copy_(base.head.bias.data)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Tier4-LoRA] rank={lora_rank}, trainable params: {trainable:,}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            if getattr(self, "upsample_size", None) is not None and x.shape[-2:] != self.upsample_size:
                x = F.interpolate(x, size=self.upsample_size, mode="bilinear", align_corners=False)
            feat = self.backbone(x).flatten(1)
            if getattr(self, "use_cosine_norm", False):
                feat = F.normalize(feat, p=2, dim=-1)
        if was_training:
            self.backbone.train()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            feat = self._encode(x)
        else:
            feat = x
            if getattr(self, "use_cosine_norm", False):
                feat = F.normalize(feat, p=2, dim=-1)
        if not getattr(self, "use_cosine_norm", False):
            feat = self.norm(feat.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            feat = self.norm(feat)
        return self.head(feat)


# ---------------------------------------------------------------------------
# Tier factory
# ---------------------------------------------------------------------------

@dataclass
class TierSpec:
    tier_id: int
    name: str
    description: str
    trainable_params: int
    model_factory: Callable[[], nn.Module]


def get_tier_specs(num_classes: int = 10, backbone_name: str = "dinov2_small") -> list[TierSpec]:
    """Return the full architecture tier ladder."""
    dim = 384 if "dinov2" in backbone_name else (2048 if any(k in backbone_name for k in ("resnet50", "bit")) else 576)
    trainable_t2 = dim * num_classes + num_classes
    trainable_t4 = 16 * (dim + num_classes)
    return [
        TierSpec(
            tier_id=0,
            name="SmallGroupNormCNN (from scratch)",
            description="Baseline 2-layer CNN ~60k params trained from scratch.",
            trainable_params=60_586,
            model_factory=lambda: SmallGroupNormCNN(num_classes),
        ),
        TierSpec(
            tier_id=1,
            name="HaarScatter + Linear Head",
            description="Fixed Haar wavelet scattering (no external data). Small linear head trained under DP.",
            trainable_params=-1,  # computed dynamically
            model_factory=lambda: Tier1ScatterLinear(num_classes),
        ),
        TierSpec(
            tier_id=2,
            name=f"Frozen {backbone_name} + Linear Head",
            description=f"Foundation backbone ({backbone_name}) with 224x224 upsampling and cosine norm. Only {trainable_t2:,} params trained under DP.",
            trainable_params=trainable_t2,
            model_factory=lambda b=backbone_name: FrozenBackboneLinear(b, num_classes, pretrained=True),
        ),
        TierSpec(
            tier_id=3,
            name=f"Frozen {backbone_name} + NCM Head",
            description=f"One-round NCM: aggregates class prototypes from {backbone_name}. Exact unlearning by subtracting stats.",
            trainable_params=0,  # No SGD, closed-form
            model_factory=lambda b=backbone_name: Tier3FrozenNCM(b, num_classes, pretrained=True),
        ),
        TierSpec(
            tier_id=4,
            name=f"Frozen {backbone_name} + LoRA-16 Head",
            description=f"LoRA rank-16 adapter on linear head with {backbone_name}.",
            trainable_params=trainable_t4,
            model_factory=lambda b=backbone_name: Tier4FrozenLoRA(b, num_classes, lora_rank=16, pretrained=True),
        ),
    ]

