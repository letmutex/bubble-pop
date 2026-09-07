"""BubbleUNet deployment architecture and training-checkpoint loader."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


class ConvBNAct(nn.Module):
    """Convolution + foldable BatchNorm + activation."""

    def __init__(
        self,
        in_ch,
        out_ch,
        k=3,
        stride=1,
        groups=1,
        dilation=1,
        act=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=stride,
            padding=k // 2 * dilation,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class EfficientChannelAttention(nn.Module):
    """Fast, nearly parameter-free channel attention (ECA) module."""

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class GeometryTower(nn.Module):
    """Small nonlinear branch for boundary, tail, and signed distance."""

    def __init__(self, feature_channels, hidden_channels=8):
        super().__init__()
        self.projection = ConvBNAct(feature_channels, hidden_channels, k=1)
        self.heads = nn.Conv2d(hidden_channels, 3, kernel_size=1)

    def forward(self, feature):
        return self.heads(self.projection(feature))


class DilatedBottleneck(nn.Module):
    """PicoSAM3-style bottleneck: dilated depthwise conv expands the receptive
    field without extra downsampling; pointwise convs expand/shrink channels."""

    def __init__(self, ch):
        super().__init__()
        hidden = int(ch * 1.25)
        self.dw_in = ConvBNAct(ch, ch, k=3, dilation=2, groups=ch)
        self.pw_expand = ConvBNAct(ch, hidden, k=1)
        self.dw_hidden = ConvBNAct(hidden, hidden, k=3, groups=hidden)
        self.pw_project = ConvBNAct(hidden, ch, k=1)

    def forward(self, x):
        return x + self.pw_project(self.dw_hidden(self.pw_expand(self.dw_in(x))))


class DecoderBlock(nn.Module):
    """Slim U-Net up-sampling block: bilinear upsample -> concat skip ->
    depthwise 3x3 -> pointwise -> depthwise 3x3 refine -> pointwise.
    No residual shortcut and no per-block attention (params/compute saved)."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        fused_ch = in_ch + skip_ch
        self.fuse = nn.Sequential(
            ConvBNAct(fused_ch, fused_ch, k=3, groups=fused_ch),
            ConvBNAct(fused_ch, out_ch, k=1),
            ConvBNAct(out_ch, out_ch, k=3, groups=out_ch),
            ConvBNAct(out_ch, out_ch, k=1),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            )
        return self.fuse(x)


class MobileNetV3Encoder(nn.Module):
    """ImageNet-pretrained MobileNetV3 feature pyramid extractor (grayscale single-channel input).
    Stage slices are defined per variant because Large and Small place their
    downsampling steps at different layer indices."""

    # Each entry yields feature maps at strides (2, 4, 8, 16, 32)
    STAGE_SLICES = {
        "large": [(0, 2), (2, 4), (4, 7), (7, 13), (13, 16)],
        "small": [(0, 1), (1, 2), (2, 4), (4, 9), (9, 12)],
    }

    def __init__(self, variant="large", pretrained=False):
        super().__init__()
        if variant == "large":
            weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
            features = mobilenet_v3_large(weights=weights).features
        elif variant == "small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            features = mobilenet_v3_small(weights=weights).features
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Replace 3-channel input convolution with 1-channel grayscale convolution
        old_conv = features[0][0]
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if pretrained and old_conv.weight is not None:
            with torch.no_grad():
                new_conv.weight.copy_(
                    old_conv.weight.sum(dim=1, keepdim=True)
                )
        features[0][0] = new_conv

        slices = self.STAGE_SLICES[variant]
        self.stage1 = nn.Sequential(*features[slices[0][0]:slices[0][1]])  # 1/2
        self.stage2 = nn.Sequential(*features[slices[1][0]:slices[1][1]])  # 1/4
        self.stage3 = nn.Sequential(*features[slices[2][0]:slices[2][1]])  # 1/8
        self.stage4 = nn.Sequential(*features[slices[3][0]:slices[3][1]])  # 1/16
        self.stage5 = nn.Sequential(*features[slices[4][0]:slices[4][1]])  # 1/32

    def forward(self, x):
        c1 = self.stage1(x)   # 1/2
        c2 = self.stage2(c1)  # 1/4
        c3 = self.stage3(c2)  # 1/8
        c4 = self.stage4(c3)  # 1/16
        c5 = self.stage5(c4)  # 1/32
        return c1, c2, c3, c4, c5


class BubbleUNet(nn.Module):
    """Grayscale Bubble U-Net returning mask, boundary, and confidence logits."""

    ARCHITECTURE_VERSION = "mobilenet_v3_geometry_no_refiner_v2"
    DECODER_OUT_CH = (64, 48, 32, 24)  # dec4, dec3, dec2, dec1
    TRUNK_CH = 16
    GEOMETRY_CH = 8

    def __init__(self, variant="large", pretrained=False):
        super().__init__()
        self.variant = variant
        self.encoder = MobileNetV3Encoder(variant=variant, pretrained=pretrained)

        # Probe encoder channel widths (works for both variants)
        # without contaminating pretrained BatchNorm running statistics with
        # the all-zero probe batch.
        encoder_was_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            c1, c2, c3, c4, c5 = self.encoder(dummy)
        self.encoder.train(encoder_was_training)
        enc_ch = {
            "c1": c1.shape[1],
            "c2": c2.shape[1],
            "c3": c3.shape[1],
            "c4": c4.shape[1],
            "c5": c5.shape[1],
        }

        # Receptive-field expansion at the deepest level (cheap, low-res)
        self.bottleneck = DilatedBottleneck(enc_ch["c5"])

        dec_out = self.DECODER_OUT_CH
        self.dec4 = DecoderBlock(enc_ch["c5"], enc_ch["c4"], dec_out[0])  # 1/32 -> 1/16
        self.dec3 = DecoderBlock(dec_out[0], enc_ch["c3"], dec_out[1])    # 1/16 -> 1/8
        self.dec2 = DecoderBlock(dec_out[1], enc_ch["c2"], dec_out[2])    # 1/8  -> 1/4
        self.dec1 = DecoderBlock(dec_out[2], enc_ch["c1"], dec_out[3])    # 1/4  -> 1/2

        # Shared trunk processing at 1/2 resolution
        final_in = dec_out[3]
        self.final_up = nn.Sequential(
            ConvBNAct(final_in, final_in, k=3, groups=final_in),
            ConvBNAct(final_in, self.TRUNK_CH, k=1),
            ConvBNAct(self.TRUNK_CH, self.TRUNK_CH, k=3),
        )
        self.eca = EfficientChannelAttention(kernel_size=3)

        # Keep semantic heads simple and give geometric tasks a small nonlinear
        # projection. This preserves the proven MobileNet trunk while reducing
        # direct competition among mask/confidence and contour supervision.
        self.head_mask = nn.Conv2d(self.TRUNK_CH, 1, kernel_size=1)
        self.head_conf = nn.Conv2d(self.TRUNK_CH, 1, kernel_size=1)
        self.geometry_tower = GeometryTower(
            feature_channels=self.TRUNK_CH,
            hidden_channels=self.GEOMETRY_CH,
        )
        # Geometry fusion that slightly improves thin and narrow metrics
        self.tail_fuse = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        self.boundary_fuse = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def _forward_half(self, x):
        c1, c2, c3, c4, c5 = self.encoder(x)

        c5 = self.bottleneck(c5)
        d4 = self.dec4(c5, c4)  # 1/16
        d3 = self.dec3(d4, c3)  # 1/8
        d2 = self.dec2(d3, c2)  # 1/4
        d1 = self.dec1(d2, c1)  # 1/2

        feat = self.eca(self.final_up(d1))

        geometry_half = self.geometry_tower(feat)
        boundary_half = geometry_half[:, 0:1]
        tail_half = geometry_half[:, 1:2]
        distance_half = geometry_half[:, 2:3]

        boundary_fuse = self.boundary_fuse(boundary_half)
        tail_fuse = self.tail_fuse(tail_half)
        mask_half = self.head_mask(feat - boundary_fuse + tail_fuse)

        return (
            mask_half,
            boundary_half,
            tail_half,
            distance_half,
            self.head_conf(feat),
        )

    @staticmethod
    def _upsample(tensor, height, width):
        return F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    def forward_export(self, x):
        """Run the fixed three-channel deployment graph.

        Public logits are concatenated before their single full-resolution
        resize. Tail and distance remain internal half-resolution guides.
        """
        height, width = x.shape[-2:]
        (
            mask_half,
            boundary_half,
            _,
            _,
            confidence_half,
        ) = self._forward_half(x)
        return self._upsample(
            torch.cat(
                (mask_half, boundary_half, confidence_half),
                dim=1,
            ),
            height,
            width,
        )

    def forward(self, x):
        return self.forward_export(x)


def load_model(checkpoint_path: Path) -> tuple[BubbleUNet, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not a supported training checkpoint "
            "(missing model_state_dict)"
        )
    variant = checkpoint.get("variant", "large")
    model = BubbleUNet(variant=variant, pretrained=False)
    architecture_version = checkpoint.get("architecture_version")
    if architecture_version != model.ARCHITECTURE_VERSION:
        raise ValueError(
            "Checkpoint architecture mismatch: "
            f"{architecture_version!r} != {model.ARCHITECTURE_VERSION!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint
