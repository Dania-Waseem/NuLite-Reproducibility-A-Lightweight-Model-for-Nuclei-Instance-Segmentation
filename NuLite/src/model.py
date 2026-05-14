"""
my_model.py
NuLite model using FastViT-T8 encoder from timm library.
UNet decoder with 3 output heads exactly as in NuLite paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─── Attention Gate (used in Experiment 2) ────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Attention Gate as described in NuLite paper Experiment 2.
    Based on Attention U-Net (Oktay et al., 2018).
    alpha = sigmoid(BN(Wpsi * ReLU(BN(Wg*g) + BN(Wx*x))))
    x_tilde = alpha * x
    """
    def __init__(self, g_channels, x_channels, inter_channels):
        super().__init__()
        self.Wg  = nn.Sequential(
            nn.Conv2d(g_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.Wx  = nn.Sequential(
            nn.Conv2d(x_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        # g: gating signal (decoder), x: skip connection (encoder)
        g_up = F.interpolate(self.Wg(g), size=x.shape[2:], mode="bilinear",
                             align_corners=False)
        alpha = self.psi(F.relu(g_up + self.Wx(x)))
        return alpha * x


# ─── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── NuLite Model ─────────────────────────────────────────────────────────────

class NuLite(nn.Module):
    """
    NuLite-T architecture:
    - FastViT-T8 encoder (from timm)
    - UNet-style decoder with 5 upsampling stages
    - 3 output heads: binary map, HV map, type map
    As described in NuLite paper (Tommasino et al., 2026)
    use_attention_gate: set True for Experiment 2
    """
    def __init__(self, num_classes=6, use_attention_gate=False):
        super().__init__()
        self.use_attention_gate = use_attention_gate

        # FastViT-T8 encoder
        # Produces features at 4 scales:
        # Stage 1: channels=48,  stride=4  -> 64x64
        # Stage 2: channels=96,  stride=8  -> 32x32
        # Stage 3: channels=192, stride=16 -> 16x16
        # Stage 4: channels=384, stride=32 -> 8x8
        self.encoder = timm.create_model(
            "fastvit_t8",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )

        enc_chs = [48, 96, 192, 384]

        # Decoder: bottom-up, 4 decode blocks
        self.decoder3 = DecoderBlock(enc_chs[3], enc_chs[2], 256)
        self.decoder2 = DecoderBlock(256,         enc_chs[1], 128)
        self.decoder1 = DecoderBlock(128,         enc_chs[0], 64)
        self.decoder0 = DecoderBlock(64,          3,          32)
        # decoder0 uses 3-channel input image as skip

        # Attention gate for Experiment 2
        # Applied at decoder0 skip connection (highest resolution)
        if use_attention_gate:
            self.attn_gate = AttentionGate(
                g_channels=64, x_channels=3, inter_channels=16)

        # Output heads (all from 32 channels at full resolution)
        self.head_binary = nn.Conv2d(32, 1, 1)           # binary map
        self.head_hv     = nn.Conv2d(32, 2, 1)           # HV distance maps
        self.head_type   = nn.Conv2d(32, num_classes, 1) # type map

    def forward(self, x):
        # x: [B, 3, 256, 256]
        img = x  # keep for skip connection in decoder0

        # Encoder (FastViT-T8)
        features = self.encoder(x)
        z1, z2, z3, z4 = features  # 4 scales

        # Decoder
        d3 = self.decoder3(z4, z3)
        d2 = self.decoder2(d3, z2)
        d1 = self.decoder1(d2, z1)

        # Attention gate on final skip connection
        if self.use_attention_gate:
            img_skip = self.attn_gate(g=d1, x=img)
        else:
            img_skip = img

        d0 = self.decoder0(d1, img_skip)

        # Output heads
        out_binary = self.head_binary(d0)  # [B, 1, 256, 256]
        out_hv     = self.head_hv(d0)      # [B, 2, 256, 256]
        out_type   = self.head_type(d0)    # [B, 6, 256, 256]

        return out_binary, out_hv, out_type