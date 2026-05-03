import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # Ensure spatial dimensions match
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(self.W_g(g) + self.W_x(x))
        return x * self.psi(psi)


# ===========================
# HOW TO USE IN NULITE MODEL
# ===========================

# ADD IN __init__:
# self.attention_gate = AttentionGate(
#     F_g=self.embed_dims[-4],
#     F_l=self.embed_dims[-4],
#     F_int=self.embed_dims[-4] // 2
# )


# REPLACE THIS BLOCK IN forward():

"""
decoder = self._forward_upsample(z1, z2, z3, z4, self.decoder)

xt = self.decoder0(x)

# FIX: use encoder skip (z1) instead of decoder
decoder = self.attention_gate(g=xt, x=z1)

xt = torch.cat([xt, decoder], dim=1)
"""