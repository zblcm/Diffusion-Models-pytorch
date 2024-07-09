
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..shared import *

# Expect x shape: (batch,frame,channel)
# Expect t shape: (batch,channel)
class SelfAttentionAndEmbedding(nn.Module):
    def __init__(self, size_chan, size_time):
        super(SelfAttentionAndEmbedding, self).__init__()
        self.layer_attention = SelfAttention(size_chan)
        self.layer_embed_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size_time, size_chan),
        )

    def forward(self, x, time):
        time = self.layer_embed_time(time)[:,None,:].repeat(1, x.shape[1], 1)
        return x + self.layer_attention(x + time)

class TinyNet(nn.Module):
    def __init__(self, siez_chan_I=93, size_chan_O=93, size_chan_M=256, size_frame=64, size_time=256, device="cuda"):
        super().__init__()
        self.run_name = "tinynet_20240617_single_pos"
        self.device = device
        self.size_time = size_time

        self.layer_I = nn.Sequential(
            nn.Linear(siez_chan_I, size_chan_M),
            GroupNormChannelLast(1, size_chan_M),
            nn.GELU(),
            nn.Linear(size_chan_M, size_chan_M),
            GroupNormChannelLast(1, size_chan_M),
        )
        self.layer_O = nn.Sequential(
            nn.Linear(size_chan_M, size_chan_M),
            GroupNormChannelLast(1, size_chan_M),
            nn.GELU(),
            nn.Linear(size_chan_M, size_chan_O),
            GroupNormChannelLast(1, size_chan_O),
        )
        self.layers = nn.ModuleList([
            SelfAttentionAndEmbedding(size_chan_M, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_time),
        ])
        self.pos_enc = nn.Parameter(torch.zeros(1, size_frame, size_chan_M))

    def encode_time(self, t, size_time):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, size_time, 2, device=self.device).float() / size_time)) # (1 ~ 1/10000)
        time_enc_sin = torch.sin(t.repeat(1, size_time // 2) * inv_freq)
        time_enc_cos = torch.cos(t.repeat(1, size_time // 2) * inv_freq)
        time_enc = torch.cat([time_enc_sin, time_enc_cos], dim=-1)
        return time_enc
    
    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encode_time(t, self.size_time)

        shape_I = x.shape
        x = x.view(shape_I[0], shape_I[1], shape_I[2] * shape_I[3])
        x = self.layer_I(x)
        x = x + self.pos_enc
        for layer in self.layers:
            x = layer(x, t)
        x = self.layer_O(x)
        x = x.view(shape_I[0], shape_I[1], x.shape[2] // shape_I[3], shape_I[3])

        return x
