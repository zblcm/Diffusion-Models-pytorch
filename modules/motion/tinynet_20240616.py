
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..shared import *

# Expect x shape: (batch,frame,channel)
# Expect t shape: (batch,channel)
class SelfAttentionAndEmbedding(nn.Module):
    def __init__(self, size_chan, size_pos, size_time):
        super(SelfAttentionAndEmbedding, self).__init__()
        self.layer_attention = SelfAttention(size_chan)
        self.layer_embed_time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size_time, size_chan),
        )
        self.layer_embed_pos = nn.Sequential(
            nn.SiLU(),
            nn.Linear(size_pos, size_chan),
        )

    def forward(self, x, pos, time):
        pos = self.layer_embed_pos(pos)[None,:,:].repeat(x.shape[0], 1, 1)
        time = self.layer_embed_time(time)[:,None,:].repeat(1, x.shape[1], 1)
        return x + self.layer_attention(x + pos + time)
        # return x + self.layer_attention(x + time)

class TinyNet(nn.Module):
    def __init__(self, siez_chan_I=93, size_chan_O=93, size_chan_M=256, size_pos=256, size_time=256, device="cuda"):
        super().__init__()
        self.run_name = "tinynet_20240616"
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
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
            SelfAttentionAndEmbedding(size_chan_M, size_pos, size_time),
        ])
        self.pos_freqs = self.encode_pos_freqs(size_pos)

    def encode_time(self, t, size_time):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, size_time, 2, device=self.device).float() / size_time)) # (1 ~ 1/10000)
        time_enc_sin = torch.sin(t.repeat(1, size_time // 2) * inv_freq)
        time_enc_cos = torch.cos(t.repeat(1, size_time // 2) * inv_freq)
        time_enc = torch.cat([time_enc_sin, time_enc_cos], dim=-1)
        return time_enc

    def encode_pos_freqs(self, size_pos):
        PI = torch.acos(torch.zeros(1)).item() * 2 # https://discuss.pytorch.org/t/np-pi-equivalent-in-pytorch/67157
        size_pos_2 = size_pos // 2

        nums = torch.arange(size_pos_2, device=self.device).float() / (size_pos_2 - 1) # 0~1
        num_max = math.log(10000)
        num_min = math.log(2)
        nums = num_min + nums * (num_max - num_min) # log(2) ~ log(10000)
        nums = torch.exp(nums) # 2 ~ 10000
        nums = PI * 2 / nums # 2pi/2 ~ 2pi/10000

        return nums

    def encode_pos(self, size_frame):
        size_pos_2 = self.pos_freqs.shape[-1]
        nums = torch.arange(size_frame, device=self.device) # 0 ~ size_frame-1
        nums = nums[:,None].repeat(1, size_pos_2) * self.pos_freqs[None,:].repeat(size_frame, 1)
        pos_enc_sin = torch.sin(nums)
        pos_enc_cos = torch.cos(nums)
        pos_enc = torch.cat([pos_enc_sin, pos_enc_cos], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.encode_time(t, self.size_time)
        p = self.encode_pos(x.shape[1])

        shape_I = x.shape
        x = x.view(shape_I[0], shape_I[1], shape_I[2] * shape_I[3])
        x = self.layer_I(x)
        for layer in self.layers:
            x = layer(x, p, t)
        x = self.layer_O(x)
        x = x.view(shape_I[0], shape_I[1], x.shape[2] // shape_I[3], shape_I[3])

        return x
