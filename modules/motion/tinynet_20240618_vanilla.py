
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, size_chan):
        super(SelfAttention, self).__init__()
        self.size_chan = size_chan
        self.mha = nn.MultiheadAttention(size_chan, 4, batch_first=True)
        self.ln = nn.LayerNorm([size_chan])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([size_chan]),
            nn.Linear(size_chan, size_chan),
            nn.GELU(),
            nn.Linear(size_chan, size_chan),
        )

    def forward(self, x):
        x = x.swapaxes(-1, -2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(-1, -2)

# Expected shape_I: (size_batch, siez_chan_I, size_frame)
# Expected shape_O: (size_batch, siez_chan_O, size_frame)
class DoubleConv(nn.Module):
    def __init__(self, size_chan_I, size_chan_O, size_chan_M=None, residual=False):
        super().__init__()
        self.residual = residual
        if not size_chan_M:
            size_chan_M = size_chan_O
        self.double_conv = nn.Sequential(
            nn.Conv1d(size_chan_I, size_chan_M, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, size_chan_M),
            nn.GELU(),
            nn.Conv1d(size_chan_M, size_chan_O, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, size_chan_O),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

# Expected shape_I: (size_batch, siez_chan_I, size_frame)
# Expected shape_O: (size_batch, siez_chan_O, size_frame // 4)
class Down(nn.Module):
    def __init__(self, size_chan_I, size_chan_O, size_time=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv(size_chan_I, size_chan_I, residual=True),
            DoubleConv(size_chan_I, size_chan_O),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                size_time,
                size_chan_O
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, size_chan_I, size_chan_O, size_time=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(size_chan_I * 2, size_chan_I * 2, residual=True),
            DoubleConv(size_chan_I * 2, size_chan_O, size_chan_I),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                size_time,
                size_chan_O
            ),
        )

    def forward(self, x, skip_x, t):
        # https://discuss.pytorch.org/t/one-dimensional-upsampling/6263
        x = x.unsqueeze(dim=3)
        x = self.up(x)[:,:,:,0]
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb


class TinyNet(nn.Module):
    # Expected shape_I: (size_batch, size_frame, siez_chan_I // 3, 3)
    # Expected shape_O: (size_batch, size_frame, siez_chan_O // 3, 3)
    def __init__(self, siez_chan_I=57, size_chan_O=57, size_time=256, device="cuda", local=True):
        super().__init__()
        self.device = device
        self.size_time = size_time

        if local:
            self.run_name = "tinynet_20240618_vanilla_local"
            size_chan_M4 = 256
            size_chan_M3 = 256
            size_chan_M2 = 256
        else:
            self.run_name = "tinynet_20240618_vanilla_remote"
            size_chan_M4 = 64
            size_chan_M3 = 128
            size_chan_M2 = 256

        size_chan_MA = size_chan_M2
        size_chan_MB = size_chan_M4

        self.inc = DoubleConv(siez_chan_I, size_chan_M4)
        self.down43 = Down(size_chan_M4, size_chan_M3)
        self.down_sa3 = SelfAttention(size_chan_M3)
        self.down32 = Down(size_chan_M3, size_chan_M2)
        self.down_sa2 = SelfAttention(size_chan_M2)
        self.down2A = Down(size_chan_M2, size_chan_MA)
        self.down_saA = SelfAttention(size_chan_MA)

        self.bot1 = DoubleConv(size_chan_MA, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, size_chan_M2)

        self.up23 = Up(size_chan_M2, size_chan_M3)
        self.up_sa3 = SelfAttention(size_chan_M3)
        self.up34 = Up(size_chan_M3, size_chan_M4)
        self.up_sa4 = SelfAttention(size_chan_M4)
        self.up4B = Up(size_chan_M4, size_chan_MB)
        self.up_saB = SelfAttention(size_chan_MB)
        self.outc = nn.Conv1d(size_chan_MB, size_chan_O, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.size_time)
        
        shape_I = x.shape
        x = x.view(shape_I[0], shape_I[1], shape_I[2] * shape_I[3]).swapaxes(-1, -2)

        x4 = self.inc(x)
        x3 = self.down43(x4, t)
        x3 = self.down_sa3(x3)
        x2 = self.down32(x3, t)
        x2 = self.down_sa2(x2)
        xA = self.down2A(x2, t)
        xA = self.down_saA(xA)

        xA = self.bot1(xA)
        xA = self.bot2(xA)
        xA = self.bot3(xA)
        x = xA

        x = self.up23(x, x2, t)
        x = self.up_sa3(x)
        x = self.up34(x, x3, t)
        x = self.up_sa4(x)
        x = self.up4B(x, x4, t)
        x = self.up_saB(x)
        x = self.outc(x)

        x = x.swapaxes(-1, -2).view(shape_I)

        return x
