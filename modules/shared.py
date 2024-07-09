import torch
import torch.nn as nn

# Expect shape: (batch,frame,channel)
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
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value
    
class GroupNormChannelLast(nn.Module):
    def __init__(self, count_group, size_chan):
        super(GroupNormChannelLast, self).__init__()
        self.norm = nn.GroupNorm(count_group, size_chan)

    def forward(self, x):
        x = torch.swapaxes(x, 1, -1)
        x = self.norm(x)
        x = torch.swapaxes(x, 1, -1)
        return x