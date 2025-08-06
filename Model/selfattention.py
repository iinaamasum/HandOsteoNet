import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.qkv.weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).transpose(1, 2)
        q, k, v = self.qkv(x_flat).chunk(3, dim=-1)
        q = q.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = k.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, N, self.heads, C // self.heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-10, max=10)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.proj(out)
        out = out.transpose(1, 2).view(B, C, H, W)
        return x + out
