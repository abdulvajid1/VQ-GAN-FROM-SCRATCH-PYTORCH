import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.GroupNorm(num_groups=32, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        )

        if in_channels != out_channels:
            self.channel_proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1)
    
    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_proj(x) + self.block(x)
        else:
            return x + self.block(x)

class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0))
    
class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
    
    def forward(self, x):
        return self.conv(x)
    

class NonLocalBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.gn = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.o = nn.Conv2d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        b, c, h, w = x.shape
        hidden = self.gn(x)
        q = self.q(hidden).view(b, c, h*w).permute(0, 2, 1)
        k = self.k(hidden).view(b, c, h*w)
        v = self.v(hidden).view(b, c, h*w).permute(0, 2, 1)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**.5)
        attn_score = F.softmax(attn, dim=-1)
        attn_h = torch.bmm(attn_score, v)
        attn_h = attn_h.permute(0, 2, 1).contiguous().view(b, c, h, w)
        return x + self.o(attn_h)
    



if __name__ == "__main__":
    c = torch.rand(1, 64, 5, 5)
    block = NonLocalBlock(channels=64)
    print(block(c).shape)

