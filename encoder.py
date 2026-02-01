import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import (ResidualBlock, 
                    NonLocalBlock,
                    DownSampleBlock)


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        channels = [128, 128, 128, 128, 256, 256, 512]
        attn_res = [16] # img resolutions where we add attn block
        num_res_blocks = 2 # no. of res block add at each time
        curr_img_resolution = 256 # initial img resolution
        layers = []
        layers.append(nn.Conv2d(args.img_channels, channels[0], 3, 1, 1))
        
        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            for _ in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels))
                in_channels = out_channels
                if curr_img_resolution in attn_res:
                    layers.append(NonLocalBlock(in_channels))

            if i != len(channels)-2:
                layers.append(DownSampleBlock(in_channels=in_channels, out_channels=out_channels))
                curr_img_resolution //= 2

        layers.append(ResidualBlock(in_channels=channels[-1], out_channels=channels[-1]))
        layers.append(NonLocalBlock(channels=channels[-1]))
        layers.append(ResidualBlock(in_channels=channels[-1], out_channels=channels[-1]))
        layers.append(nn.GroupNorm(32, num_channels=channels[-1]))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, 3, 1, 1))

        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)
        
