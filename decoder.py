import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import (ResidualBlock, 
                    NonLocalBlock,
                    UpSampleBlock)


class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels = [512, 256, 256, 128, 128]
        attn_resolution = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels=in_channels, out_channels=in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels=in_channels, out_channels=in_channels)
            ]
        
        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels=in_channels, out_channels=out_channels))
                in_channels = out_channels
                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(in_channels))
            
            if i!=0:
                layers.append(UpSampleBlock(in_channels, in_channels))
                resolution*=2

        layers.append(nn.GroupNorm(32, in_channels))
        layers.append(nn.SiLU())
        layers.append(nn.Conv2d(in_channels, args.img_channels, 3, 1, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)

