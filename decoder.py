import torch
import torch.nn as nn
import torch.nn.functional as F
from helper import (ResidualBlock, 
                    NonLocalBlock,
                    UpSampleBlock)


class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
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



if __name__ == "__main__":
    import argparse

    args = argparse.Namespace(
    latent_dim=256,
    image_size=256,
    num_codebook_vectors=1024,
    beta=0.25,
    img_channels=3,
    dataset_path=r"C:\Users\dome\datasets\flowers",
    checkpoint_path=r".\checkpoints\vqgan_last_ckpt.pt",
    device="cuda",
    batch_size=20,
    epochs=100,
    learning_rate=2.25e-05,
    beta1=0.5,
    beta2=0.9,
    disc_start=10000,
    disc_factor=1.0,
    l2_loss_factor=1.0,
    perceptual_loss_factor=1.0,
    pkeep=0.5,
    sos_token=0
)
    decoder = Decoder(args)
    x = torch.rand(1, 3, 256, 256)
    print(decoder(x).shape)
