import torch.nn as nn
import torch
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook


class VQGan(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.codebook = Codebook(args)
        self.pre_quant_layer = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
        self.post_quant_layer = nn.Conv2d(args.latent_dim, args.latent_dim, 1)
    
    def forward(self, x):
        encoded_img = self.encoder(x)
        encoded_quant = self.pre_quant_layer(encoded_img)
        codebook_mapping, codebook_indices, q_loss = self.codebook(encoded_quant)
        post_quant_encoded = self.post_quant_layer(codebook_mapping)
        decoded_img = Decoder(post_quant_encoded)

        return decoded_img
    
    def encode(self, imgs):
        encoded_img = self.encoder(imgs)
        return self.pre_quant_layer(encoded_img)
    
    def decode(self, z):
        post_quant = self.post_quant_layer(z)
        decoded_img = self.decode(post_quant)
        return decoded_img
    
    def calculate_lambda(self, perceptual_loss, gen_loss):
        last_layer_weight = self.decoder.model[-1].weight
