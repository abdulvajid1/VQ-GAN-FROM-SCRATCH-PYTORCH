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
        decoded_img = self.decoder(post_quant_encoded)
        return decoded_img, codebook_indices, q_loss
    
    def encode(self, imgs):
        encoded_img = self.encoder(imgs)
        quant_encoded_img = self.pre_quant_layer(encoded_img)
        codebook_mapping, codebook_indieces, q_loss = self.codebook(quant_encoded_img)
        return codebook_mapping, codebook_indieces, q_loss
    
    def decode(self, z):
        post_quant = self.post_quant_layer(z)
        decoded_img = self.decode(post_quant)
        return decoded_img
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        λ = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))
