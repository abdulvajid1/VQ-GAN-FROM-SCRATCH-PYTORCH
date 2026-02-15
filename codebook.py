import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0/self.num_codebook_vectors, 1.0/self.num_codebook_vectors)
    
    def forward(self, z): # the coming latent vector, we will map to close codebook vector
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flatten = z.view(-1, self.latent_dim) # (256, 16, 16) -> (16, 16, 256) -> (256, 256) 
        # efficient way of euclidian distance (a-b)^2 == a**2 + b**2 - 2ab

        e_distances = torch.sum(z_flatten**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
                (2 * (torch.matmul(z_flatten, self.embedding.weight.T)))
        
        # for each z, taking its closest embedding
        dist_indices = torch.argmin(e_distances, dim=1)
        zq = self.embedding(dist_indices).view(z.shape) # same shape as image

        # loss
        loss = F.mse_loss(zq, z.detach()) - self.beta * F.mse_loss(z, zq.detach())
        zq = z + (zq - z).detach()
        zq = zq.permute(0, 3, 1, 2)

        return zq, dist_indices, loss
    

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
    code = Codebook(args)
    z = torch.rand(2, 256, 16, 16)
    print(code(z)[0].shape)