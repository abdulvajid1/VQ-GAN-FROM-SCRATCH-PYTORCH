import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.num_vectors = args.num_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0/self.num_vectors, 1.0/self.num_vectors)
    
    def forward(self, z): # the coming latent vector, we will map to close codebook vector
        z_flatten = z.permute(0, 2, 3, 1).view(-1, self.latent_dim) # pixel space, where channels = latent_dim

        # efficient way of euclidian distance (a-b)^2 == a2 + b2 - 2ab
        e_distances = torch.sum(z_flatten**2, dim=-1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=-1) - \
                (2 * (torch.matmul(z_flatten, self.embedding.weight.T)))
        
        # for each z, taking its closest embedding
        dist_indices = torch.argmin(e_distances)
        zq = self.embedding(dist_indices).view(z.shape) # same shape as image

        # loss
        loss = F.mse_loss(zq, z.detach()) - self.beta * F.mse_loss(z, zq.detach())\
        
        zq = z + (zq - z).detach()
        zq = zq.permute(0, 3, 1, 2)

        return zq, dist_indices, loss