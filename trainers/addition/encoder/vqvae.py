from torch import nn
import torch
import torch.nn.functional as F

class ResidualLayer(nn.Sequential):
    def __init__(self, n_channels, n_res_channels):
        super().__init__(nn.Conv2d(n_channels, n_res_channels, kernel_size=3, padding=1),
                         nn.ReLU(True),
                         nn.Conv2d(n_res_channels, n_channels, kernel_size=1))

    def forward(self, x):
        return F.relu(x + super().forward(x), True)

class VQ(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, ema=True, ema_decay=0.99, ema_eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.ema = ema
        self.ema_decay = ema_decay
        self.ema_eps = ema_eps

        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.kaiming_uniform_(self.embedding.weight, 1)

        if ema:
            self.embedding.weight.requires_grad_(False)
            self.register_buffer('ema_cluster_size', torch.zeros(n_embeddings))
            self.register_buffer('ema_weight', self.embedding.weight.clone().detach())

    def embed(self, encoding_indices):
        return self.embedding(encoding_indices).permute(0, 4, 1, 2, 3).squeeze(2)

    def forward(self, z):
        flat_z = z.permute(0,2,3,1).reshape(-1, self.embedding_dim)
        distances = flat_z.pow(2).sum(1, True) + self.embedding.weight.pow(2).sum(1) - 2 * flat_z.matmul(self.embedding.weight.t())
        encoding_indices = distances.argmin(1).reshape(z.shape[0], 1, *z.shape[2:])
        z_q = self.embed(encoding_indices)

        if self.ema and self.training:
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices.flatten(), self.n_embeddings).float().to(z.device)
                self.ema_cluster_size -= (1 - self.ema_decay) * (self.ema_cluster_size - encodings.sum(0))
                dw = z.permute(1, 0, 2, 3).flatten(1) @ encodings
                self.ema_weight -= (1 - self.ema_decay) * (self.ema_weight - dw.t())
                n = self.ema_cluster_size.sum()
                updated_cluster_size = (self.ema_cluster_size + self.ema_eps) / (n + self.n_embeddings * self.ema_eps) * n
                self.embedding.weight.data = self.ema_weight / updated_cluster_size.unsqueeze(1)

        return encoding_indices, z_q

class VQVAE2(nn.Module):
    def __init__(self, input_dims, n_embeddings, embedding_dim, n_channels, n_res_channels, n_res_layers,
                 ema=True, ema_decay=0.99, ema_eps=1e-5, **kwargs):
        super().__init__()
        self.ema = ema

        self.enc1 = nn.Sequential(nn.Conv2d(input_dims[0], n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.enc2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.Conv2d(n_channels//2, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.Conv2d(n_channels, embedding_dim, kernel_size=1))

        self.dec2 = nn.Sequential(nn.Conv2d(embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, embedding_dim, kernel_size=4, stride=2, padding=1))

        self.dec1 = nn.Sequential(nn.Conv2d(2*embedding_dim, n_channels, kernel_size=3, padding=1),
                                  nn.ReLU(True),
                                  nn.Sequential(*[ResidualLayer(n_channels, n_res_channels) for _ in range(n_res_layers)]),
                                  nn.ConvTranspose2d(n_channels, n_channels//2, kernel_size=4, stride=2, padding=1),
                                  nn.ReLU(True),
                                  nn.ConvTranspose2d(n_channels//2, input_dims[0], kernel_size=4, stride=2, padding=1))

        self.proj_to_vq1 = nn.Conv2d(2*embedding_dim, embedding_dim, kernel_size=1)
        self.upsample_to_dec1 = nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=4, stride=2, padding=1)

        self.vq1 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)
        self.vq2 = VQ(n_embeddings, embedding_dim, ema, ema_decay, ema_eps)

    def encode(self, x):
        z1 = self.enc1(x)
        z2 = self.enc2(z1)
        return (z1, z2)

    def embed(self, encoding_indices):
        encoding_indices1, encoding_indices2 = encoding_indices
        return (self.vq1.embed(encoding_indices1), self.vq2.embed(encoding_indices2))

    def quantize(self, z_e):
        z1, z2 = z_e

        encoding_indices2, zq2 = self.vq2(z2)

        quantized2 = z2 + (zq2 - z2).detach()
        dec2_out = self.dec2(quantized2)
        vq1_input = torch.cat([z1, dec2_out], 1)
        vq1_input = self.proj_to_vq1(vq1_input)
        encoding_indices1, zq1 = self.vq1(vq1_input)
        return (encoding_indices1, encoding_indices2), (zq1, zq2)

    def decode(self, z_e, z_q):
        zq1, zq2 = z_q
        if z_e is not None:
            z1, z2 = z_e
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()

        zq2_upsampled = self.upsample_to_dec1(zq2)
        combined_latents = torch.cat([zq1, zq2_upsampled], 1)
        return self.dec1(combined_latents)

    def forward(self, x, commitment_cost=None):
        z_e = self.encode(x)
        encoding_indices, z_q = self.quantize(z_e)

        return encoding_indices