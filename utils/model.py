import torch
import torch.nn.functional as F
from torch import nn

#Input img -> hidden dim -> mean, std, -> parametrization trick -> Decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 400, z_dim = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        
          
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_sigma = nn.Linear(h_dim, z_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, input_dim),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_sigma(h)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, sigma = self.encode(x.view(-1, self.input_dim))
        
        # Reparameterization trick
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
        
if __name__ == "__main__":
    x = torch.randn(4, 28*28) #28 x 28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape, mu.shape, sigma.shape)