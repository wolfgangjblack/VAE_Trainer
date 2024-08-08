import torch
import torch.nn.functional as F
from torch import nn

#Input img -> hidden dim -> mean, std, -> parametrization trick -> Decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim = 200, z_dim = 20):
        super().__init__()
        
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        
        ##Encoder
        self.img_2hid = nn.Linear(input_dim, h_dim)
        ##In the loss function we'll specify these with the KL divergence to try to push this towards a standard gaussian
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        
        ##Decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, input_dim)
        
        self.relu = nn.ReLU(inplace=True)
            
    def encode(self, x):
        #q_phi(z|x)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self, z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        
        #return between 0 and 1
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self, x):
        mu, sigma = self.encode(x)
        
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        x_reconstructed = self.decode(z_reparametrized)
        
        ##return mu, sigma for the KL divergence loss function and x for the reconstruction loss
        return x_reconstructed, mu, sigma
        
if __name__ == "__main__":
    x = torch.randn(4, 28*28) #28 x 28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape, mu.shape, sigma.shape)