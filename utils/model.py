import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image

#Input img -> hidden dim -> mean, std, -> parametrization trick -> Decoder -> output img
class VariationalAutoEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 h_dim = 400,
                 z_dim = 100):
        
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
        x = x.view(-1, self.input_dim)
        mu, sigma = self.encode(x)
        
        # Reparameterization trick
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma * epsilon
        
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, sigma
    
    def generate(self, input, digit, num_examples=1, output_dir="outputs/"):
        os.makedirs(output_dir, exist_ok=True)
        ch, w, h = input.shape
        
        images = []
        
        with torch.no_grad():
            mu, sigma = self.encode(input.view(1, -1))
        
        for i in range(num_examples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = self.decode(z)
            out = out.view(ch, w, h)
            images.append(out)
            save_image(out, f"{output_dir}/generated_{digit}_ex{i}.png")
        
        return images
        
class ResidualBlock(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels):
        
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

class ResNetVAE(nn.Module):
    def __init__(self,
                 in_channels=3,
                 latent_dim=256):
        
        super(ResNetVAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 512),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 2 * 2)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
            # nn.Sigmoid() ##If we use BCE loss we need to use sigmoid, otherwise for MSE we don't need it
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 512, 2, 2)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def generate(self,
                 input_data,
                 num_examples,
                 output_dir = 'output/'):
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_data = input_data.to(next(self.parameters()).device)
        
        if input_data.dim() == 3:
            input_data = input_data.unsqueeze(0)
        
        with torch.no_grad():
            mu, logvar = self.encode(input_data)
        
        generated_images = []
        for i in range(num_examples):
            #sample from latent space
            z = self.reparameterize(mu, logvar)
            
            #Decode the sampled latent vector
            out = self.decode(z)
            
            #move back to cpu for plotting
            out = out.cpu()
            if out.size(0) == 1:
                out = out.squeeze(0)
                
            out = torch.clamp(out, 0, 1)
            generated_images.append(out)
                    
            # Save the generated image
            save_image(out, f"{output_dir}/generated_ex{i}.png")

        print("Generated images of shape: ", out.shape)
        return generated_images


if __name__ == "__main__":
    x = torch.randn(4, 28*28) #28 x 28 = 784
    vae = VariationalAutoEncoder(input_dim=784)
    x, mu, sigma = vae(x)
    print(x.shape, mu.shape, sigma.shape)