import os
import torch
from utils.model import ResNetVAE
from utils.data import plot_save_image
from torchvision import transforms
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils.training import train_vae, log_generation

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 400
BATCH_SIZE = 64 ##Make this smaller to add more variation
LR_RATE = 1e-4 #Karpathy Constant 3e-4

#Dataset - CIFAR100
transform = transforms.Compose([
    transforms.ToTensor()
    ])
dataset = datasets.CIFAR100(root="dataset/", train=True, transform=transform, download=True)


### Standard VAE 
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = ResNetVAE().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

# Training configuration
train_config = {
    'BATCH_SIZE': BATCH_SIZE,
    'LR_RATE': LR_RATE,
    'use_cyclical_annealing': True,
    'cycle_length': 25,
    'device': str(DEVICE),
    'early_stopping': False,
    'early_stopping_patience': 15,
    'early_stopping_threshold': 0.00001,
    'save_frequency': 50
}

train_config['output_dir'] = f"output/{dataset.__class__.__name__}_{NUM_EPOCHS}_{model.__class__.__name__}_annealing_{train_config['use_cyclical_annealing']}_earlystop_{train_config['early_stopping']}"

os.makedirs(train_config['output_dir'], exist_ok=True)

print(f'Training for {NUM_EPOCHS} Epochs')
log_dir = train_vae(model, train_loader, NUM_EPOCHS, DEVICE, optimizer, **train_config)

model.eval()
with torch.no_grad():
    for i in range(10):
        
        single_image = dataset[i][0]
        
        plot_save_image(single_image.permute(1, 2, 0),
                f"{train_config['output_dir']}/original_{dataset.__class__.__name__}_ex{i}.png")
        
        numbers = model.generate(single_image,
                    num_examples=5)
        
        for j, number in enumerate(numbers):
            
            plot_save_image(number.permute(1, 2, 0).detach().numpy(), 
                f"{train_config['output_dir']}/generated_ex{i}_{j}.png")
            
        log_generation(train_config['output_dir'], i+1)

print(f"Training completed. Logs and outputs saved in {log_dir}")
