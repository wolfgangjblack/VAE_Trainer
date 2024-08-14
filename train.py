import torch
from utils.model import ResNetVAE
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils.training import train_vae, log_generation

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
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
    'use_early_stopping': True,
    'early_stopping_patience': 15,
    'early_stopping_threshold': 0.001,
}

train_config['output_dir'] = f"output/{dataset.__class__.__name__}_{NUM_EPOCHS}_{model.__class__.__name__}_annealing_{train_config['use_cyclical_annealing']}_earlystop_{train_config['use_early_stopping']}"

print(f'Training for {NUM_EPOCHS} Epochs')
log_dir = train_vae(model, train_loader, NUM_EPOCHS, DEVICE, optimizer, **train_config)

model.eval()
with torch.no_grad():
    for i in range(10):
        single_image = dataset[i][0]
        single_image
        model.generate(single_image,
                    num_examples=5,
                    output_dir=f"{train_config['output_dir']}/cls_{i}")
        log_generation(train_config['output_dir'], i+1)

print(f"Training completed. Logs and outputs saved in {log_dir}")
