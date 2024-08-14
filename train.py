import torch
from torchvision import transforms
from utils.training import train_vae
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils.model import ResNetVAE

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
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
    'NUM_EPOCHS': NUM_EPOCHS,
    'BATCH_SIZE': BATCH_SIZE,
    'LR_RATE': LR_RATE,
    'use_cyclical_annealing': True,
    'cycle_length': 25,
    'device': str(DEVICE)
}

print('Training')
log_dir = train_vae(model, train_loader, NUM_EPOCHS, DEVICE, optimizer, **train_config)

model = model.to("cpu")

for i in range(10):
    single_image = dataset[i][0]
    model.generate(single_image,
                   num_examples=5,
                   output_dir=f"output/cifar100_100epochs_resnetvae_cyclic/cls_{i}")
