import torch
from torchvision import transforms
from utils.training import train_vae
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from utils.model import VariationalAutoEncoder, ResNetVAE

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 100
BATCH_SIZE = 64 ##Make this smaller to add more variation
LR_RATE = 1e-4 #Karpathy Constant 3e-4


## Datsets  ---------------------------------
#Dataset - MNIST
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)

#Dataset - MNIST Fashion
# dataset = datasets.FashionMNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
##Captial T in ToTensor divides by 255


#Dataset - CIFAR100
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.CIFAR100(root="dataset/", train=True, transform=transform, download=True)


### Standard VAE 
ch, w, h = dataset[0][0].shape
INPUT_DIM = ch * w * h
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model = VariationalAutoEncoder(INPUT_DIM).to(DEVICE)
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
single_image = dataset[0][0]
model.generate(single_image, num_examples=5, output_dir="output/cifar100_100epochs_resnetvae_cyclic/")

# outs = {}
# print("Generating images")
# for digit in range(10):
#     inference(model, dataset, digit, num_examples=5, output_dir="output/cifar100_500epochs/")
    