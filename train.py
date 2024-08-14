import torch
import torchvision.datasets as datasets
from torch import nn
import torch.nn.functional as F
from utils.model import VariationalAutoEncoder, ResNetVAE
from utils.training import train_vae
from torchvision import transforms
from torch.utils.data import DataLoader

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

NUM_EPOCHS = 100

##Make this smaller to add more variation
BATCH_SIZE = 64

LR_RATE = 1e-4 #Karpathy Constant 3e-4

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

ch, w, h = dataset[0][0].shape
INPUT_DIM = ch * w * h

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model = VariationalAutoEncoder(INPUT_DIM).to(DEVICE)
model = ResNetVAE().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)

print('Training')
anneal_config = {'use_cyclical_annealing': True, 'cycle_length': 25, 'resnet': True}
train_vae(model, train_loader, NUM_EPOCHS, DEVICE, optimizer, **anneal_config)

model = model.to("cpu")
single_image = dataset[0][0]
model.generate(single_image, num_examples=5, output_dir="output/cifar100_100epochs_resnetvae/")

# outs = {}
# print("Generating images")
# for digit in range(10):
#     inference(model, dataset, digit, num_examples=5, output_dir="output/cifar100_500epochs/")
    