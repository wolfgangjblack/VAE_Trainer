import torch
import torchvision.datasets as datasets
from torch import nn
import torch.nn.functional as F
from utils.model import VariationalAutoEncoder
from utils.training import train_vae, inference
from torchvision import transforms
from torch.utils.data import DataLoader

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(DEVICE)

NUM_EPOCHS = 500
BATCH_SIZE = 256
LR_RATE = 1e-4 #Karpathy Constant 3e-4

#Dataset - MNIST
# dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
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

model = VariationalAutoEncoder(INPUT_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum") #reconstruction loss
# loss_fn = F.mse_loss #reconstruction loss
##KL loss is kullback-leibler divergence

print('Training')
train_vae(model, train_loader, NUM_EPOCHS, DEVICE, loss_fn, optimizer)

model = model.to("cpu")
outs = {}
print("Generating images")
for digit in range(10):
    inference(model, dataset, digit, num_examples=5, output_dir="output/cifar100_500epochs/")
    