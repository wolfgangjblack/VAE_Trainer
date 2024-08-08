import torch
import torchvision.datasets as datasets
from torch import nn
from utils.model import VariationalAutoEncoder
from utils.training import train_vae, inference
from torchvision import transforms
from torch.utils.data import DataLoader

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20

print(DEVICE)

NUM_EPOCHS = 40
BATCH_SIZE = 32
LR_RATE = 1e-4 #Karpathy Constant 3e-4

#Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
##Captial T in ToTensor divides by 255

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum") #reconstruction loss
##KL loss is kullback-leibler divergence

train_vae(model, train_loader, NUM_EPOCHS, DEVICE, loss_fn, optimizer)

model = model.to("cpu")
outs = {}
print("Generating images")
for digit in range(10):
    inference(model, dataset, digit, num_examples=5)
    