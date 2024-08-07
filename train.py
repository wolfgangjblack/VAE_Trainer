import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn
from utils.model import VariationalAutoEncoder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28*28
H_DIM = 200
Z_DIM = 20

NUM_EPOCHS = 10
BATCH_SIZE = 32
LR_RATE = 3e-4 #Karpathy Constant

#Dataset
dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
##Captial T in ToTensor divides by 255

train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.BCELoss(reduction="sum") #reconstruction loss
##KL loss is kullback-leibler divergence

for epoch in range(NUM_EPOCHS):
    loop = tqdm((enumerate(train_loader)), total=len(train_loader), leave=False)
    for batch_idx, (x, _) in loop:
        x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
        x_reconst, mu, log_var = model(x)
        
        ## Compute Loss
        reconst_loss = loss_fn(x_reconst, x)
       