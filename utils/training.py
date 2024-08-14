import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image

def train_vae(model, train_loader, NUM_EPOCHS, DEVICE, loss_fn, optimizer):
    
    INPUT_DIM = model.input_dim
    
    for epoch in range(NUM_EPOCHS):
        # model.train()
        epoch_loss = 0
        epoch_reconst_loss = 0
        epoch_kl_div = 0
        loop = tqdm((enumerate(train_loader)), total=len(train_loader), leave=False)

        for batch_idx, (x, _) in loop:
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconst, mu, sigma = model(x)
            
            ## Compute Loss
            reconst_loss = loss_fn(x_reconst, x)/x.size(0) #reconstruction loss
            ##KL loss is kullback-leibler divergence
            
            kl_div = -0.5 * torch.sum(1+ torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))            
            #back prob
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update epoch losses
            epoch_loss += loss.item()
            epoch_reconst_loss += reconst_loss.item()
            epoch_kl_div += kl_div.item()
            
            
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(loss=loss.item(), reconst_loss=reconst_loss.item(), kl_div=kl_div.item())
          # Calculate average losses for the epoch
        
        avg_loss = loss / len(train_loader)
        avg_reconst_loss = epoch_reconst_loss / len(train_loader)
        avg_kl_div = epoch_kl_div / len(train_loader)
        
        # Print epoch summary
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Reconstruction Loss: {avg_reconst_loss:.4f}")
        print(f"Average KL Divergence: {avg_kl_div:.4f}")
        print("-" * 50)


def inference(model, dataset, digit, num_examples = 1, output_dir = "outputs/"):
    """
    Generates a number of examples of a given digit.
    Specifically we extract an example of each digit, 
    then after we have the mu, sigma representation for 
    each digit we can sample from the normal distribution.
    
    After we sample we can run the decoder part of the 
    VAE to generate a new image.
    Args:
        digit (_type_): _description_
        num_examples (int, optional): _description_. Defaults to 1.
    """
    
    ch, w, h = dataset[0][0].shape 
    
    images = []
    idx = 0
    for x,y in dataset:
        if y == idx:
            images.append(x)
            idx +=1
        if idx == 1000:
            break
        
    encodings_digit = []
    
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, model.input_dim))
        encodings_digit.append((mu, sigma))
        
    mu, sigma = encodings_digit[digit]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        out = model.decode(z)
        out = out.view(-1, ch, w, h)
        save_image(out, f"{output_dir}/generated_{digit}_ex{example}.png")
    