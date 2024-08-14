import os
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image

def cyclical_annealing_schedule(epoch, 
                                cycle_length = 10):
    
    cycle_progress = (epoch % cycle_length) / cycle_length
    
    sine_value = math.sin(math.pi*cycle_progress)
    return abs(sine_value)

def vae_loss(recon_x,
             x,
             mu,
             logvar,
             kld_weight=1.0):
    
    batch_size = x.size(0)
    MSE = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return MSE + kld_weight * KLD, MSE, KLD

def train_vae(model,
              train_loader,
              NUM_EPOCHS,
              DEVICE,
              optimizer,
              **kwargs):
    
    ##Set default values
    kld_weight = kwargs.get('kld_weight', 1.0)
    use_cyclical_annealing = kwargs.get('use_cyclical_annealing', False)
    
    if use_cyclical_annealing:
        cycle_length = kwargs.get('cycle_length', 10)
        min_value = kwargs.get('min_value', 0.0)
        max_value = kwargs.get('max_value', 1.0)
    
    for epoch in range(NUM_EPOCHS):

        epoch_loss = 0
        epoch_kl_div = 0
        epoch_reconstuct_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True) 
        loop.set_description(f'Epoch {epoch}')
        
        if use_cyclical_annealing:
            kld_weight = cyclical_annealing_schedule(epoch, cycle_length, min_value, max_value)
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, reconst_loss, kl_div = vae_loss(recon_batch, data, mu, logvar, kld_weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reconstuct_loss += reconst_loss.item()
            epoch_kl_div += kl_div.item()
            
            loop.set_postfix(loss=f"{loss.item():.2f}", reconst_loss=f"{reconst_loss.item():.2f}", kl_div=f"{kl_div.item():.2f}")
        
        print(f''' Epoch {epoch} | Avg Loss: {epoch_loss / len(train_loader.dataset):.4f} | Avg KL Div: {epoch_kl_div / len(train_loader.dataset):.4f}|Avg Reconstruct Loss: {epoch_reconstuct_loss / len(train_loader.dataset):.4f} | KLD weight {kld_weight} \n{'-'*50}''')

def inference(model,
              dataset,
              digit,
              num_examples = 1,
              output_dir = "outputs/"):
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
   
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma*epsilon
        out = model.decode(z)
        out = out.view(-1, ch, w, h)
        save_image(out, f"{output_dir}/generated_{digit}_ex{example}.png")
    