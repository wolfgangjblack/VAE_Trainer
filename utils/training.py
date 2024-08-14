import os
import math
import json
import torch
import logging
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torchvision.utils import save_image

def setup_logging(model, config):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/run_{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, "training.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Log model architecture
    logging.info(f"Model Architecture:\n{model}")

    # Log training configuration
    logging.info(f"Training Configuration:\n{json.dumps(config, indent=2)}")

    return log_dir

def log_epoch(epoch, avg_loss, avg_kl_div, avg_reconstruct_loss, kld_weight):
    logging.info(f"Epoch {epoch} | "
                 f"Avg Loss: {avg_loss:.4f} | "
                 f"Avg KL Div: {avg_kl_div:.4f} | "
                 f"Avg Reconstruct Loss: {avg_reconstruct_loss:.4f} | "
                 f"KLD weight: {kld_weight:.4f}")

def log_generation(output_dir, num_examples):
    logging.info(f"Generated {num_examples} images. Saved in {output_dir}")

def cyclical_annealing_schedule(epoch, 
                                cycle_length = 10):
    
    cycle_progress = (epoch % cycle_length) / cycle_length
    
    sine_value = math.sin(math.pi*cycle_progress)
    return abs(sine_value)

def mse_vae_loss(recon_x,
             x,
             mu,
             logvar,
             kld_weight=1.0):
    
    batch_size = x.size(0)
    MSE = F.mse_loss(recon_x, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return MSE + kld_weight * KLD, MSE, KLD

def bce_vae_loss(recon_x, x, mu, logvar, kld_weight=1.0):
    batch_size = x.size(0)
    x = x.view(batch_size, -1)  # Flatten the input
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum') / batch_size
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    return BCE + kld_weight * KLD, BCE, KLD

def train_vae(model,
              train_loader,
              NUM_EPOCHS,
              DEVICE,
              optimizer,
              model_path = None,
              **kwargs):
 
    if model_path is None:
        model_path = os.path.join(model_path, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}", model.__class__.__name__)
 
    ##Set up log
    log_dir = setup_logging(model, kwargs)
    
    ##Set default values
    kld_weight = kwargs.get('kld_weight', 1.0)
    use_cyclical_annealing = kwargs.get('use_cyclical_annealing', False)
    resnet = kwargs.get('resnet', False)
    
    if use_cyclical_annealing:
        cycle_length = kwargs.get('cycle_length', 10)
    
    for epoch in range(NUM_EPOCHS):

        epoch_loss = 0
        epoch_kl_div = 0
        epoch_reconstuct_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True) 
        loop.set_description(f'Epoch {epoch}')
        
        if use_cyclical_annealing:
            kld_weight = cyclical_annealing_schedule(epoch, cycle_length)
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            if resnet:
                loss, reconst_loss, kl_div = mse_vae_loss(recon_batch, data, mu, logvar, kld_weight)
            else:
                loss, reconst_loss, kl_div = bce_vae_loss(recon_batch, data, mu, logvar, kld_weight)
                
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reconstuct_loss += reconst_loss.item()
            epoch_kl_div += kl_div.item()
            
            loop.set_postfix(loss=f"{loss.item():.2f}", reconst_loss=f"{reconst_loss.item():.2f}", kl_div=f"{kl_div.item():.2f}")
        
        loop.close()
        
        log_epoch(epoch, epoch_loss / len(train_loader.dataset), epoch_kl_div / len(train_loader.dataset), epoch_reconstuct_loss / len(train_loader.dataset), kld_weight)
        print(f''' Epoch {epoch} | Avg Loss: {epoch_loss / len(train_loader.dataset):.4f} | Avg KL Div: {epoch_kl_div / len(train_loader.dataset):.4f}|Avg Reconstruct Loss: {epoch_reconstuct_loss / len(train_loader.dataset):.4f} | KLD weight {kld_weight:.4f} \n{'-'*50}''')
        
        if (epoch + 1) % 10 == 0:
            model_path = os.path.join(model_path, f"epoch_{epoch+1}.safetensors")
            model.save(model_path)
            logging.info(f"Model saved at {model_path}")
    
    #Save final model
    model_path = os.path.join(model_path, f"final_model.safetensors")
    model.save(model_path)
    logging.info(f"Final Model saved at {model_path}")
    
    return log_dir