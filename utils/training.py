import os
import math
import json
import torch
import logging
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torchvision.utils import save_image

def setup_logging(model, current_time, config):

    log_dir = config.get(os.path.join("logs", current_time),os.path.join('./logs/', current_time, model.__class__.__name__))

    log_dir = f"logs/r{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, "training.log")
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Log model architecture
    logging.info(f"Model Architecture for {model.__class__.__name__}:\n{model}\n")

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
    
    start_time = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}"
    model_path = kwargs.get("model_path",os.path.join('./models/', start_time, model.__class__.__name__))
 
    ##Set up logging - this keeps track of our model training
    log_dir = setup_logging(model, start_time, kwargs)
    
    ##Set default values for kl divergence weight and cyclical annealing
    kld_weight = kwargs.get('kld_weight', 1.0) #Do not go above 1.0    
    use_cyclical_annealing = kwargs.get('use_cyclical_annealing', False)
    cycle_length = kwargs.get('cycle_length', 10) #Typically set at 25
    
    ##if model is resnet type
    if 'resnet' in model.__class__.__name__.lower():
        loss_fn = 'mse'
    else:
        loss_fn = 'bce'
    
    ##Early Stopping Parameters
    early_stopping = kwargs.get('early_stopping', False)
    early_stopping_patience = kwargs.get('early_stopping_patience', 5)
    early_stopping_threshold = kwargs.get('early_stopping_threshold', 0.001) 
    
    ## Log some information
    logging.info(f"Training for {NUM_EPOCHS} Epochs")
    logging.info(f"Training on Dataset: {train_loader.dataset.__class__.__name__}")
    logging.info(f'Dataset has {len(train_loader.dataset)} samples of size {train_loader.dataset[0][0].shape}\n')
    
    
    best_loss = float('inf')
    patience = 0
    
    for epoch in range(NUM_EPOCHS):

        epoch_loss = 0
        epoch_kl_div = 0
        epoch_reconstruct_loss = 0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=True) 
        loop.set_description(f'Epoch {epoch}')
        
        if use_cyclical_annealing:
            kld_weight = cyclical_annealing_schedule(epoch, cycle_length)
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            
            ## If we are using resnet model, we use MSE loss, otherwise we use BCE loss. BCE Loss is for really simple models
            if loss_fn == 'mse':
                loss, reconst_loss, kl_div = mse_vae_loss(recon_batch, data, mu, logvar, kld_weight)
            else:
                loss, reconst_loss, kl_div = bce_vae_loss(recon_batch, data, mu, logvar, kld_weight)
                
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_reconstruct_loss += reconst_loss.item()
            epoch_kl_div += kl_div.item()
            
            loop.set_postfix(loss=f"{loss.item():.2f}", reconst_loss=f"{reconst_loss.item():.2f}", kl_div=f"{kl_div.item():.2f}")
        
        loop.close()
        
        ##Calc average loss
        avg_loss = epoch_loss / len(train_loader.dataset)
        avg_kl_div = epoch_kl_div / len(train_loader.dataset)
        avg_reconstruct_loss = epoch_reconstruct_loss / len(train_loader.dataset)
        
        log_epoch(epoch, avg_loss, avg_kl_div, avg_reconstruct_loss , kld_weight)
        print(f''' Epoch {epoch} | Avg Loss: {avg_loss:.4f} | Avg KL Div: {avg_kl_div:.4f}|Avg Reconstruct Loss: {avg_reconstruct_loss:.4f} | KLD weight {kld_weight:.4f} \n{'-'*50}''')
        
        if (epoch + 1) % 10 == 0:
            model.save(os.path.join(model_path, f"epoch_{epoch+1}.safetensors"))
            logging.info(f"Model saved at {os.path.join(model_path, f"epoch_{epoch+1}.safetensors")}")
        
        ## Early stopping    
        if early_stopping:
            if avg_loss < best_loss - early_stopping_threshold:
                best_loss = avg_loss
                patience = 0
                best_model_path = os.path.join(model_path, f"best_model.safetensors")
                model.save(best_model_path)
                logging.info(f"Best Model saved at {best_model_path}")
            else:
                patience += 1
        if patience > early_stopping_patience:
            early_stop_model_path = os.path.join(model_path, f"final_model.safetensors")
            model.save(early_stop_model_path)
            logging.info(f"Early Stopping at Epoch {epoch}. Final Model saved at {early_stop_model_path}")
            logging.info(f"Final Model Scores: Avg Loss: {avg_loss:.4f} | Avg KL Div: {avg_kl_div:.4f} | Avg Reconstruct Loss: {avg_reconstruct_loss:.4f}")
            print(f"Early Stopping at Epoch {epoch}. Final Model saved at {early_stop_model_path}")
            break
                
    #Save final model
    model_path = os.path.join(model_path, f"final_model.safetensors")
    model.save(model_path)
    logging.info(f"Final Model saved at {model_path}")
    
    return log_dir