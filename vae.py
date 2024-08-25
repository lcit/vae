import torch
from torch import nn

from cnn import CNN
        
class VAE(nn.Module):
    
    def __init__(self, encoder, decoder, alpha=0.5):
        super().__init__()
        self.encoder = CNN(encoder)
        self.decoder = CNN(decoder)
        self.alpha = alpha
        
    def forward(self, x):

        mean, log_variance = self.encoder(x)
        
        variance = torch.exp(log_variance)
        std = torch.sqrt(variance)
        
        kl_regularization = torch.sum(0.5*(mean**2 + variance - log_variance -1))
        
        z = mean + std*torch.randn_like(std) # Reparameterization trick
        x_hat = self.decoder(z)
        
        reconstruction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        
        return {"loss": reconstruction_loss + kl_regularization*self.alpha,
                "reconstruction_loss": reconstruction_loss,
                "kl_regularization": kl_regularization}
        