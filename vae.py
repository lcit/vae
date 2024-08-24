import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, normalization='batch', activation='relu'):
        super().__init__()
        
        self.conv = nn.Conv2d(input_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        
        if normalization=='batch':
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = None
        
        if activation=='relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x
        
class DownBlock(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, normalization='batch', activation='relu'):
        super().__init__()
        self.conv_block1 = ConvBlock(input_channels, hidden_channels, normalization, activation)
        self.conv_block2 = ConvBlock(hidden_channels, hidden_channels, normalization, activation)
        self.pooling = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.pooling(x)
        return x

class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.block1 = DownBlock(1, 32)
        self.block2 = DownBlock(32, 64)
        self.block3 = ConvBlock(64, 128, normalization='batch', activation='relu')
        self.mean_head = nn.Linear(128*7*7, 128*7*7)
        self.log_var_head = nn.Linear(128*7*7, 128*7*7)
        
    def forward(self, x):
        # x: (batch-size, 1, 28, 28)
        
        x = self.block1(x) # (batch-size, 32, 14, 14)
        x = self.block2(x) # (batch-size, 64, 7, 7)
        x = self.block3(x) # (batch-size, 128, 7, 7)
        x = x.view(x.size(0), -1)
        
        mean = self.mean_head(x)
        log_variance = self.log_var_head(x)
        
        return mean, log_variance
    
class UpBlock(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, normalization='batch', activation='relu'):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(input_channels, hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        if normalization=='batch':
            self.norm = nn.BatchNorm2d(hidden_channels)
        else:
            self.norm = None
        
        if activation=='relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
            
        #self.upsample = torch.Upsample(scale_factor=2.0, mode='linear')
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        #if self.upsample is not None:
        #    x = self.upsample(x)
        return x    
    
class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.block1 = UpBlock(128, 64)
        self.block2 = UpBlock(64, 32)
        self.block3 = nn.Conv2d(32, 1, kernel_size=1)
        self.linear = nn.Linear(128*7*7, 128*7*7)
        
    def forward(self, z):

        x = self.linear(z)
        x = x.view(x.size(0), 128, 7, 7)
        
        x = self.block1(x) # (B, 64, 14, 14)
        x = self.block2(x) # (B, 32, 28, 28)
        x = self.block3(x) # (B, 1, 28, 28)
        
        return x
        
class VAE(nn.Module):
    
    def __init__(self, alpha=0.5):
        super().__init__()
        
        self.alpha = alpha
        self.encoder = Encoder()
        self.decoder = Decoder()
        
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
        