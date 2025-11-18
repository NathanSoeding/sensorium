import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[32, 16], batch_norm=True, dropout=0.1, nonlinearity='ReLU', leakyness=0.2):
        super(VAE, self).__init__()
        
        if nonlinearity == 'ReLU':
            transfer_fn = nn.ReLU()
        elif nonlinearity == 'GELU': 
            transfer_fn = nn.GELU()
        elif nonlinearity == 'LeakyReLU':
            transfer_fn = nn.LeakyReLU(leakyness) 

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            decoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
