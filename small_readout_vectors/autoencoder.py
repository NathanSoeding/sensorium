import torch
from torch import nn

class Autoenc(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        hidden_layers=1, # If 'hidden_dims' is int this specifies amount of layers 
        hidden_dims=64, # Can be either int or list
        batch_norm=False, 
        dropout=0.0, 
        nonlinearity='ReLU', 
        leakyness=0.2, 
    ):
        super(Autoenc, self).__init__()
    
        assert (type(hidden_dims) is list) == (hidden_layers is None)

        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims] * hidden_layers

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
            encoder_layers.append(transfer_fn)
            encoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            decoder_layers.append(transfer_fn)
            decoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        
        return recon


class VAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        hidden_layers=1, # If 'hidden_dims' is int this specifies amount of layers 
        hidden_dims=64, # Can be either int or list
        batch_norm=False, 
        dropout=0.0, 
        nonlinearity='ReLU', 
        leakyness=0.2, 
    ):
        super(VAE, self).__init__()
    
        assert (type(hidden_dims) is list) == (hidden_layers is None)

        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims] * hidden_layers

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
            encoder_layers.append(transfer_fn)
            encoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.mu = nn.Linear(prev_dim, latent_dim)
        self.logvar = nn.Linear(prev_dim, latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            decoder_layers.append(transfer_fn)
            decoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
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

        return mu + eps * logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_params=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        
        return (recon, mu, logvar) if return_params else recon
    

class MoGVAE(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        K=5, 
        hidden_layers=1, # If 'hidden_dims' is int this specifies amount of layers 
        hidden_dims=64, # Can be either int or list
        batch_norm=False, 
        dropout=0.0, 
        nonlinearity='ReLU', 
        leakyness=0.2, 
    ):
        super(MoGVAE, self).__init__()
    
        assert (type(hidden_dims) is list) == (hidden_layers is None)

        if type(hidden_dims) is int:
            hidden_dims = [hidden_dims] * hidden_layers

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
            encoder_layers.append(transfer_fn)
            encoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.mu = nn.Linear(prev_dim, K * latent_dim)
        self.logvar = nn.Linear(prev_dim, K * latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            decoder_layers.append(transfer_fn)
            decoder_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.K = K
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, return_params=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon = self.decode(z)
        
        return (recon, mu, logvar) if return_params else recon
    