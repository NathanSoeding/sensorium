import torch
import torch.nn as nn

class Whitener(nn.Module):
    def __init__(self, model_dim, ema_decay):
        super().__init__()

        self.register_buffer('mu_ema', torch.zeros(1, 1, model_dim))
        self.register_buffer('cov_ema', torch.eye(model_dim))
        self.ema_decay = ema_decay

    def update_ema(self, features):
        batch, neurons, c = features.shape
        
        X = features.detach().flatten(0, 1)
        mu_hat = X.mean(dim=0, keepdims=True)

        self.mu_ema.copy_(
            (1 - self.ema_decay) * self.mu_ema
            + self.ema_decay * mu_hat
        )
        
        Xc = X - self.mu_ema.squeeze(0)

        sig_hat = Xc.T @ Xc / (batch * neurons - 1)

        self.cov_ema.copy_(
            (1 - self.ema_decay) * self.cov_ema 
            + self.ema_decay * sig_hat
        )
        
    def whiten(self, features, eps=1e-5):
        eye = torch.eye(self.cov_ema.shape[0], dtype=self.cov_ema.dtype, device=self.cov_ema.device)
        L = torch.linalg.cholesky(self.cov_ema + eps * eye)
        # W = L.inverse().T
        centered = features - self.mu_ema  # (B, N, D)
        whitened = torch.linalg.solve_triangular(L, centered.transpose(-1, -2), upper=False).transpose(-1, -2)

        # return (features - self.mu_ema) @ W
        return whitened
    
    def whiten_readouts(self, readouts, eps=1e-5):
        eye = torch.eye(self.cov_ema.shape[0], dtype=self.cov_ema.dtype, device=self.cov_ema.device)
        L = torch.linalg.cholesky(self.cov_ema + eps * eye)
        
        whitened_readouts = L.T @ readouts.squeeze()
        return whitened_readouts[None, :, None, :]

    def forward(self, features):
        features = features.transpose(1, 2)
        
        if self.training:
            self.update_ema(features)

        return self.whiten(features).transpose(1, 2).unsqueeze(3)