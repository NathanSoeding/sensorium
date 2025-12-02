import torch
from torch import nn
from torch.nn import functional as F


class GumbelSoftmax(nn.Module):
    '''Sample from the Gumbel-Softmax distribution and optionally discretize. '''
    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
        
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 
    
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y


class Gaussian(nn.Module):
    ''' Linear layer from input_dim to latent_dim for mu and var then sample from gaussian using reparameterization trick. '''
    def __init__(self, input_dim, latent_dim):
        super(Gaussian, self).__init__()
        
        self.mu = nn.Linear(input_dim, latent_dim)
        self.var = nn.Linear(input_dim, latent_dim)

    def reparameterize(self, mu, var):
        std = torch.sqrt(var + 1e-10)
        noise = torch.randn_like(std)
        z = mu + noise * std
        
        return z      

    def forward(self, x):
        mu = self.mu(x)
        var = F.softplus(self.var(x))
        z = self.reparameterize(mu, var) if self.training else mu  # Don't sample when validating 
        
        return mu, var, z
    

class InferenceNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        hidden_dims, 
        batch_norm, 
        dropout, 
        transfer_fn, 
        num_components,  
    ):
        super(InferenceNet, self).__init__()

        # q(y|x)  Predicts mixture component
        qyx_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            qyx_layers.append(nn.Linear(prev_dim, h_dim))
            qyx_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            qyx_layers.append(transfer_fn)
            qyx_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        qyx_layers.append(GumbelSoftmax(prev_dim, num_components))
        self.qyx_layers = nn.ModuleList(qyx_layers)

        # q(z|y, x)  Predicts latent rep based on mixture component
        qzyx_layers = []
        prev_dim = input_dim + num_components
        for h_dim in hidden_dims:
            qzyx_layers.append(nn.Linear(prev_dim, h_dim))
            qzyx_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            qzyx_layers.append(transfer_fn)
            qzyx_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        qzyx_layers.append(Gaussian(prev_dim, latent_dim))
        self.qzyx_layers = nn.ModuleList(qzyx_layers)

    
    # q(y|x)  Predicts mixture component
    def qyx(self, x, temperature, hard):
        for layer in self.qyx_layers[:-1]:
            x = layer(x)
        logits, probs, y = self.qyx_layers[-1](x, temperature, hard)  # Last layers is Gumbel Softmax

        return logits, probs, y
    
    # q(z|y, x)  Predicts latent rep based on mixture component
    def qzyx(self, x, y):
        xy = torch.cat([x, y], dim=1)
        for layer in self.qzyx_layers[:-1]:
            xy = layer(xy)
        mu, var, z = self.qzyx_layers[-1](xy) # Last layer is Gaussian

        return mu, var, z

    def forward(self, x, temperature, hard):
        logits, probs, y = self.qyx(x, temperature, hard)

        mu, var, z = self.qzyx(x, y)

        return {
            'mu': mu, 
            'var': var, 
            'z': z, 
            'logits': logits, 
            'probs': probs, 
            'y': y, 
        }


class GenerativeNet(nn.Module):
    def __init__(
        self, 
        input_dim, 
        latent_dim, 
        hidden_dims, 
        batch_norm, 
        dropout, 
        transfer_fn,
        num_components,
    ):
        super(GenerativeNet, self).__init__()
        
        # p(z|y)  Mixture of Gaussians prior (learn mu/var vor every component)
        self.y_mu = nn.Linear(num_components, latent_dim)
        self.y_var = nn.Linear(num_components, latent_dim)

        # p(x|z)  Reconstructs input out of latent space
        pxz_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            pxz_layers.append(nn.Linear(prev_dim, h_dim))
            pxz_layers.append(nn.BatchNorm1d(h_dim)) if batch_norm else None
            pxz_layers.append(transfer_fn)
            pxz_layers.append(nn.Dropout(dropout)) if dropout > 0.0 else None
            prev_dim = h_dim
        pxz_layers.append(nn.Linear(prev_dim, input_dim))
        self.pxz_layers = nn.ModuleList(pxz_layers)

    # p(z|y)  Mixture of Gaussians prior (learn mu/var vor every component)
    def pzy(self, y):
        y_mu = self.y_mu(y)
        y_var = F.softplus(self.y_var(y))

        return y_mu, y_var
    
    # p(x|z)  Reconstructs input out of latent space
    def pxz(self, z):
        for layer in self.pxz_layers:
            z = layer(z)

        return z
    
    def forward(self, z, y):
        y_mu, y_var = self.pzy(y)

        x_rec = self.pxz(z)

        return {
            'y_mu': y_mu, 
            'y_var': y_var, 
            'x_rec': x_rec, 
        }


class MoGVAE(nn.Module):
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
        num_components=1,  
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

        # Inference Model

        self.inference = InferenceNet(input_dim, latent_dim, hidden_dims, batch_norm, dropout, transfer_fn, num_components)
        self.generative = GenerativeNet(input_dim, latent_dim, hidden_dims, batch_norm, dropout, transfer_fn, num_components)

        # weight initialization
        for m in self.modules():
            if type(m) == nn.Linear:
                nn.init.xavier_normal_(m.weight)
                if m.bias.data is not None:
                    nn.init.constant_(m.bias, 0) 

    
    def forward(self, x, temperature=1.0, hard=False, return_params=False):
        out_inf = self.inference(x, temperature, hard)

        z, y = out_inf['z'], out_inf['y']
        out_gen = self.generative(z, y)

        if not return_params:
            return out_gen['x_rec']
        else:
            return out_inf | out_gen  # Return merged output
