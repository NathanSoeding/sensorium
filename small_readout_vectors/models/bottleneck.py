import torch
import torch.nn as nn

class BatchOnlyNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var", torch.ones(1, dim))

    def forward(self, x):
        b, n, d = x.shape

        if self.training:
            mean = x.mean(dim=0)                    # (n, d)
            var = x.var(dim=0, unbiased=False)      # (n, d)

            # update running stats
            self.running_mean = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean.mean(dim=0).detach()
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var
                + self.momentum * var.mean(dim=0).detach()
            )
        else:
            mean = self.running_mean.repeat(n, 1)
            var = self.running_var.repeat(n, 1)

        x = (x - mean) / torch.sqrt(var + self.eps)

        return x
        
class Bottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dims, embedding_dim):
        super().__init__()

        def _get_mlp(in_dim, hidden_dims, embedding_dim):
            layers = []

            for out_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())

                in_dim = out_dim
            
            layers.append(nn.Linear(in_dim, embedding_dim))

            return nn.Sequential(*layers)

        self.feature_mlp = _get_mlp(in_dim, hidden_dims, embedding_dim)
        self.readout_mlp = _get_mlp(in_dim, hidden_dims, embedding_dim)

        self.feature_norm = BatchOnlyNorm(embedding_dim)

        self.cache = None

    def get_last_embeds(self):
        return self.cache
        
    def forward(self, feature_vec, readout_vec):
        feature_vec = feature_vec.permute(0, 2, 1)  # batch x num_neurons x c
        readout_vec = readout_vec.permute(0, 2, 1)  #     1 x num_neurons x c
        
        feature_emb = self.feature_mlp(feature_vec)  # batch x num_neurons x d_emb
        readout_emb = self.readout_mlp(readout_vec)  #     1 x num_neurons x d_emb

        feature_emb = self.feature_norm(feature_emb)

        self.cache = feature_emb, readout_emb

        pred = (feature_emb * readout_emb).sum(dim=-1)
        return pred