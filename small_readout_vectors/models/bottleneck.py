import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dims, embedding_dim):
        super(Bottleneck, self).__init__()

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

        self.feature_bn = nn.BatchNorm1d(embedding_dim, affine=False)

        self.cache = None

    def get_last_embeds(self):
        return self.cache
        
    def forward(self, feature_vec, readout_vec):
        feature_vec = feature_vec.permute(0, 2, 1)  # batch x num_neurons x c
        readout_vec = readout_vec.permute(0, 2, 1)  #     1 x num_neurons x c
        
        feature_emb = self.feature_mlp(feature_vec)  # batch x num_neurons x d_emb
        readout_emb = self.readout_mlp(readout_vec)  #     1 x num_neurons x d_emb

        feature_emb = self.feature_bn(
            feature_emb.transpose(1, 2)
        ).transpose(1, 2)

        self.cache = feature_emb, readout_emb

        pred = (feature_emb * readout_emb).sum(dim=-1)
        return pred