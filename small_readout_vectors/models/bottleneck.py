import torch
import torch.nn as nn

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

        self.feature_norm = nn.BatchNorm2d(embedding_dim, affine=False)

        self.cache = None

    def get_last_embeds(self):
        return self.cache
    
    def embed_neurons(self, readout_vec):
        readout_emb = self.readout_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb
        return readout_emb
        
    def forward(self, feature_map, readout_pos_sample, readout_vec):
        # feature_map: b, c, h, w
        # readout_vec: 1, c, neurons
        print(feature_map.shape)
        
        feature_map_emb = self.feature_mlp(
            feature_map.permute(0, 2, 3, 1)  # b, h, w, c
        ).permute(0, 3, 1, 2)  # b, d_emb, h, w
        feature_map_emb = self.feature_norm(feature_map_emb)
        readout_emb = self.readout_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb

        self.cache = feature_map_emb, readout_emb

        feature_vec_emb = readout_pos_sample(feature_map_emb).squeeze(-1).transpose(1, 2)  # b, neurons, d_emb

        pred = (feature_vec_emb * readout_emb).sum(dim=-1)
        return pred