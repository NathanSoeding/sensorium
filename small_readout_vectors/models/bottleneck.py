import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self, in_dim, hidden_dims, embedding_dim, weight_sharing=False, identity=False, sample_first=False):
        super().__init__()

        self.weight_sharing = weight_sharing
        self.sample_first = sample_first

        def _get_mlp(in_dim, hidden_dims, embedding_dim):
            layers = []

            for out_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.ReLU())

                in_dim = out_dim
            
            layers.append(nn.Linear(in_dim, embedding_dim))

            return nn.Sequential(*layers)

        if not identity:
            if weight_sharing:
                self.shared_mlp = _get_mlp(in_dim, hidden_dims, embedding_dim)
            else:
                self.feature_mlp = _get_mlp(in_dim, hidden_dims, embedding_dim)
                self.readout_mlp = _get_mlp(in_dim, hidden_dims, embedding_dim) 
        else:
            self.weight_sharing = True
            self.shared_mlp = nn.Identity()

        if self.sample_first:
            self.feature_norm = nn.BatchNorm1d(embedding_dim, affine=False)
        else:
            self.feature_norm = nn.BatchNorm2d(embedding_dim, affine=False)

        self.cache = None

    def get_last_embeds(self):
        return self.cache
    
    def embed_neurons(self, readout_vec):
        if self.weight_sharing:
            readout_emb = self.shared_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb
        else: 
            readout_emb = self.readout_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb
        return readout_emb
        
    def forward(self, feature_map, readout_pos_sample, readout_vec):
        # feature_map: b, c, h, w
        # readout_vec: 1, c, neurons
        if self.weight_sharing:
            feature_map_emb = self.shared_mlp(
                feature_map.permute(0, 2, 3, 1)  # b, h, w, c
            ).permute(0, 3, 1, 2)  # b, d_emb, h, w
            readout_emb = self.shared_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb
        else:
            feature_map_emb = self.feature_mlp(
                feature_map.permute(0, 2, 3, 1)  # b, h, w, c
            ).permute(0, 3, 1, 2)  # b, d_emb, h, w
            readout_emb = self.readout_mlp(readout_vec.permute(0, 2, 1))     # 1, neurons, d_emb

        if self.sample_first:
            feature_vec_emb = readout_pos_sample(feature_map_emb).squeeze(-1).transpose(1, 2)  # b, neurons, d_emb
            feature_vec_emb = self.feature_norm(feature_vec_emb.transpose(1, 2)).transpose(1, 2)

            self.cache = feature_vec_emb, readout_emb
        else:
            feature_map_emb = self.feature_norm(feature_map_emb)
            feature_vec_emb = readout_pos_sample(feature_map_emb).squeeze(-1).transpose(1, 2)  # b, neurons, d_emb
            
            self.cache = feature_map_emb, readout_emb

        pred = (feature_vec_emb * readout_emb).sum(dim=-1)
        return pred, feature_vec_emb