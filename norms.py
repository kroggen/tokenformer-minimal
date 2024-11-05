import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(config):
    eps = config.get('layernorm_epsilon', 1e-5)
    if config['norm'] == "layernorm_nonparam":
        norm = LayerNorm_NonParam
    elif config['norm'] == "layernorm":
        norm = nn.LayerNorm
    else:
        raise ValueError(f"norm {config['norm']} not recognized")
    return norm, eps


def get_final_norm(config):
    eps = config.get('layernorm_epsilon', 1e-5)
    if config['final_norm'] == "layernorm_nonparam":
        norm = LayerNorm_NonParam
    elif config['final_norm'] == "layernorm":
        norm = nn.LayerNorm
    else:
        raise ValueError(f"norm {config['final_norm']} not recognized")
    return norm, eps


class LayerNorm_NonParam(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.num_channels = dim
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, normalized_shape=(self.num_channels,), eps=self.eps)
