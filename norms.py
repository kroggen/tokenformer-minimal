from tinygrad.tensor import Tensor


def get_norm(config):
    eps = config.get('layernorm_epsilon', 1e-5)
    if config['norm'] == "layernorm_nonparam":
        norm = LayerNorm_NonParam
    elif config['norm'] == "layernorm":
        norm = LayerNorm
    else:
        raise ValueError(f"norm {config['norm']} not recognized")
    return norm, eps


def get_final_norm(config):
    eps = config.get('layernorm_epsilon', 1e-5)
    if config['final_norm'] == "layernorm_nonparam":
        norm = LayerNorm_NonParam
    elif config['final_norm'] == "layernorm":
        norm = LayerNorm
    else:
        raise ValueError(f"norm {config['final_norm']} not recognized")
    return norm, eps


class LayerNorm_NonParam:
    def __init__(self, dim, eps=1e-5):
        self.num_channels = dim
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        variance = x.var(axis=-1, keepdim=True)
        return (x - mean) / ((variance + self.eps) ** 0.5)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        variance = x.var(axis=-1, keepdim=True)
        return ((x - mean) / ((variance + self.eps) ** 0.5)) * self.weight + self.bias
