from tinygrad.tensor import Tensor


class SinusoidalPositionalEmbedding:
    def __init__(self, dim, base=10000, dtype=None):
        inv_freq = 1.0 / (base ** (Tensor.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq
        self.dtype = dtype

    def forward(self, x, seq_dim=1):
        t = Tensor.arange(x.shape[seq_dim]).cast(self.inv_freq.dtype)
        sinusoid_inp = t.reshape(-1, 1) * self.inv_freq
        sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
        emb = Tensor.cat([sin, cos], dim=-1)
        return emb.reshape(1, *emb.shape)


class RotaryEmbedding:
    def __init__(self, dim, max_seq_len, base=10000, dtype=None, save_inv_freqs=False):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.dtype = dtype
        
        # Mark cached tensors as non-persistent (won't be saved in state dict)
        self._cos_cached = None
        self._sin_cached = None

    def _prepare_cache(self, seq_len):
        inv_freq = 1.0 / (self.base ** (Tensor.arange(0, self.dim, 2).float() / self.dim))
        t = Tensor.arange(seq_len).cast(inv_freq.dtype)
        freqs = t.reshape(-1, 1) * inv_freq
        emb = Tensor.cat(freqs, freqs, dim=-1)
        
        cos_cached = emb.cos().reshape(seq_len, 1, 1, -1)
        sin_cached = emb.sin().reshape(seq_len, 1, 1, -1)
        
        return cos_cached, sin_cached

    def __call__(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        assert seq_len <= self.max_seq_len
        
        # Initialize cache if not already done
        if self._cos_cached is None or self._sin_cached is None:
            self._cos_cached, self._sin_cached = self._prepare_cache(self.max_seq_len)
        
        return (
            self._cos_cached[:seq_len],
            self._sin_cached[:seq_len]
        )


def rotate_half(x):
    half_dim = x.shape[-1] // 2
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    return Tensor.cat(-x2, x1, dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, offset: int = 0):
    cos = cos[offset:q.shape[0] + offset]
    sin = sin[offset:q.shape[0] + offset]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
