import math
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding
from tinygrad.nn.state import load_state_dict, torch_load
from tinygrad.dtype import dtypes
from tokenizer import build_tokenizer
from norms import get_norm, get_final_norm
from positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb
import argparse
import yaml
import numpy as np

def nonlinear_normalization(inputs, normalization_type, dim=-1):
    if normalization_type == 'gelu_l2_norm':
        nonlinear_outputs = inputs.gelu()
        norm = (nonlinear_outputs ** 2).sum(axis=dim, keepdim=True).sqrt()
        outputs = nonlinear_outputs / norm * math.sqrt(inputs.shape[dim])
    elif normalization_type == 'l2_norm_gelu':
        norm = (inputs ** 2).sum(axis=dim, keepdim=True).sqrt()
        norm_outputs = inputs / norm * math.sqrt(inputs.shape[dim])
        outputs = norm_outputs.gelu()
    else:
        raise NotImplementedError
    return outputs

class Pattention:
    """Parameter-based attention layer"""
    def __init__(self, input_channels, output_channels, param_token_num, normalization_type):
        self.key_param_tokens = Tensor.uniform(param_token_num, input_channels)
        self.value_param_tokens = Tensor.uniform(param_token_num, output_channels)
        self.normalization_type = normalization_type

    def __call__(self, inputs):
        # Compute attention weights using dot product
        attn_weights = inputs @ self.key_param_tokens.transpose()
        
        # Apply nonlinear normalization
        attn_weights = nonlinear_normalization(attn_weights, self.normalization_type)
        
        # Compute weighted sum with value parameters
        output = attn_weights @ self.value_param_tokens
        return output

class TokenFormerBlock:
    """Main TokenFormer layer block"""
    def __init__(self, hidden_size, qkv_slot_num, ffn_slot_num, proj_slot_num, config):
        self.normalization_type = config['norm_activation_type']
        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        
        # Get norm layers based on config
        norm, eps = get_norm(config)
        self.norm1 = norm(hidden_size, eps=eps)
        self.norm2 = norm(hidden_size, eps=eps)
        
        # Self-attention with parameter tokens
        self.query = Pattention(hidden_size, hidden_size, qkv_slot_num, self.normalization_type)
        self.key = Pattention(hidden_size, hidden_size, qkv_slot_num, self.normalization_type)
        self.value = Pattention(hidden_size, hidden_size, qkv_slot_num, self.normalization_type)
        self.proj = Pattention(hidden_size, hidden_size, proj_slot_num, self.normalization_type)
        
        # FFN
        self.ffn = Pattention(hidden_size, hidden_size, ffn_slot_num, self.normalization_type)

        # Initialize rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            dim=int(self.hidden_size_per_attention_head * 0.25),  # 25% of dimensions
            base=10000,  # default base
            max_seq_len=config.get('max_position_embeddings', 2048),
            dtype=dtypes.float16 if config.get('fp16', False) else dtypes.float32
        )

    def __call__(self, x):
        # Self attention
        residual = x
        x = self.norm1(x)
        
        # Get Q,K,V through parameter attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compute self-attention
        x = self.attention(q, k, v)
        
        # Project and add residual connection
        x = self.proj(x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

    def attention(self, q, k, v):

        # Reshape for attention heads
        b, s, h = q.shape
        q = q.reshape(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)
        k = k.reshape(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)
        v = v.reshape(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)

        # At this point, dimensions are:
        # q, k: [batch_size, seq_len, num_heads, head_size]

        # Apply rotary embeddings to first 25% of dimensions
        rotary_ndims = int(self.hidden_size_per_attention_head * 0.25)
        
        # Split the dimensions for partial rotary
        q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
        k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
        
        # After split:
        # q_rot, k_rot: [batch_size, seq_len, num_heads, rotary_ndims]
        # q_pass, k_pass: [batch_size, seq_len, num_heads, (head_size - rotary_ndims)]

        # Apply rotary embeddings
        seq_len = q.shape[1]
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, offset=0)
        
        # Recombine rotary and pass-through dimensions
        q = Tensor.cat(q_rot, q_pass, dim=-1)
        k = Tensor.cat(k_rot, k_pass, dim=-1)
        
        # Transpose for attention
        q = q.permute(0, 2, 1, 3)  # [b, nh, s, hs]
        k = k.permute(0, 2, 1, 3)  # [b, nh, s, hs]
        v = v.permute(0, 2, 1, 3)  # [b, nh, s, hs]
        
        # Create causal mask
        causal_mask = Tensor.triu(Tensor.ones(1, 1, seq_len, seq_len), diagonal=1)
        attn_mask = causal_mask.where(float('-inf'), 0.0)
        
        # Compute attention scores
        scale = 1 / math.sqrt(self.hidden_size_per_attention_head)
        attn_weights = (q @ k.transpose(-2, -1)) * scale  # [b, nh, s, s]
        attn_weights = attn_weights + attn_mask
        attn_weights = attn_weights.softmax(axis=-1)
        
        # Apply attention to values
        x = attn_weights @ v  # [b, nh, s, hs]
        
        # Reshape and return
        x = x.permute(0, 2, 1, 3).reshape(b, s, h)
        return x


class TokenFormer:
    """TokenFormer model for inference"""
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=768,
        num_layers=12,
        qkv_slot_num=768,
        ffn_slot_num=3072,
        proj_slot_num=768,
        max_position_embeddings=2048,
        config=None
    ):
        # Token embeddings
        self.word_embeddings = Embedding(vocab_size, hidden_size)
        
        # TokenFormer layers
        self.layers = [
            TokenFormerBlock(
                hidden_size=hidden_size,
                qkv_slot_num=qkv_slot_num,
                ffn_slot_num=ffn_slot_num,
                proj_slot_num=proj_slot_num,
                config=config
            ) for _ in range(num_layers)
        ]
        
        # Get final norm based on config
        final_norm, final_eps = get_final_norm(config)
        self.norm_f = final_norm(hidden_size, eps=final_eps)
        
        self.output = Linear(hidden_size, vocab_size, bias=False)
        
    def __call__(self, input_ids):
        # Embeddings
        hidden_states = self.word_embeddings(input_ids)
        
        # Forward through layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        # Final norm and output projection
        hidden_states = self.norm_f(hidden_states)
        logits = self.output(hidden_states)
        
        return logits

def generate_text(model, tokenizer, prompt, max_length=20, temperature=0.0, top_k=50):
    # Encode the prompt
    input_ids = Tensor([tokenizer.tokenize(prompt)])
    
    # Print the prompt without a newline
    print(tokenizer.detokenize(input_ids.numpy()[0].tolist()), end="", flush=True)
    
    for _ in range(max_length):
        # Get predictions
        outputs = model(input_ids)
        next_token_logits = outputs[:, -1, :]
        
        if temperature == 0:
            next_token = next_token_logits.argmax(axis=-1).unsqueeze(-1)
        else:
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                values, _ = next_token_logits.topk(top_k)
                indices_to_remove = next_token_logits < values[..., -1, None]
                next_token_logits = next_token_logits.where(indices_to_remove, float('-inf'))
            
            # Sample next token
            probs = next_token_logits.softmax(axis=-1)
            next_token = probs.multinomial(num_samples=1)
        
        # Print token
        print(tokenizer.detokenize(next_token.numpy()[0].tolist()), end="", flush=True)
        
        # Append to input_ids
        input_ids = Tensor.cat(input_ids, next_token, dim=1)
        
        # Check if we've generated an EOS token
        if next_token[0].item() == tokenizer.eod_id:
            break
    
    print()
    return tokenizer.detokenize(input_ids.numpy()[0].tolist())


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_mapping(num_layers):
    # Create mapping for model weights
    remap_dict = {
        'sequential.0.word_embeddings': 'word_embeddings'
    }
    
    # Map layers
    for i in range(2, num_layers + 2):
        new_idx = i - 2
        remap_dict[f'sequential.{i}.attention'] = f'layers.{new_idx}'
        remap_dict[f'sequential.{i}.mlp'] = f'layers.{new_idx}.ffn'
    
    # Map final norm
    remap_dict[f'sequential.{num_layers + 3}.norm'] = 'norm_f'
    
    return remap_dict


def load_model(model, weights_path, num_layers):
    """Load model weights from state dict"""
    state_dict = torch_load(weights_path)
    
    remap_dict = create_mapping(num_layers)
    
    # Copy output weight from word embeddings if missing
    if 'output.weight' not in state_dict:
        state_dict['output.weight'] = state_dict['sequential.0.word_embeddings.weight']
    
    # Rename keys in-place by replacing prefixes
    for old_prefix, new_prefix in remap_dict.items():
        for key in list(state_dict.keys()):
            if key.startswith(old_prefix):
                new_key = key.replace(old_prefix, new_prefix, 1)
                state_dict[new_key] = state_dict.pop(key)
    
    # Load state dict into model
    load_state_dict(model, state_dict)


# Example usage:
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='TokenFormer text generation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights file')
    parser.add_argument('--prompt', type=str, default="Once upon a time", 
                       help='Text prompt to start generation (default: "Once upon a time")')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Initialize tokenizer
    tokenizer = build_tokenizer(
        tokenizer_type=config['tokenizer-type'],
        vocab_file=config['vocab-file'],
        padding_multiple=128
    )
    
    # Initialize model
    model = TokenFormer(
        vocab_size = tokenizer.padded_vocab_size,
        hidden_size = config['hidden_size'],
        num_layers = config['num_layers'],
        qkv_slot_num = config['qkv_slot_num'],
        ffn_slot_num = config['ffn_slot_num'],
        proj_slot_num = config['proj_slot_num'],
        max_position_embeddings = config['max_position_embeddings'],
        config=config
    )
    
    print("Loading model weights...")
    load_model(model, args.model, config['num_layers'])
    
    # Generate text with the provided prompt
    print("Generating text...")
    generated_text = generate_text(model, tokenizer, args.prompt)
