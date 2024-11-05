import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import build_tokenizer
from norms import get_norm, get_final_norm
from positional_embeddings import RotaryEmbedding, apply_rotary_pos_emb
import argparse
import yaml

def nonlinear_normalization(inputs, normalization_type, dim=-1):
    if normalization_type == 'softmax': 
        # NOTE: softmax = exp_l1_norm
        nonlinear_outputs = torch.exp(inputs)
        norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=1, dim=dim, keepdim=True) * inputs.shape[dim]
        outputs = norm_outputs
    elif normalization_type == 'gelu_l2_norm':
        nonlinear_outputs = F.gelu(inputs)
        norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
        outputs = norm_outputs
    elif normalization_type == 'l2_norm_gelu':
        norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
        nonlinear_outputs = F.gelu(norm_outputs)
        outputs = nonlinear_outputs
    else:
        raise NotImplementedError
    return outputs

class Pattention(nn.Module):
    """Parameter-based attention layer"""
    def __init__(self, input_channels, output_channels, param_token_num, normalization_type):
        super().__init__()
        self.key_param_tokens = nn.Parameter(torch.empty(param_token_num, input_channels))
        self.value_param_tokens = nn.Parameter(torch.empty(param_token_num, output_channels))
        self.normalization_type = normalization_type

        # scale factor and attention bias are optional
        self.register_buffer('scale', None)
        self.register_buffer('attn_bias', None)

    def forward(self, inputs):
        # Compute attention weights using dot product between inputs and key parameters
        attn_weights = inputs @ self.key_param_tokens.transpose(-2, -1)

        # Apply scale factor
        scale_factor = math.sqrt(inputs.shape[-1])
        attn_weights = attn_weights * scale_factor
        
        # Apply attention bias if present
        if self.attn_bias is not None:
            attn_weights = attn_weights + self.attn_bias.expand(attn_weights.shape)

        # Apply nonlinear normalization
        attn_weights = nonlinear_normalization(attn_weights, self.normalization_type)
    
        # Compute weighted sum with value parameters
        output = attn_weights @ self.value_param_tokens
        return output

class TokenFormerBlock(nn.Module):
    """Main TokenFormer layer block"""
    def __init__(self, hidden_size, qkv_slot_num, ffn_slot_num, proj_slot_num, config):
        super().__init__()

        self.num_attention_heads = config['num_attention_heads']
        self.hidden_size_per_attention_head = hidden_size // self.num_attention_heads
        self.normalization_type = config['norm_activation_type']
        
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
            precision=torch.float16 if config.get('fp16', False) else torch.float32
        )

    def forward(self, x):
        # Self attention
        residual = x
        x = self.norm1(x)
        
        # Get Q,K,V through parameter attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Reshape for attention heads
        b, s, h = x.size()
        q = q.view(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)
        k = k.view(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)
        v = v.view(b, s, self.num_attention_heads, self.hidden_size_per_attention_head)

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

        # Get rotary embeddings
        seq_len = q.size(1)
        cos, sin = self.rotary_emb(q, seq_len=seq_len)
        
        # Apply rotary embeddings
        q_rot, k_rot = apply_rotary_pos_emb(
            q_rot, k_rot, 
            cos, sin,
            offset=0
        )
        
        # Recombine rotary and pass-through dimensions
        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        
        # Transpose for attention calculation
        q = q.transpose(1, 2)  # [b, nh, s, hs]
        k = k.transpose(1, 2)  # [b, nh, s, hs]
        v = v.transpose(1, 2)  # [b, nh, s, hs]
        
        # Compute attention with proper scaling and normalization
        scale = math.sqrt(self.hidden_size_per_attention_head)
        
        # Create causal mask with proper dimensions
        causal_mask = torch.triu(
            torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=x.device), 
            diagonal=1
        )
        # Always use -1e4 for masking future tokens
        mask_value = torch.tensor(-1e4)
        attn_mask = torch.where(causal_mask, mask_value, torch.zeros_like(mask_value))
        
        # Compute attention scores
        attn_weights = (q @ k.transpose(-2, -1)) / scale  # [b, nh, s, s]
        attn_weights = attn_weights + attn_mask
        attn_weights = nonlinear_normalization(attn_weights, self.normalization_type)
        
        # Apply attention to values
        x = attn_weights @ v  # [b, nh, s, hs]
        
        # Reshape and project
        x = x.transpose(1, 2).contiguous().view(b, s, h)
        x = self.proj(x)
        
        # Add residual connection
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x

class TokenFormer(nn.Module):
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
        super().__init__()
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # TokenFormer layers
        self.layers = nn.ModuleList([
            TokenFormerBlock(
                hidden_size=hidden_size,
                qkv_slot_num=qkv_slot_num,
                ffn_slot_num=ffn_slot_num,
                proj_slot_num=proj_slot_num,
                config=config
            ) for _ in range(num_layers)
        ])
        
        # Get final norm based on config
        final_norm, final_eps = get_final_norm(config)
        self.norm_f = final_norm(hidden_size, eps=final_eps)
        
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        
    def forward(self, input_ids):
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
    """Simple text generation function"""
    model.eval()
    device = next(model.parameters()).device
    
    # Encode the prompt
    input_ids = torch.tensor([tokenizer.tokenize(prompt)], device=device)

    # print the token id
    print(input_ids[0].tolist())

    with torch.no_grad():
        for _ in range(max_length):
            # Get model predictions
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]

            if temperature == 0:
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
            else:
                # Apply temperature
                next_token_logits = next_token_logits / temperature
            
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # print the token id
            print(next_token[0].item())

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check if we've generated an EOS token
            if next_token[0].item() == tokenizer.eod_id:
                break
    
    return tokenizer.detokenize(input_ids[0].tolist())


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_mapping(num_layers):
    """Setup remapping rules for model loading"""
    # Map prefixes only - will be used to replace at start of keys
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
    state_dict = torch.load(weights_path, weights_only=True)
    
    """
    print("\nOriginal keys in loaded state dict:")
    for key in sorted(state_dict.keys()):
        print(f" {key} - size: {state_dict[key].size()}")

    print("\nExpected keys in model:")
    for key in sorted(model.state_dict().keys()):
        print(f" {key} - size: {model.state_dict()[key].size()}")
    """
    
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
    
    model.load_state_dict(state_dict)


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
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    
    print("Loading model weights...")
    load_model(model, args.model, config['num_layers'])
    
    # Generate text with the provided prompt
    generated_text = generate_text(model, tokenizer, args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Generated: {generated_text}")
