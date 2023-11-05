import torch

from dataclasses import dataclass
from typing import Optional


@dataclass
class MistralModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    head_dim: int = 128
    n_kv_heads: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    use_biases: bool = False
    model_parallel: int = 1
    is_sequence_parallel: bool = False
    efficient_attn: bool = True
    fused_rms_norm: bool = True
    ragged_attention: str = "4096"
    checkpoint: bool = True
    use_cache: bool = False
    norm_eps: float = 1e-5
    hidden_dim: int = 14336
    max_seq_len: int = 2048
    init: str = 'DEFAULT'
    dropout: float = 0.0  # unless we bring back
    ffn_dim_multiplier: Optional[float] = None
    compute_dtype: torch.dtype = torch.float16
    offload_location: str = 'disk'  # 'disk' or 'ram'
    rope_theta: float = 10000.0
    lora_rank: int = 8
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    served_model_path: str = ''  # relative path by default
    cached_data_path: str = ''  # relative path by default
    init_frozen: bool = True
    frozen_dtype: torch.dtype = torch.float16
    vocab_size: int = 32000
    vocab_size_override: int = 32000
    max_concurrent_tokens: int = 65536
    rms_norm: str = 'PRE'
    attn_tanh_gating: Optional[bool] = None
    softmax_tanh_gating: Optional[bool] = None
    cust_bwd: bool = False
    recompute_w1_out: bool = True
    recompute_w3_out: bool = True
    recompute_attn: bool = True
    zero2: bool = False
    cutlass: bool = False
