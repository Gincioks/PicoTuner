import os
import logging
import torch
from dotenv import load_dotenv
from dataclasses import dataclass, field

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatasetsConfig:
    # Datasets settings
    # default_dataset: str = "Open-Orca/OpenOrca"
    default_dataset: str = "databricks/databricks-dolly-15k"
    seq_len: int = int(os.getenv("SEQ_LEN", 2048))
    format_type: str = os.getenv("FORMAT_TYPE", 'llama2')


@dataclass
class LoraConfig:
    # LORA settings
    log_lora_grad: bool = bool(os.getenv("LOG_LORA_GRAD", False))
    log_lora_weight: bool = bool(os.getenv("LOG_LORA_WEIGHT", False))
    lora_rank: int = int(os.getenv("LORA_RANK", 4))


@dataclass
class TrainingConfig:
    # Training settings
    compute_dtype: torch.dtype = field(default_factory=lambda: torch.float16)
    frozen_dtype: torch.dtype = field(default_factory=lambda: torch.float16)
    batch_size: int = int(os.getenv("BATCH_SIZE", 6))
    device: str = os.getenv("DEVICE", 'mps')
    adamw_eps: float = float(os.getenv("ADAMW_EPS", 1e-6))
    lr: float = float(os.getenv("LR", 1e-6))


@dataclass
class OthersConfig:
    # Testing and others settings
    seed: int = int(os.getenv("SEED", 54321))
    offload_to: str = os.getenv("OFFLOAD_TO", 'disk')
    eval_before_training: bool = bool(os.getenv("EVAL_BEFORE_TRAINING", False))
    eval_period: int = int(os.getenv("EVAL_PERIOD", 20))
    gen_tokens: int = int(os.getenv("GEN_TOKENS", 32))
    prompt: str = os.getenv(
        "PROMPT", 'Write python script to print numbers from 1 to 10')
    log_level: int = int(os.getenv("LOG_LEVEL", logging.DEBUG))
