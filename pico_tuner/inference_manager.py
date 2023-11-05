import logging
import torch
import os

from pico_tuner.llama import load_frozen_llama, LlamaTokenizer
from pico_tuner.mistral import load_frozen_mistral, MistralTokenizer
from pico_tuner.utils.torch_utils import greedy_gen


class InferenceManager:
    seed = 54321

    def __init__(self, frozen_dtype, compute_dtype,  device):
        self.frozen_dtype = frozen_dtype
        self.compute_dtype = compute_dtype
        self.device = device

        logging.basicConfig(
            format='%(asctime)s %(message)s', level=logging.DEBUG)
        torch.random.manual_seed(self.seed)

    def generate(self, frozen_model_path: str, lora_weights: str, prompt: str = None, length: int = 50):
        tokenizer_path = os.path.join(frozen_model_path, 'tokenizer.model')

        if "llama" in frozen_model_path:
            tokenizer = LlamaTokenizer(tokenizer_path)
            model = load_frozen_llama(frozen_model_path, dropout=0.0, lora_rank=4,
                                      frozen_dtype=self.frozen_dtype, compute_dtype=self.compute_dtype).to(self.device)
        else:
            tokenizer = MistralTokenizer(tokenizer_path)
            model = load_frozen_mistral(frozen_model_path, dropout=0.0, lora_rank=4,
                                        frozen_dtype=self.frozen_dtype, compute_dtype=self.compute_dtype).to(self.device)
        if lora_weights is not None:
            logging.debug(model.load_state_dict(
                torch.load(lora_weights), strict=False))

        logging.info('Model loaded.')

        if prompt is None:
            prompt = 'Cubestat reports the following metrics: '
        if length is None:
            length = 50

        greedy_gen(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            prompt=prompt,
            max_new_tokens=length
        )
