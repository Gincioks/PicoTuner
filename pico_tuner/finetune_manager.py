import os
import torch
import logging

from .llama import load_frozen_llama
from .mistral import load_frozen_mistral
from .utils import greedy_gen, log_lora


class FinetuneManager:
    seed = 54321

    def __init__(
            self,
            model_path,
            frozen_model,
            frozen_dtype,
            seq_len,
            tokenizer,
            lora_rank,
            log_lora_weight,
            log_lora_grad,
            batch_size,
            tokens,
            adamw_eps,
            lr,
            device,
            compute_dtype,
            eval_before_training,
            eval_period,
            test_prompts,
            gen_tokens,
    ):
        # Model config
        self.model_path = model_path
        self.frozen_model = frozen_model
        self.frozen_dtype = frozen_dtype
        self.tokenizer = tokenizer
        # Lora config
        self.lora_rank = lora_rank
        self.log_lora_weight = log_lora_weight
        self.log_lora_grad = log_lora_grad
        # Training config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.tokens = tokens
        self.adamw_eps = adamw_eps
        self.lr = lr
        self.device = device
        self.compute_dtype = compute_dtype
        # Eval config
        self.eval_before_training = eval_before_training
        self.eval_period = eval_period
        self.test_prompts = test_prompts
        self.gen_tokens = gen_tokens

        logging.basicConfig(format='%(asctime)s %(message)s',
                            level=logging.DEBUG, filename='logs/finetune.log')
        torch.random.manual_seed(self.seed)

    def get_batch(self, batch_size, seq_len, tokens, device):
        index = torch.randint(
            len(tokens) - seq_len, (batch_size,))
        x = torch.stack(
            [torch.tensor(tokens[i:i + seq_len]).to(torch.int64) for i in index])
        y = torch.stack(
            [torch.tensor(tokens[i + 1:i + seq_len + 1]).to(torch.int64) for i in index])
        return x.to(device), y.to(device)

    def train(self,  iterations: int):
        model_output_path = os.path.join(self.model_path, 'finetuned')
        if not os.path.exists(model_output_path):
            os.makedirs(model_output_path)

        if "llama" in self.frozen_model:
            model = load_frozen_llama(
                path=self.frozen_model,
                compute_dtype=self.compute_dtype,
                lora_rank=self.lora_rank,
                frozen_dtype=self.frozen_dtype
            ).to(self.device).to(self.compute_dtype)
        else:
            model = load_frozen_mistral(
                path=self.frozen_model,
                compute_dtype=self.compute_dtype,
                lora_rank=self.lora_rank,
                frozen_dtype=self.frozen_dtype
            ).to(self.device).to(self.compute_dtype)

        opt = torch.optim.AdamW(
            params=model.parameters(),
            lr=self.lr,
            eps=self.adamw_eps
        )

        last_loss = None
        for i in range(iterations):
            if i % self.eval_period == 0 and (i > 0 or self.eval_before_training):
                greedy_gen(model, self.tokenizer, self.device,
                           self.test_prompts, self.gen_tokens)
            logging.info(f'starting iteration {i}')
            print(f'\nIteration nr: {i}')
            X, y = self.get_batch(
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                tokens=self.tokens,
                device=self.device
            )
            opt.zero_grad()
            # both forward and backward passes are here.
            # returned loss is a scalar, not variable
            loss = model.manual_loop(X, y)
            opt.step()

            # optional logging of lora weights/gradients
            log_lora(
                lora_layers=model.lora_layers,
                log_weights=self.log_lora_weight,
                log_grad=self.log_lora_grad
            )

            logging.info(f'backprop done, loss after forward pass = {loss}')
            print(f'\nbackprop done, loss after forward pass = {loss}')
            if last_loss is None:
                last_loss = loss
            elif loss < last_loss:
                last_loss = loss
                logging.info('saving snapshot')
                torch.save(model.state_dict(), os.path.join(
                    model_output_path, f'state_dict_{i}.pth'))
