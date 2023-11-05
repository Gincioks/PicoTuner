import logging

import torch


def device_map(device):
    if str(device).startswith('mps'):
        return 'mps'
    return str(device)


def device_supports_dtype(device, dtype):
    try:
        torch.tensor([1.0, 2.0]).to(device).to(dtype)
        return True
    except TypeError:
        return False


global_id_auto_prepare = 0
global_id_auto_finetune = 0


def next_id(is_finetune=False):
    if is_finetune:
        global global_id_auto_finetune
        new_id = global_id_auto_finetune
        global_id_auto_finetune += 1
        return new_id
    global global_id_auto_prepare
    new_id = global_id_auto_prepare
    global_id_auto_prepare += 1
    return new_id


def save_rng_state(device='cpu'):
    if device == 'cpu':
        import torch
        return torch.random.get_rng_state()
    elif device.startswith('cuda'):
        import torch.cuda
        return torch.cuda.get_rng_state(device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        return torch.mps.get_rng_state()
    else:
        raise ValueError(f"Unsupported device: {device}")


def restore_rng_state(rng_state, device='cpu'):
    if device == 'cpu':
        import torch
        torch.random.set_rng_state(rng_state)
    elif device.startswith('cuda'):
        import torch.cuda
        torch.cuda.set_rng_state(rng_state, device=int(device.split(':')[1]))
    elif device.startswith('mps'):
        import torch.mps
        torch.mps.set_rng_state(rng_state)
    else:
        raise ValueError(f"Unsupported device: {device}")


def greedy_gen(model, tokenizer, device, prompt, max_new_tokens=50):
    tokens = torch.tensor(tokenizer.encode(
        prompt, True, False)).view(1, -1).to(device)
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(tokens)
        logits = logits[:, -1, :]
        _, next_token = torch.topk(logits, k=1, dim=-1)
        logging.info(
            f'next token: {next_token} {tokenizer.decode(next_token.tolist())}')
        print(
            f'{tokenizer.decode(next_token.tolist()[0])}', end="", flush=True)
        tokens = torch.cat((tokens, next_token), dim=1)

    print(f'Prompt: {prompt}')
    print('Generated tokens:')
    logging.info(f'Prompt: {prompt}')
    logging.info('Generated tokens:')
    for i, output in enumerate(tokens):
        print(f'{tokenizer.decode(output.tolist())}')
        logging.info(f'{tokenizer.decode(output.tolist())}')


def cleanup_cache(device='cpu'):
    if device.startswith('mps'):
        import torch.mps
        torch.mps.empty_cache()
