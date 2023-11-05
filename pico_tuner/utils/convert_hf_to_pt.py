#!/usr/bin/env python3
import json
import os

from transformers import AutoTokenizer, MistralModel


def download_and_convert_to_custom_format(model_name: str, save_directory: str):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # Load the model and tokenizer
    model = MistralModel.from_pretrained(
        model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model weights in PyTorch .pth format
    # torch.save(model.state_dict(),
    #            f"{save_directory}/consolidated.00.pth")

    # Save model's configuration parameters in params.json
    with open(f"{save_directory}/params.json", 'w') as f:
        json.dump(model.config.to_dict(), f)

    model.save_pretrained(save_directory)

    # For the tokenizer, we'll save it and then rename the key file to tokenizer.model
    tokenizer.save_pretrained(save_directory)

# # Specify the model name and the directory
# model_name = "Open-Orca/Mistral-7B-OpenOrca"
# save_directory = "./OpenOrca_Mistral7B_Custom"


# download_and_convert_to_custom_format(model_name, save_directory)
