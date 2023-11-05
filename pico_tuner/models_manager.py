import json
import logging
import os

import torch

from .llama import prepare_llama_model
from .mistral import prepare_mistal_model
from .utils.types import ModelCollection, ModelPathConfig


class ModelsManager:
    seed = 54321

    def __init__(self):
        logging.basicConfig(format='%(asctime)s %(message)s',
                            level=logging.DEBUG, filename='logs/model.log')
        torch.random.manual_seed(self.seed)

    def download_model(self, model_name: str, model_path: str):
        #     magnet_link = 'your_magnet_link_here'
        #     save_path = './'
        #     downloader = TorrentDownloader()
        #     downloader.download(magnet_link, save_path)
        pass

    def get_models_list(self, models_folder_path: str):
        result = []

        # List all files in the given folder
        files = [f for f in os.listdir(
            models_folder_path) if f.endswith('.json')]

        for file_name in files:
            # Construct the full path of the file
            file_path = os.path.join(models_folder_path, file_name)

            # Read and parse the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

            versions = [ModelPathConfig(**item) for item in data]

            model_collection = ModelCollection(
                name=os.path.splitext(file_name)[0],
                path=file_name,
                versions=versions,
            )
            result.append(model_collection)

        return result

    def get_model_paths(self, models_folder_path: str, model_collection_path: str, model: str, create_frozen: bool):
        model_path = os.path.join(
            models_folder_path, model_collection_path, model)
        model_base = os.path.join(model_path, 'base')
        frozen_model = os.path.join(model_path, 'prepared')
        if not os.path.exists(model_base):
            os.makedirs(model_base)
        if not os.path.exists(frozen_model) and create_frozen:
            os.makedirs(frozen_model)
        return model_path, model_base, frozen_model

    def prepare_model(self, model_base: str, frozen_model: str, compute_dtype, frozen_dtype, offload_to, lora_rank):
        # if frozen model exist and contain files prepare model
        if os.path.exists(frozen_model) and os.path.exists(os.path.join(frozen_model, "tokenizer.model")):
            logging.info(f"Model already prepared at {frozen_model}")
            print(f"Model already prepared at {frozen_model}")
        else:
            if "llama" in frozen_model:
                prepare_llama_model(
                    llama2_path=model_base,
                    frozen_path=frozen_model,
                    compute_dtype=compute_dtype,
                    frozen_dtype=frozen_dtype,
                    offload_location=offload_to,
                    lora_rank=lora_rank,
                )
            else:
                prepare_mistal_model(
                    mistral_path=model_base,
                    frozen_path=frozen_model,
                    compute_dtype=compute_dtype,
                    frozen_dtype=frozen_dtype,
                    offload_location=offload_to,
                    lora_rank=lora_rank,
                )
