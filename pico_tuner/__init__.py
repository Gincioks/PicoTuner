import os

from .datasets_manager import DatasetsManager
from .finetune_manager import FinetuneManager
from .inference_manager import InferenceManager
from .models_manager import ModelsManager

if not os.path.exists("logs"):
    os.makedirs("logs")
