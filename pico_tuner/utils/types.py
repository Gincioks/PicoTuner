from dataclasses import dataclass


@dataclass
class ModelPathConfig:
    model_name: str
    params: str
    huggingface_name: str = None
    huggingface_url: str = None
    magnetic_link: str = None
    url: str = None


class ModelCollection:
    def __init__(self, name: str, path: str, versions: list[ModelPathConfig]):
        self.name = name
        self.path = path
        self.versions = versions
