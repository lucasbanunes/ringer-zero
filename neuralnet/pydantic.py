from abc import ABC
from typing import Self
from pathlib import Path
from pydantic import BaseModel
import yaml


class YamlModel(BaseModel, ABC):

    @classmethod
    def from_yaml(cls, yaml_file: Path | str, **kwargs) -> Self:
        """Load MLPTrainingJob from a YAML file."""
        if isinstance(yaml_file, str):
            yaml_file = Path(yaml_file)
        with yaml_file.open('r') as f:
            data = yaml.safe_load(f)
        for key, value in kwargs.items():
            data[key] = value
        return cls(**data)
