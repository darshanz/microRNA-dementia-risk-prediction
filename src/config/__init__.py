import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

@dataclass
class ModelConfig:
    threshold: float
    n_components: int
    random_state: int
    
    @classmethod
    def from_yaml(cls, path: Path):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict['model'])