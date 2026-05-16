"""
Model Configuration Classes
========================

Clean separation of concerns for model, data, and training configurations.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Minimal config for model loading and inference.
    Used by evaluation scripts, PAF, and training alike.
    """
    model_name: str
    num_classes: int = 1000
    pretrained: bool = True
    device: str = 'cpu'
    models_config_path: str = 'config/models_config.yaml'
    dataset: str = 'imagenet'

    def __post_init__(self):
        if self.num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if self.device not in ('cpu', 'cuda', 'mps'):
            raise ValueError(f"Unknown device: {self.device}")

    @classmethod
    def from_yaml(cls, path: str, key: str = 'model') -> 'ModelConfig':
        """
        Load ModelConfig from YAML file.

        Parameters
        ----------
        path : str
            Path to YAML file
        key : str
            Top-level key in YAML. If not found, tries root dict.

        Returns
        -------
        ModelConfig
        """
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d.get(key, d))

    def to_yaml(self, path: str) -> None:
        """Save ModelConfig to YAML file."""
        from dataclasses import asdict
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(d, f)

@dataclass
class DataConfig:
    """
    Dataset and dataloader configuration.
    Used by evaluation scripts and training.
    """
    data_path: str = './data'
    dataset: str = 'imagenet'
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = False

    def __post_init__(self):
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    @classmethod
    def from_yaml(cls, path: str, key: str = 'data') -> 'DataConfig':
        """
        Load DataConfig from YAML file.

        Parameters
        ----------
        path : str
            Path to YAML file
        key : str
            Top-level key in YAML. If not found, tries root dict.

        Returns
        -------
        DataConfig
        """
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d.get(key, d))

    def to_yaml(self, path: str) -> None:
        """Save DataConfig to YAML file."""
        from dataclasses import asdict
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(d, f)


@dataclass
class TrainingConfig:
    """
    Training-only configuration.
    Composes ModelConfig and DataConfig — do not use in evaluation scripts.
    """
    model: ModelConfig
    data: DataConfig
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = 'adam'
    momentum: float = 0.9
    save_path: Optional[str] = None
    load_path: Optional[str] = None

    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.optimizer not in ('adam', 'sgd'):
            raise ValueError(f"optimizer must be adam or sgd")

    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        with open(path) as f:
            d = yaml.safe_load(f)
        model_cfg = ModelConfig(**d.pop('model', {}))
        data_cfg = DataConfig(**d.pop('data', {}))
        return cls(model=model_cfg, data=data_cfg, **d)

    def to_yaml(self, path: str) -> None:
        from dataclasses import asdict
        d = asdict(self)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(d, f)
