"""
Model Factory - Unified Module

Combines all model operations:
- Loading models from YAML config
- Building transforms/preprocessing for pretrained models
- Training utilities
- Full integration with PyTorch

Single source of truth for all model operations.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import torchvision
import yaml
from pathlib import Path
import importlib
from torchvision.datasets import ImageFolder
import os
from datasets import load_dataset
from core.model.config import ModelConfig, DataConfig

# ============================================================================
# SECTION 1: MODEL DEFINITIONS & CONFIG
# ============================================================================

@dataclass
class ModelDefinition:
    """Single model configuration from YAML"""
    name: str
    class_name: str
    module: str
    pretrained_param: str = "pretrained"
    modifications: Optional[Dict[str, Dict[str, Any]]] = None
    preprocessing: Optional[Dict[str, Dict[str, Any]]] = None
    description: str = ""
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelDefinition":
        return cls(**d)


# ============================================================================
# SECTION 2: CONFIG LOADER (Core Registry)
# ============================================================================

class ModelConfigLoader:
    """Load model definitions and preprocessing from YAML"""
    
    def __init__(self, config_path: str = "config/models_config.yaml"):
        self.config_path = Path(config_path)
        self.models: Dict[str, ModelDefinition] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load model definitions from YAML"""
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        
        if "models" not in config:
            raise ValueError("Config must have 'models' section")
        
        for model_name, model_def in config["models"].items():
            try:
                self.models[model_name.lower()] = ModelDefinition.from_dict(
                    {**model_def, "name": model_name}
                )
            except Exception as e:
                print(f"Warning: Failed to load model '{model_name}': {e}")
    
    def get_model_def(self, name: str) -> Optional[ModelDefinition]:
        """Get model definition by name"""
        return self.models.get(name.lower())
    
    def list_models(self) -> list:
        """List all available models"""
        return sorted(list(self.models.keys()))
    
    def print_summary(self) -> None:
        """Print available models"""
        print("\n" + "="*80)
        print("AVAILABLE MODELS")
        print("="*80 + "\n")
        
        for name in self.list_models():
            definition = self.models[name]
            desc = f" - {definition.description}" if definition.description else ""
            print(f"  {name:20s} | {definition.class_name:20s}{desc}")
        
        print(f"\nTotal: {len(self.models)} models\n")


# ============================================================================
# SECTION 3: TRANSFORMS BUILDER
# ============================================================================

def build_transforms_from_config(
    model_name: str,
    model_def: ModelDefinition,
    dataset: str = "imagenet",
    augment: bool = False,
) -> transforms.Compose:
    """
    Build transforms for a pretrained model from config.
    
    Parameters
    ----------
    model_name : str
        Model name
    model_def : ModelDefinition
        Model definition with preprocessing config
    dataset : str
        Dataset name (e.g., "imagenet", "cifar10")
    augment : bool
        Apply data augmentation (for training)
        
    Returns
    -------
    transform : transforms.Compose
        Composed transforms for the model
        
    Example
    -------
    config = loader.get_model_def("resnet18")
    transform = build_transforms_from_config("resnet18", config, dataset="imagenet")
    dataset = ImageNet(..., transform=transform)
    """
    
    if not model_def.preprocessing or dataset not in model_def.preprocessing:
        print(f"Warning: No preprocessing config for {model_name}/{dataset}")
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    
    prep_config = model_def.preprocessing[dataset]
    transform_list = []
    
    # Augmentation (only for training)
    if augment:
        transform_list.extend([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        # Resize and center crop (for evaluation)
        if "resize" in prep_config:
            transform_list.append(transforms.Resize(prep_config["resize"]))
        if "center_crop" in prep_config:
            transform_list.append(transforms.CenterCrop(prep_config["center_crop"]))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if "normalize" in prep_config:
        norm_config = prep_config["normalize"]
        transform_list.append(
            transforms.Normalize(
                mean=norm_config["mean"],
                std=norm_config["std"],
            )
        )
    
    return transforms.Compose(transform_list)

# ============================================================================
# SECTION 3.5: DUAL DATASET
# ============================================================================

class DualDataset(Dataset):
    """Wrapper that returns (processed_tensor, original_image, label)"""
    def __init__(self, base_dataset, original_dataset):
        self.base_dataset = base_dataset
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x_proc, y = self.base_dataset[index]
        x_orig, _ = self.original_dataset[index]

        if not isinstance(x_orig, torch.Tensor):
            x_orig = transforms.ToTensor()(x_orig)

        return x_proc, x_orig, y

    def __len__(self):
        return len(self.base_dataset)

# ============================================================================
# SECTION 4: MODEL FACTORY
# ============================================================================

class ModelFactory:
    """
    Creates models and dataloaders from ModelConfig and DataConfig.
    No training logic — that belongs in trainer.py.
    """

    def __init__(self, config_path: str = 'config/models_config.yaml'):
        self.loader = ModelConfigLoader(config_path)
    
    def create_model(self, cfg: ModelConfig) -> nn.Module:
        """Create and return model. No TrainingConfig needed."""
        model_def = self.loader.get_model_def(cfg.model_name)
        if model_def is None:
            raise ValueError(
                f"Unknown model '{cfg.model_name}'. "
                f"Available: {self.loader.list_models()}"
            )

        module = importlib.import_module(model_def.module)
        ModelClass = getattr(module, model_def.class_name)

        if cfg.pretrained:
            weights_name = f"{model_def.class_name}_Weights"
            try:
                weights_enum = getattr(module, weights_name)
                model = ModelClass(**{model_def.pretrained_param: weights_enum.DEFAULT})
            except AttributeError:
                model = ModelClass(**{model_def.pretrained_param: True})
        else:
            model = ModelClass(**{model_def.pretrained_param: None})

        if cfg.num_classes != 1000:
            self._apply_modifications(model, model_def, cfg.num_classes)

        device = torch.device(cfg.device)
        model.name = cfg.model_name
        model = model.to(device).eval()
        print(f"✓ {cfg.model_name} loaded (pretrained={cfg.pretrained}, device={cfg.device})")
        return model
    
    def get_dataloader(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> DataLoader:
        """
        Create dataloader. Takes DataConfig + ModelConfig — no TrainingConfig.
        """
        transform = self.get_transforms(model_cfg, augment=data_cfg.shuffle)

        if data_cfg.dataset == 'imagenette':
            dataset = self._load_imagenette(data_cfg, transform)
        elif data_cfg.dataset == 'zh-plus/tiny-imagenet':
            hf_split = 'valid' if data_cfg.dataset == 'zh-plus/tiny-imagenet' else 'validation'
            dataset = self._load_hf_dataset(data_cfg.dataset, hf_split, transform)
            return DataLoader(
                dataset,
                batch_size=data_cfg.batch_size,
                shuffle=data_cfg.shuffle,
                num_workers=data_cfg.num_workers,
                collate_fn=self._get_hf_collate_fn(),
            )
        else:
            dataset = self._load_local_dataset(data_cfg, transform)

        return DataLoader(
            dataset,
            batch_size=data_cfg.batch_size,
            shuffle=data_cfg.shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=(model_cfg.device == 'cuda'),
        )
    
    def _get_hf_split(self, subset: str, dataset_name: str) -> str:
        """Convert internal subset name to Hugging Face split name."""
        if subset == "val":
            if dataset_name == "zh-plus/tiny-imagenet":
                return "valid"
            return "validation"
        return subset


    def _load_hf_dataset(self, name: str, split: str, transform) -> Dataset:
        from datasets import load_dataset
        raw = load_dataset(name, split=split)
        def _apply(examples):
            examples['pixel_values'] = [
                transform(img.convert('RGB')) for img in examples['image']
            ]
            return examples
        return raw.with_transform(_apply)


    def _get_hf_collate_fn(self):
        """Collate function for Hugging Face datasets (converts dict to (images, labels))."""
        def collate_fn(batch):
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            labels = torch.tensor([item["label"] for item in batch])
            return pixel_values, labels
        return collate_fn
    
    def _load_imagenette(self, cfg: DataConfig, transform) -> Dataset:
        return torchvision.datasets.Imagenette(
            root=cfg.data_path, split='val',
            size='full', download=True, transform=transform,
        )

    def _load_local_dataset(self, cfg: DataConfig, transform) -> Dataset:
        full_path = os.path.join(cfg.data_path, 'val')
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Not found: {full_path}")
        return ImageFolder(root=full_path, transform=transform)
        
    def get_torchvision_loader(
        self,
        dataset_name: str,
        model_cfg: ModelConfig,
        data_cfg: DataConfig,
        is_train: bool = False,
    ) -> DataLoader:
        """
        Load CIFAR-10/CIFAR-100 dataset with transforms for pretrained model.

        Parameters
        ----------
        dataset_name : str
            "cifar10" or "cifar100"
        model_cfg : ModelConfig
            Model configuration (for transforms)
        data_cfg : DataConfig
            Data configuration (batch size, num workers, data path)
        is_train : bool
            Training mode (applies augmentation)

        Returns
        -------
        DataLoader
        """
        from torchvision import datasets

        transform = self.get_transforms(model_cfg, augment=is_train)

        ds_cls = datasets.CIFAR10 if dataset_name.lower() == "cifar10" else datasets.CIFAR100

        ds_proc = ds_cls(
            root=data_cfg.data_path,
            train=is_train,
            download=True,
            transform=transform,
        )
        ds_orig = ds_cls(
            root=data_cfg.data_path,
            train=is_train,
            download=True,
            transform=None,
        )

        dual_ds = DualDataset(ds_proc, ds_orig)

        return DataLoader(
            dual_ds,
            batch_size=data_cfg.batch_size,
            shuffle=is_train,
            num_workers=data_cfg.num_workers,
        )
    
    def get_transforms(
        self,
        cfg: ModelConfig,
        augment: bool = False,
    ) -> transforms.Compose:
        """
        Get transforms for a model.

        Parameters
        ----------
        cfg : ModelConfig
            Model configuration
        augment : bool
            Apply augmentation (for training)

        Returns
        -------
        transform : transforms.Compose
        """

        model_def = self.loader.get_model_def(cfg.model_name)
        if model_def is None:
            raise ValueError(f"Unknown model: {cfg.model_name}")

        return build_transforms_from_config(cfg.model_name, model_def, cfg.dataset, augment)
    
    def _apply_modifications(
        self,
        model: nn.Module,
        model_def: ModelDefinition,
        num_classes: int,
    ) -> None:
        """Apply modifications to adapt for num_classes"""
        
        if not model_def.modifications:
            return
        
        for layer_name, mod_config in model_def.modifications.items():
            mod_type = mod_config.get("type", "linear")
            
            if mod_type == "linear":
                ref = mod_config.get('in_features_from', f'{layer_name}.in_features')
                in_f  = self._get_attr(model, ref)
                new_layer = nn.Linear(in_f, num_classes)
                self._set_attr(model, layer_name, new_layer)
                print(f"  Modified {layer_name}: Linear({in_f}, {num_classes})")
    
    @staticmethod
    def _get_attr(obj, path: str):
        for part in path.split('.'):
            obj = getattr(obj, part)   # attribute access only — never call
        return obj
    
    @staticmethod
    def _set_attr(obj, path: str, value) -> None:
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)



# ============================================================================
# SECTION 6: QUICK API FUNCTIONS
# ============================================================================

def create_model(
    model_name: str,
    num_classes: int,
    config_path: str = "config/models_config.yaml",
    pretrained: bool = False,
) -> nn.Module:
    """Quick function to create model"""
    from core.model_config import ModelConfig
    cfg = ModelConfig(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained
    )
    factory = ModelFactory(config_path)
    return factory.create_model(cfg)


def get_transforms(
    model_name: str,
    dataset: str = "imagenet",
    augment: bool = False,
    config_path: str = "config/models_config.yaml",
):
    """Quick function to get transforms"""
    from core.model_config import ModelConfig
    cfg = ModelConfig(model_name=model_name, dataset=dataset)
    factory = ModelFactory(config_path)
    return factory.get_transforms(cfg)


# ============================================================================
# ============================================================================

if __name__ == "__main__":
    factory = ModelFactory('config/models_config_example.yaml')
    loader = factory.loader
    loader.print_summary()

    print("\nCreating models from config (no code changes needed!):\n")

    try:
        model_cfg = ModelConfig(model_name='resnet18', num_classes=1000, pretrained=False)
        model1 = factory.create_model(model_cfg)
        print(f"✓ Created: {model1.__class__.__name__}\n")

        model_cfg = ModelConfig(model_name='vgg16', num_classes=10, pretrained=False)
        model2 = factory.create_model(model_cfg)
        print(f"✓ Created: {model2.__class__.__name__}\n")

        model_cfg = ModelConfig(model_name='mobilenet', num_classes=10, pretrained=False)
        model3 = factory.create_model(model_cfg)
        print(f"✓ Created: {model3.__class__.__name__}\n")

    except Exception as e:
        print(f"Error: {e}")

    print("="*80)
    print("✓ Config-based model loading works!")
    print("="*80 + "\n")
    
    print("""
KEY BENEFITS:

✅ Add new models by editing YAML - NO Python code changes!
✅ Non-technical users can add models
✅ Version control config separately
✅ Easy to share model definitions
✅ Supports custom models (just specify module path)
✅ Easy to modify parameters per model

HOW TO USE:

1. Edit models_config.yaml
2. Add new model section:
   
   my_custom_model:
     class_name: MyModel
     module: my_package.models
     pretrained_param: pretrained
     modifications:
       fc:
         type: linear
         in_features_from: "fc.in_features"

3. Use in code:

   model = create_model_from_config(
       "my_custom_model",
       num_classes=10,
       device=device
   )

That's it! No code changes needed!
""")
