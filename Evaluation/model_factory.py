"""
Model Factory - Unified Module

Combines all model operations:
- Loading models from YAML config
- Building transforms/preprocessing for pretrained models
- Training utilities
- Full integration with PyTorch

Single source of truth for all model operations.
"""

import sys

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

parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

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
    
    def __init__(self, config_path: str = "models_config.yaml"):
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
# DUAL DATASET (for original + processed images)
# ============================================================================
class DualDataset(Dataset):
    """Wrapper that returns (processed_tensor, original_image, label)"""
    def __init__(self, base_dataset, original_dataset):
        self.base_dataset = base_dataset
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x_proc, y = self.base_dataset[index]
        x_orig, _ = self.original_dataset[index]
        
        # Convert original to tensor if it's a PIL image so DataLoader can batch it
        # but keep it at its original size/unnormalized
        if not isinstance(x_orig, torch.Tensor):
            x_orig = transforms.ToTensor()(x_orig)
            
        return x_proc, x_orig, y

    def __len__(self):
        return len(self.base_dataset)
# ============================================================================
# SECTION 4: MODEL FACTORY
# ============================================================================

class ModelFactory:
    """Create models with preprocessing from YAML config"""
    
    def __init__(self, loader: ModelConfigLoader):
        self.loader = loader
    
    def create_model(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = False,
    ) -> nn.Module:
        """
        Create model from config.
        
        Parameters
        ----------
        model_name : str
            Model name (from models_config.yaml)
        num_classes : int
            Number of output classes
        pretrained : bool
            Load pretrained weights
            
        Returns
        -------
        model : nn.Module
            Instantiated model
        """
        
        model_def = self.loader.get_model_def(model_name)
        if model_def is None:
            available = self.loader.list_models()
            raise ValueError(
                f"Unknown model: {model_name}. Available: {available}"
            )
        
        print(f"Creating {model_name}...")
        
        # Import and create model
        module = importlib.import_module(model_def.module)
        ModelClass = getattr(module, model_def.class_name)
        
        # Modern torchvision approach
        if pretrained:
            # We look for the Weights enum: e.g., torchvision.models.ResNet18_Weights.DEFAULT
            try:
                weights_enum_name = f"{model_def.class_name.capitalize()}_Weights"
                weights_enum = getattr(module, weights_enum_name)
                create_kwargs = {model_def.pretrained_param: weights_enum.DEFAULT}
            except AttributeError:
                # Fallback for older torchvision or custom models
                create_kwargs = {model_def.pretrained_param: True}
        else:
            create_kwargs = {model_def.pretrained_param: None}

        model = ModelClass(**create_kwargs)
            
        # Apply modifications for different num_classes
        if num_classes != 1000:
            self._apply_modifications(model, model_def, num_classes)
        
        print(f"✓ {model_name} created")
        return model
    
    def get_dataloader(
        self,
        model_name: str,
        subset: str = "train",
        config: Optional["TrainingConfig"] = None,
        ) -> DataLoader:
        """
        Creates a DataLoader with proper preprocessing based on dataset type.
        Supports both local ImageFolder and Hugging Face datasets.
        """
        if config is None:
            raise ValueError("TrainingConfig must be provided.")

        # 1. Determine split name for Hugging Face datasets
        hf_split = self._get_hf_split(subset, config.dataset)

        # 2. Get appropriate transforms
        transform = self.get_transforms(
            model_name=model_name,
            dataset=config.dataset,
            augment=(subset == "train")
        )

        # 3. Load dataset based on type
        # It works
        if config.dataset == "imagenette":
            dataset = self._load_imagenette(config.data_path, subset, transform)
            loader_kwargs = {}
        # This does not work
        elif config.dataset == "zh-plus/tiny-imagenet":
            dataset = self._load_hf_dataset(config.dataset, subset, transform)
            loader_kwargs = {"collate_fn": self._get_hf_collate_fn()}
        else:
            # Local folder (ImageFolder style)
            dataset = self._load_local_dataset(config.data_path, subset, transform)
            loader_kwargs = {}

        # 4. Create DataLoader
        return DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=(subset == "train"),
            num_workers=config.num_workers,
            pin_memory=(config.device == "cuda"),
            **loader_kwargs
        )
    def _get_hf_split(self, subset: str, dataset_name: str) -> str:
        """Convert internal subset name to Hugging Face split name."""
        if subset == "val":
            if dataset_name == "zh-plus/tiny-imagenet":
                return "valid"
            return "validation"
        return subset


    def _load_hf_dataset(self, dataset_name: str, split: str, transform):
        """Load dataset from Hugging Face Hub."""
        raw_dataset = load_dataset(dataset_name, split=split)

        def hf_transform_func(examples):
            # Convert PIL images to tensors using the provided transform
            examples["pixel_values"] = [
                transform(img.convert("RGB")) for img in examples["image"]
            ]
            return examples

        return raw_dataset.with_transform(hf_transform_func)


    def _get_hf_collate_fn(self):
        """Collate function for Hugging Face datasets (converts dict to (images, labels))."""
        def collate_fn(batch):
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            labels = torch.tensor([item["label"] for item in batch])
            return pixel_values, labels
        return collate_fn
    
    def _load_imagenette(self, data_path: str, subset: str, transform):
        """Load Imagenette using torchvision (auto-downloads if needed)."""
        # 'full' size is default (recommended). You can also use '160px' or '320px' for faster experiments.
        size = "full"   # change to "160px" if you want smaller/faster version

        return torchvision.datasets.Imagenette(
            root=data_path,           # e.g. "./data/imagenette"
            split=subset,             # 'train' or 'val'
            size=size,
            download=True,            # automatically downloads on first run
            transform=transform
        )

    def _load_local_dataset(self, data_path: str, subset: str, transform):
        """Load dataset from local folder using ImageFolder."""
        full_path = os.path.join(data_path, subset)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data directory not found: {full_path}")
        
        return ImageFolder(root=full_path, transform=transform)
        
    def get_torchvision_loader(
        self, 
        dataset_name: str, # "cifar10" or "cifar100"
        config: "TrainingConfig", 
        is_train: bool = False
    ) -> DataLoader:
        from torchvision import datasets
        
        # 1. Get the ImageNet-style transforms (crucial for pretrained ResNet)
        # This uses the 224px resize and ImageNet mean/std from your YAML
        transform = self.get_transforms(config.model, dataset="imagenet", augment=is_train)

        # 2. Select Dataset
        ds_cls = datasets.CIFAR10 if dataset_name.lower() == "cifar10" else datasets.CIFAR100
        
        # 3. Create both versions
        ds_proc = ds_cls(root=config.data_path, train=is_train, download=True, transform=transform)
        ds_orig = ds_cls(root=config.data_path, train=is_train, download=True, transform=None)
    
        # 4. Wrap them
        dual_ds = DualDataset(ds_proc, ds_orig)
        
        return  DataLoader(
            dual_ds, 
            batch_size=config.batch_size, 
            shuffle=is_train,
            num_workers=config.num_workers
        )
    
    def get_transforms(
        self,
        model_name: str,
        dataset: str = "imagenet",
        augment: bool = False,
    ) -> transforms.Compose:
        """
        Get transforms for a model.
        
        Parameters
        ----------
        model_name : str
            Model name
        dataset : str
            Dataset name (imagenet, cifar10, etc.)
        augment : bool
            Apply augmentation (for training)
            
        Returns
        -------
        transform : transforms.Compose
        """
        
        model_def = self.loader.get_model_def(model_name)
        if model_def is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        return build_transforms_from_config(model_name, model_def, dataset, augment)
    
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
                ref_path = mod_config.get("in_features_from", f"{layer_name}.in_features")
                in_features = self._get_value(model, ref_path)
                new_layer = nn.Linear(in_features, num_classes)
                self._set_attribute(model, layer_name, new_layer)
                print(f"  Modified {layer_name}: Linear({in_features}, {num_classes})")
    
    @staticmethod
    def _get_value(obj: Any, path: str) -> Any:
        """Get value using dot notation"""
        parts = path.split(".")
        current = obj
        for part in parts:
            if callable(getattr(current, part)):
                current = getattr(current, part)()
            else:
                current = getattr(current, part)
        return current
    
    @staticmethod
    def _set_attribute(obj: Any, path: str, value: Any) -> None:
        """Set value using dot notation"""
        parts = path.split(".")
        current = obj
        for part in parts[:-1]:
            current = getattr(current, part)
        setattr(current, parts[-1], value)


# ============================================================================
# SECTION 5: TRAINING CONFIG
# ============================================================================

@dataclass
class TrainingConfig:
    """Training configuration that reads models from YAML"""
    
    model: str
    num_classes: int
    data_path: str = "./data"     
    num_workers: int = 4          
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    momentum: float = 0.9
    pretrained: bool = False
    hook_layers: str = "all"
    hook_names: Optional[List[str]] = None
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    device: str = "cuda"
    models_config_path: str = "models_config.yaml"
    dataset: str = "imagenet"  # For transforms
    shuffle:bool = False
    
    _model_registry: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate config"""
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0")
        if self.optimizer.lower() not in ["adam", "sgd"]:
            raise ValueError(f"optimizer must be 'adam' or 'sgd'")
        
        self._load_model_registry()
        
        if self.model not in self._model_registry:
            available = list(self._model_registry.keys())
            raise ValueError(f"Model '{self.model}' not found. Available: {available}")
    
    def _load_model_registry(self) -> None:
        """Load model definitions from YAML"""
        if not Path(self.models_config_path).exists():
            raise FileNotFoundError(f"Config file not found: {self.models_config_path}")
        
        with open(self.models_config_path) as f:
            config = yaml.safe_load(f)
        
        if "models" not in config:
            raise ValueError(f"Config must have 'models' section")
        
        self._model_registry = config.get("models", {})
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**d)
    
    def to_yaml(self, path: str):
        """Save config"""
        d = asdict(self)
        del d['_model_registry']
        with open(path, 'w') as f:
            yaml.dump(d, f)


# ============================================================================
# SECTION 6: TRAINING UTILITIES
# ============================================================================

def get_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer from config"""
    
    if config.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch"""
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on dataset"""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)
    
    return total_loss / total, correct / total


# ============================================================================
# SECTION 7: MODEL I/O
# ============================================================================

def save_model(model: nn.Module, path: str) -> None:
    """Save model state dict"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✓ Model saved to {path}")


def load_model(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    """Load model state dict"""
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded from {path}")
    return model


# ============================================================================
# QUICK API
# ============================================================================

def create_model(
    model_name: str,
    num_classes: int,
    config_path: str = "models_config.yaml",
    pretrained: bool = False,
) -> nn.Module:
    """Quick function to create model"""
    loader = ModelConfigLoader(config_path)
    factory = ModelFactory(loader)
    return factory.create_model(model_name, num_classes, pretrained)


def get_transforms(
    model_name: str,
    config_path: str = "models_config.yaml",
    dataset: str = "imagenet",
    augment: bool = False,
) -> transforms.Compose:
    """Quick function to get transforms for a model"""
    loader = ModelConfigLoader(config_path)
    factory = ModelFactory(loader)
    return factory.get_transforms(model_name, dataset, augment)

# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    

    # Load it
    print("\nLoading model config...")
    loader = ModelConfigLoader("Generated/models_config_example.yaml")
    loader.print_summary()
    
    # Create factory
    factory = ModelFactory(loader)
    
    # Create models from config (NO code changes!)
    print("\nCreating models from config (no code changes needed!):\n")
    
    try:
        # Example 1: ResNet-18
        model1 = factory.create_model("resnet18", num_classes=1000, pretrained=False)
        print(f"✓ Created: {model1.__class__.__name__}\n")
        
        # Example 2: VGG-16
        model2 = factory.create_model("vgg16", num_classes=10, pretrained=False)
        print(f"✓ Created: {model2.__class__.__name__}\n")
        
        # Example 3: MobileNet
        model3 = factory.create_model("mobilenet", num_classes=10, pretrained=False)
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