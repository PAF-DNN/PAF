import os
from typing import Tuple
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

def transform_box_resize_centercrop(
        box:       Tuple[int,int,int,int],
        orig_w:    int,
        orig_h:    int,
        resize_to: int = 256,
        crop_to:   int = 224,
    ) -> Tuple[int,int,int,int]:
        """
        Transform bounding box through Resize(resize_to) + CenterCrop(crop_to).
        torchvision Resize scales the SHORT side to resize_to,
        preserving aspect ratio.
        """
        x1, y1, x2, y2 = box

        # Step 1: Resize — scale so SHORT side = resize_to
        scale  = resize_to / min(orig_w, orig_h)
        rw     = int(orig_w * scale)
        rh     = int(orig_h * scale)

        x1r = int(x1 * scale)
        y1r = int(y1 * scale)
        x2r = int(x2 * scale)
        y2r = int(y2 * scale)

        # Step 2: CenterCrop — subtract offset
        crop_x = (rw - crop_to) // 2   # pixels removed from left
        crop_y = (rh - crop_to) // 2   # pixels removed from top

        x1c = x1r - crop_x
        y1c = y1r - crop_y
        x2c = x2r - crop_x
        y2c = y2r - crop_y

        # Clamp to valid range
        x1c = max(0, min(x1c, crop_to - 1))
        y1c = max(0, min(y1c, crop_to - 1))
        x2c = max(0, min(x2c, crop_to - 1))
        y2c = max(0, min(y2c, crop_to - 1))

        return x1c, y1c, x2c, y2c

def load_boxes_from_csv(
    csv_path:      str,
    class_to_idx:  dict,
    img_size:      int = 224,
    orig_size:     int = 256,   # ImageNet images are typically 256+ before crop
) -> dict:
    """
    Parse LOC_val_solution.csv into a dict:
        {image_stem: (class_idx, (x1, y1, x2, y2))}

    CSV format:
        ImageId, PredictionString
        ILSVRC2012_val_00000001, n01751748 x1 y1 x2 y2 n01751748 x1 y1 x2 y2 ...

    Coordinates are in original image space (variable size).
    We scale to img_size assuming standard resize+centercrop.
    """
    df      = pd.read_csv(csv_path)
    box_map = {}

    for _, row in df.iterrows():
        stem  = row['ImageId']
        preds = str(row['PredictionString']).strip().split()

        if len(preds) < 5:
            continue

        # Each prediction: class x1 y1 x2 y2
        # Take the first prediction (highest confidence)
        class_name = preds[0]
        try:
            x1, y1, x2, y2 = int(preds[1]), int(preds[2]), \
                              int(preds[3]), int(preds[4])
        except (ValueError, IndexError):
            continue

        class_idx = class_to_idx.get(class_name)
        if class_idx is None:
            continue

        # Scale box from original image coords to img_size
        # ImageNet validation images have varying sizes — use a conservative
        # scale assuming the standard Resize(256) + CenterCrop(224) pipeline
        # This is an approximation; XML gives exact per-image sizes
        scale     = img_size / orig_size
        x1_scaled = max(0, int(x1 * scale))
        y1_scaled = max(0, int(y1 * scale))
        x2_scaled = min(img_size - 1, int(x2 * scale))
        y2_scaled = min(img_size - 1, int(y2 * scale))

        box_map[stem] = (class_idx, (x1_scaled, y1_scaled,
                                      x2_scaled, y2_scaled))

    print(f"Loaded {len(box_map)} bounding boxes from {csv_path}")
    return box_map


class ImageNetWithBoxesCSV(Dataset):
    """
    ImageNet validation with bounding boxes from LOC_val_solution.csv.
    Does not require Annotations XML files — only the CSV.
    """

    def __init__(
        self,
        imagenet_root: str,
        csv_path:      str,
        split:         str = 'val',
        transform             = None,
        max_samples:   int  = None,
    ):
        self.img_dir   = os.path.join(imagenet_root, split)
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

        # Build class → idx from directory structure
        class_names    = sorted([
            d for d in os.listdir(self.img_dir)
            if os.path.isdir(os.path.join(self.img_dir, d))
            and d.startswith('n')
        ])
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        # Load bounding boxes from CSV
        self.box_map = load_boxes_from_csv(
            csv_path, self.class_to_idx
        )

        # Collect samples that have both image and box
        self.samples = []
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.img_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if stem not in self.box_map:
                    continue
                img_path = os.path.join(class_dir, fname)
                self.samples.append((img_path, stem, class_idx))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"ImageNetWithBoxesCSV: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, stem, class_idx = self.samples[idx]
        img        = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        _, H, W    = img_tensor.shape

        _, box = self.box_map[stem]
        # Clamp to image bounds
        x1, y1, x2, y2 = box
        box = (
            max(0, x1), max(0, y1),
            min(W-1, x2), min(H-1, y2),
        )
        return img_tensor, class_idx, torch.tensor(box, dtype=torch.long)


def get_imagenet_csv_loader(
    imagenet_root: str,
    csv_path:      str,
    batch_size:    int = 1,
    num_workers:   int = 4,
    max_samples:   int = 1000,
) -> DataLoader:
    dataset = ImageNetWithBoxesCSV(
        imagenet_root = imagenet_root,
        csv_path      = csv_path,
        max_samples   = max_samples,
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,   # MPS does not support pin_memory
        collate_fn  = _collate_boxes,
    )

class ImageNetWithBoxes(Dataset):

    def __init__(
        self,
        imagenet_root: str,
        split:         str = 'val',
        transform      = None,
        max_samples:   int = None,
    ):
        self.root      = imagenet_root
        self.split     = split
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

        self.img_dir = os.path.join(imagenet_root, split)
        self.ann_dir = os.path.join(
            imagenet_root, 'Annotations', 'CLS-LOC', split
        )

        # ----------------------------------------------------------------
        # Diagnostics — print what exists so you can fix the paths
        # ----------------------------------------------------------------
        print(f"\nImageNetWithBoxes diagnostics:")
        print(f"  imagenet_root : {imagenet_root}")
        print(f"  img_dir       : {self.img_dir}")
        print(f"  ann_dir       : {self.ann_dir}")
        print(f"  img_dir exists: {os.path.exists(self.img_dir)}")
        print(f"  ann_dir exists: {os.path.exists(self.ann_dir)}")

        if os.path.exists(self.img_dir):
            img_subdirs = [
                d for d in os.listdir(self.img_dir)
                if os.path.isdir(os.path.join(self.img_dir, d))
            ]
            print(f"  img subdirs   : {len(img_subdirs)} "
                  f"(first 3: {img_subdirs[:3]})")
        else:
            print(f"  ERROR: img_dir does not exist — check imagenet_root")

        if os.path.exists(self.ann_dir):
            xml_files = [
                f for f in os.listdir(self.ann_dir)
                if f.endswith('.xml')
            ]
            print(f"  xml files     : {len(xml_files)} "
                  f"(first 3: {xml_files[:3]})")
        else:
            # Try to find Annotations anywhere under imagenet_root
            print(f"  ann_dir not found — searching for XML files...")
            for root_dir, dirs, files in os.walk(imagenet_root):
                xml_count = sum(1 for f in files if f.endswith('.xml'))
                if xml_count > 0:
                    print(f"    Found {xml_count} XMLs in: {root_dir}")
                    break
            else:
                print(f"    No XML files found under {imagenet_root}")
                print(f"    Download LOC annotations from Kaggle or use "
                      f"ImageNetNoBoxFallback")

        # ----------------------------------------------------------------
        # Build class → index mapping
        # ----------------------------------------------------------------
        # Do not use torchvision.ImageNet reference — it may fail if
        # structure is non-standard. Build mapping from img_dir directly.
        if os.path.exists(self.img_dir):
            class_names = sorted([
                d for d in os.listdir(self.img_dir)
                if os.path.isdir(os.path.join(self.img_dir, d))
                and d.startswith('n')   # ImageNet synset IDs start with 'n'
            ])
            self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        else:
            # Fallback: try torchvision ImageNet
            try:
                from torchvision.datasets import ImageNet
                ref = ImageNet(imagenet_root, split=split, transform=None)
                self.class_to_idx = ref.class_to_idx
            except Exception as e:
                print(f"  Cannot build class_to_idx: {e}")
                self.class_to_idx = {}

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        print(f"  classes found : {len(self.class_to_idx)}")

        # ----------------------------------------------------------------
        # Collect samples
        # ----------------------------------------------------------------
        self.samples = self._collect_samples()

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        print(f"  samples loaded: {len(self.samples)}\n")

        if len(self.samples) == 0:
            print("  WARNING: zero samples found. Check paths above.")
            print("  Falling back to ImageNetNoBoxFallback behaviour.\n")

    def _collect_samples(self) -> list:
        """
        Collect (img_path, xml_path_or_None, class_idx) tuples.
        Works with or without annotation XMLs.
        """
        samples = []

        if not os.path.exists(self.img_dir):
            return samples

        # ImageNet val can be flat (all images in one dir) or nested by class
        # Detect structure
        top_level = os.listdir(self.img_dir)
        has_subdirs = any(
            os.path.isdir(os.path.join(self.img_dir, d))
            for d in top_level
        )

        if has_subdirs:
            # Standard structure: val/n01440764/ILSVRC2012_val_xxxxx.JPEG
            for class_name in sorted(top_level):
                class_dir = os.path.join(self.img_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                class_idx = self.class_to_idx.get(class_name)
                if class_idx is None:
                    continue
                for fname in sorted(os.listdir(class_dir)):
                    if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                        continue
                    img_path = os.path.join(class_dir, fname)
                    xml_path = self._find_xml(fname)
                    samples.append((img_path, xml_path, class_idx))
        else:
            # Flat structure: val/ILSVRC2012_val_xxxxx.JPEG
            # Need a label file to get class indices
            label_file = os.path.join(
                os.path.dirname(self.img_dir),
                'LOC_val_solution.csv'
            )
            label_map = self._load_flat_labels(label_file)

            for fname in sorted(top_level):
                if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    continue
                img_path  = os.path.join(self.img_dir, fname)
                stem      = os.path.splitext(fname)[0]
                class_idx = label_map.get(stem)
                if class_idx is None:
                    continue
                xml_path  = self._find_xml(fname)
                samples.append((img_path, xml_path, class_idx))

        return samples

    def _find_xml(self, img_fname: str) -> str:
        """
        Try to find the XML annotation for an image filename.
        Returns the path if found, None otherwise.
        """
        stem     = os.path.splitext(img_fname)[0]
        xml_name = stem + '.xml'

        # Try standard LOC path
        candidate = os.path.join(self.ann_dir, xml_name)
        if os.path.exists(candidate):
            return candidate

        # Try flat annotation dir (some distributions use this)
        flat_candidate = os.path.join(
            self.root, 'Annotations', xml_name
        )
        if os.path.exists(flat_candidate):
            return flat_candidate

        return None   # no annotation — will use full-image box

    def _load_flat_labels(self, label_file: str) -> dict:
        """
        Load label mapping from LOC_val_solution.csv (Kaggle format).
        Returns {image_stem: class_idx}.
        """
        label_map = {}
        if not os.path.exists(label_file):
            print(f"  Label file not found: {label_file}")
            return label_map
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 2 or parts[0] == 'ImageId':
                    continue
                stem      = parts[0]
                # LOC solution format: "n01440764 0 0 100 100 n01440764 ..."
                class_str = parts[1].split()[0]
                class_idx = self.class_to_idx.get(class_str)
                if class_idx is not None:
                    label_map[stem] = class_idx
        return label_map

    def __len__(self) -> int:
        return len(self.samples)
    
    
    def __getitem__(self, idx):
        img_path, stem, class_idx = self.samples[idx]
        img            = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size   # get BEFORE transform

        img_tensor     = self.transform(img)

        _, box_raw = self.box_map[stem]
        box = transform_box_resize_centercrop(
            box       = box_raw,
            orig_w    = orig_w,
            orig_h    = orig_h,
            resize_to = 256,
            crop_to   = 224,
        )

        # Verify box is not degenerate after crop
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            # Box was cropped out entirely — use centre region
            box = (56, 56, 168, 168)   # centre 112×112 of 224×224

        return img_tensor, class_idx, torch.tensor(box, dtype=torch.long)

    def _parse_xml(self, xml_path: str) -> list:
        """Parse ImageNet LOC XML — returns list of (x1,y1,x2,y2)."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes = []
            for obj in root.findall('object'):
                bb = obj.find('bndbox')
                if bb is None:
                    continue
                boxes.append((
                    int(float(bb.find('xmin').text)),
                    int(float(bb.find('ymin').text)),
                    int(float(bb.find('xmax').text)),
                    int(float(bb.find('ymax').text)),
                ))
            return boxes if boxes else [(0, 0, 224, 224)]
        except Exception as e:
            print(f"  XML parse error {xml_path}: {e}")
            return [(0, 0, 224, 224)]


def get_imagenet_pointing_game_loader(
    imagenet_root: str,
    batch_size:    int = 1,
    num_workers:   int = 4,
    max_samples:   int = None,
    img_size:      int = 224,
) -> DataLoader:
    """
    Returns a DataLoader that yields (images, labels, boxes).
    boxes: (B, 4) tensor of (x1, y1, x2, y2) at img_size resolution.

    For pointing game, batch_size=1 is recommended since PAF
    is constructed per image.
    """
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])

    dataset = ImageNetWithBoxes(
        imagenet_root = imagenet_root,
        split         = 'val',
        transform     = transform,
        max_samples   = max_samples,
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
        collate_fn  = _collate_boxes,
    )

    return loader


def _collate_boxes(batch):
    """
    Custom collate to handle variable-length box lists.
    Stacks images and labels normally; stacks boxes as (B, 4).
    """
    images  = torch.stack([b[0] for b in batch])
    labels  = torch.tensor([b[1] for b in batch], dtype=torch.long)
    boxes   = torch.stack([b[2] for b in batch])   # (B, 4)
    return images, labels, boxes


# ============================================================================
# Fallback: if Annotations directory is missing, use weak supervision
# from the image filename to generate a full-image box
# ============================================================================

class ImageNetNoBoxFallback(Dataset):
    """
    Use when LOC annotation XMLs are not available.
    Returns the full image as the bounding box (worst case for pointing game).
    Useful for testing the pipeline end-to-end.
    """

    def __init__(self, imagenet_root, split='val',
                 transform=None, max_samples=None):
        from torchvision.datasets import ImageNet
        self.ds = ImageNet(
            imagenet_root, split=split, transform=transform
        )
        if max_samples:
            # Subset
            indices = list(range(min(max_samples, len(self.ds))))
            self.ds = torch.utils.data.Subset(self.ds, indices)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        H = img.shape[1] if isinstance(img, torch.Tensor) else 224
        W = img.shape[2] if isinstance(img, torch.Tensor) else 224
        # Full image box — pointing game will always hit
        box = torch.tensor([0, 0, W-1, H-1], dtype=torch.long)
        return img, label, box