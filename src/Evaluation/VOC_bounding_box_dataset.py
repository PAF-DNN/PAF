from typing import Tuple

from torchvision.datasets import VOCDetection
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from Evaluation.bounding_box_dataset import _collate_boxes

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

class VOCWithBoxes(Dataset):
    """
    PASCAL VOC 2012 with bounding boxes for pointing game.
    Returns (image, label, box) matching ImageNet label space.
    """

    # VOC class name → ImageNet class index mapping
    # ResNet18 trained on ImageNet uses these indices
    VOC_TO_IMAGENET = {
        'aeroplane'  : 404,   # airliner
        'bicycle'    : 671,   # mountain bike
        'bird'       : 80,    # jay (generic bird)
        'boat'       : 628,   # lifeboat
        'bottle'     : 898,   # water bottle
        'bus'        : 779,   # school bus
        'car'        : 817,   # sports car
        'cat'        : 281,   # tabby cat
        'chair'      : 559,   # folding chair
        'cow'        : 345,   # ox
        'diningtable': 532,   # dining table
        'dog'        : 208,   # golden retriever
        'horse'      : 603,   # horse cart
        'motorbike'  : 670,   # moped
        'person'     : 980,   # scuba diver (closest person class)
        'pottedplant': 727,   # pot
        'sheep'      : 348,   # ram
        'sofa'       : 849,   # studio couch
        'train'      : 820,   # freight car
        'tvmonitor'  : 782,   # screen
    }

    def __init__(
        self,
        root:        str,
        transform          = None,
        max_samples: int   = None,
    ):
        self.voc = VOCDetection(
            root      = root,
            year      = '2012',
            image_set = 'val',
            download  = True,
            transform = None,   # apply manually to get orig size
        )
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])

        # Build sample list — one sample per dominant object
        self.samples = []
        for idx in range(len(self.voc)):
            _, target = self.voc[idx]
            objects   = target['annotation']['object']
            if not isinstance(objects, list):
                objects = [objects]
            # Take the largest object as the target
            best_obj  = max(objects, key=lambda o: (
                (int(o['bndbox']['xmax']) - int(o['bndbox']['xmin'])) *
                (int(o['bndbox']['ymax']) - int(o['bndbox']['ymin']))
            ))
            class_name = best_obj['name']
            if class_name not in self.VOC_TO_IMAGENET:
                continue
            self.samples.append((idx, class_name, best_obj['bndbox']))

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"VOCWithBoxes: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        voc_idx, class_name, bndbox = self.samples[idx]
        img, target = self.voc[voc_idx]

        orig_w, orig_h = img.size
        img_tensor     = self.transform(img)
        class_idx      = self.VOC_TO_IMAGENET[class_name]

        box_raw = (
            int(bndbox['xmin']),
            int(bndbox['ymin']),
            int(bndbox['xmax']),
            int(bndbox['ymax']),
        )
        box = transform_box_resize_centercrop(
            box_raw, orig_w, orig_h
        )
        return img_tensor, class_idx, torch.tensor(box, dtype=torch.long)


def get_voc_pointing_game_loader(
    root:        str,
    batch_size:  int = 1,
    num_workers: int = 0,
    max_samples: int = 500,
) -> DataLoader:
    dataset = VOCWithBoxes(root=root, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
        collate_fn  = _collate_boxes,
    )