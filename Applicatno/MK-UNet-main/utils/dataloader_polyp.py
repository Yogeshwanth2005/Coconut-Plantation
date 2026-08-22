import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PolypDataset(data.Dataset):
    """
    Unified adaptive dataloader for segmentation.
    Supports both binary (num_classes=1) and multi-class segmentation.
    Uses Albumentations for augmentation.
    """
    def __init__(self, image_root, gt_root, trainsize, augmentation, split='train', 
                 color_image=True, num_classes=1, preload=False):
        self.trainsize = trainsize
        self.color_image = color_image
        self.augmentation = augmentation
        self.split = split
        self.num_classes = num_classes
        self.preload = preload
        
        # Load and sort file paths
        exts = ('.jpg', '.png', '.jpeg', '.tif')
        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(exts)])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(exts)])
        
        self.filter_files()
        self.size = len(self.images)

        # Transformation Setup
        mean = [0.485, 0.456, 0.406] if color_image else [0.5]
        std = [0.229, 0.224, 0.225] if color_image else [0.229]

        if self.split == 'train' and self.augmentation:
            self.transform = A.Compose([\
                A.Rotate(limit=90, p=0.5),
                A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=self.trainsize, width=self.trainsize),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])

        # Preloading/Caching in RAM
        if self.preload:
            print(f"[{self.split.upper()}] Preloading {self.size} images/masks into RAM...")
            self.cached_images = []
            self.cached_masks = []
            if self.split != 'train':
                self.cached_original_shapes = []
                self.cached_names = []
            
            for idx in range(self.size):
                image = cv2.imread(self.images[idx])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY)
                mask_np = cv2.imread(self.gts[idx], cv2.IMREAD_GRAYSCALE)
                
                self.cached_images.append(image)
                self.cached_masks.append(mask_np)
                
                if self.split != 'train':
                    with Image.open(self.gts[idx]) as img:
                        original_shape = img.size
                    name = os.path.basename(self.images[idx])
                    if name.lower().endswith('.jpg'):
                        name = name.rsplit('.', 1)[0] + '.png'
                    self.cached_original_shapes.append(original_shape)
                    self.cached_names.append(name)
            print(f"[{self.split.upper()}] Preloading completed successfully!")

    def filter_files(self):
        valid_images, valid_gts = [], []
        for img_p, gt_p in zip(self.images, self.gts):
            if os.path.exists(img_p) and os.path.exists(gt_p):
                valid_images.append(img_p)
                valid_gts.append(gt_p)
        self.images, self.gts = valid_images, valid_gts

    def __getitem__(self, index):
        # 1. Load/Retrieve Image and Mask
        if self.preload:
            image = self.cached_images[index]
            mask_np = self.cached_masks[index]
        else:
            image = cv2.imread(self.images[index])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB if self.color_image else cv2.COLOR_BGR2GRAY)
            mask_np = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)

        # 2. Apply Transformations
        augmented = self.transform(image=image, mask=mask_np)
        image = augmented['image']
        mask = augmented['mask']

        # 3. Mask Processing
        if self.num_classes > 1:
            mask = mask.long().clamp(0, self.num_classes - 1)
        else:
            max_val = mask.max()
            if max_val > 127.0:
                mask = (mask > 20).long()
            else:
                mask = (mask >= 1).long()

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        # 4. Return Logic
        if self.split == 'train':
            return image, mask
        else:
            if self.preload:
                original_shape = self.cached_original_shapes[index]
                name = self.cached_names[index]
            else:
                with Image.open(self.gts[index]) as img:
                    original_shape = img.size
                name = os.path.basename(self.images[index])
                if name.lower().endswith('.jpg'):
                    name = name.rsplit('.', 1)[0] + '.png'
            return image, mask, original_shape, name

    def __len__(self):
        return self.size

def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=False, num_workers=4, 
               pin_memory=True, augmentation=False, split='train', color_image=True, num_classes=1,
               preload=False, persistent_workers=False):
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation, split, color_image, num_classes, preload=preload)
    pw = persistent_workers if num_workers > 0 else False
    return data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle, 
                           num_workers=num_workers, pin_memory=pin_memory, persistent_workers=pw)