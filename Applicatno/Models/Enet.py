"""
Enhanced ENet for Tea Plantation Segmentation
Key Improvements: Focal Loss, Better Augmentation, Class Balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import pandas as pd
from datetime import datetime

import numpy as np
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - IMPROVED
# ============================================================================
class Config:
    # Paths
    TRAIN_IMAGES_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Train_images"
    TRAIN_MASKS_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Train_masks"

    TEST_IMAGES_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Test_images"
    TEST_MASKS_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Test_masks"

    VAL_IMAGES_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Val_images"
    VAL_MASKS_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\Train Test Val Split\Val_masks"

    OUTPUT_DIR = r"D:\STUDIES\Amit Sir\Combined_Patches\ENet\Result\Try5"
    RESUME_CHECKPOINT = r"D:\STUDIES\Amit Sir\Combined_Patches\ENet\Result\Try5\last_enet_model.pth"
    
    # Output subdirectories
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
    PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, 'graphs')
    
    # Model
    NUM_CLASSES = 2
    INPUT_CHANNELS = 1
    
    # Training - IMPROVED
    IMAGE_SIZE = 320
    BATCH_SIZE = 32  # Slightly smaller for larger crops
    NUM_EPOCHS = 250
    LEARNING_RATE = 8e-4  # Slightly higher
    NUM_WORKERS = 4
    
    # Loss weights - REBALANCED for better Dice
    CE_WEIGHT = 0.3  # Reduced CE
    DICE_WEIGHT = 0.5  # Increased Dice focus
    IOU_WEIGHT = 0.2
    FOCAL_GAMMA = 2.0  # For hard examples
    FOCAL_ALPHA = 0.75  # Class balance
    
    # Scheduler
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # Early stopping - MORE PATIENT
    PATIENCE = 300
    
    # Other
    SEED = 42
    CHECKPOINT_FREQ = 10
    NUM_VIS_SAMPLES = 15
    
    # Class weights
    MIN_CLASS_WEIGHT = 0.5
    MAX_CLASS_WEIGHT = 15.0  # Allow higher weights

    EXCEL_LOG = os.path.join(OUTPUT_DIR, 'training_log.xlsx')

# ============================================================================
# SEED
# ============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

# ============================================================================
# ENET BUILDING BLOCKS (Same as before)
# ============================================================================
class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, 3, 2, 1, bias=False)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        main = self.conv(x)
        side = self.maxpool(x)
        x = torch.cat([main, side], dim=1)
        x = self.bn(x)
        return self.prelu(x)

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(DownsamplingBottleneck, self).__init__()
        internal_channels = in_channels // 4
        
        self.main_conv1 = nn.Conv2d(in_channels, internal_channels, 2, 2, bias=False)
        self.main_bn1 = nn.BatchNorm2d(internal_channels)
        self.main_prelu1 = nn.PReLU()
        
        self.main_conv2 = nn.Conv2d(internal_channels, internal_channels, 3, 1, 1, bias=False)
        self.main_bn2 = nn.BatchNorm2d(internal_channels)
        self.main_prelu2 = nn.PReLU()
        
        self.main_conv3 = nn.Conv2d(internal_channels, out_channels, 1, bias=False)
        self.main_bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_prob)
        
        self.side_maxpool = nn.MaxPool2d(2, 2, return_indices=True)
        self.side_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.side_bn = nn.BatchNorm2d(out_channels)
        
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        main = self.main_conv1(x)
        main = self.main_bn1(main)
        main = self.main_prelu1(main)
        
        main = self.main_conv2(main)
        main = self.main_bn2(main)
        main = self.main_prelu2(main)
        
        main = self.main_conv3(main)
        main = self.main_bn3(main)
        main = self.dropout(main)
        
        side, indices = self.side_maxpool(x)
        side = self.side_conv(side)
        side = self.side_bn(side)
        
        out = main + side
        return self.prelu(out), indices

class RegularBottleneck(nn.Module):
    def __init__(self, channels, dropout_prob=0.0, dilation=1):
        super(RegularBottleneck, self).__init__()
        internal_channels = channels // 4
        
        self.conv1 = nn.Conv2d(channels, internal_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.prelu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(internal_channels, internal_channels, 3, 1, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(internal_channels)
        self.prelu2 = nn.PReLU()
        
        self.conv3 = nn.Conv2d(internal_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout2d(dropout_prob)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        main = self.conv1(x)
        main = self.bn1(main)
        main = self.prelu1(main)
        
        main = self.conv2(main)
        main = self.bn2(main)
        main = self.prelu2(main)
        
        main = self.conv3(main)
        main = self.bn3(main)
        main = self.dropout(main)
        
        return self.prelu(main + x)

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.0):
        super(UpsamplingBottleneck, self).__init__()
        internal_channels = in_channels // 4
        
        self.main_conv1 = nn.Conv2d(in_channels, internal_channels, 1, bias=False)
        self.main_bn1 = nn.BatchNorm2d(internal_channels)
        self.main_prelu1 = nn.PReLU()
        
        self.main_deconv = nn.ConvTranspose2d(internal_channels, internal_channels, 3, 2, 1, 1, bias=False)
        self.main_bn2 = nn.BatchNorm2d(internal_channels)
        self.main_prelu2 = nn.PReLU()
        
        self.main_conv3 = nn.Conv2d(internal_channels, out_channels, 1, bias=False)
        self.main_bn3 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_prob)
        
        self.side_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.side_bn = nn.BatchNorm2d(out_channels)
        self.side_unpool = nn.MaxUnpool2d(2)
        
        self.prelu = nn.PReLU()
    
    def forward(self, x, indices):
        main = self.main_conv1(x)
        main = self.main_bn1(main)
        main = self.main_prelu1(main)
        
        main = self.main_deconv(main)
        main = self.main_bn2(main)
        main = self.main_prelu2(main)
        
        main = self.main_conv3(main)
        main = self.main_bn3(main)
        main = self.dropout(main)
        
        side = self.side_conv(x)
        side = self.side_bn(side)
        side = self.side_unpool(side, indices)
        
        out = main + side
        return self.prelu(out)

# ============================================================================
# ENET MODEL (Same)
# ============================================================================
class ENet(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super(ENet, self).__init__()
        
        self.initial = InitialBlock(in_channels, 16)
        
        self.downsample1_0 = DownsamplingBottleneck(16, 64, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, dropout_prob=0.01)
        
        self.downsample2_0 = DownsamplingBottleneck(64, 128, dropout_prob=0.1)
        self.regular2_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_2 = RegularBottleneck(128, dropout_prob=0.1, dilation=2)
        self.regular2_3 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_4 = RegularBottleneck(128, dropout_prob=0.1, dilation=4)
        self.regular2_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_6 = RegularBottleneck(128, dropout_prob=0.1, dilation=8)
        self.regular2_7 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_8 = RegularBottleneck(128, dropout_prob=0.1, dilation=16)
        
        self.regular3_0 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_1 = RegularBottleneck(128, dropout_prob=0.1, dilation=2)
        self.regular3_2 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_3 = RegularBottleneck(128, dropout_prob=0.1, dilation=4)
        self.regular3_4 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_5 = RegularBottleneck(128, dropout_prob=0.1, dilation=8)
        self.regular3_6 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_7 = RegularBottleneck(128, dropout_prob=0.1, dilation=16)
        
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4_1 = RegularBottleneck(64, dropout_prob=0.1)
        self.regular4_2 = RegularBottleneck(64, dropout_prob=0.1)
        
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5_1 = RegularBottleneck(16, dropout_prob=0.1)
        
        self.deconv = nn.ConvTranspose2d(16, num_classes, 2, 2, bias=False)
    
    def forward(self, x):
        x = self.initial(x)
        
        x, indices1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        
        x, indices2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.regular2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.regular2_7(x)
        x = self.dilated2_8(x)
        
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.regular3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.regular3_6(x)
        x = self.dilated3_7(x)
        
        x = self.upsample4_0(x, indices2)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        
        x = self.upsample5_0(x, indices1)
        x = self.regular5_1(x)
        
        x = self.deconv(x)
        
        return x

# ============================================================================
# IMPROVED DATASET - WITH SAMPLE WEIGHTS
# ============================================================================
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, return_weights=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.return_weights = return_weights
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        
        self.image_files.sort()
        
        self.valid_pairs = []
        self.sample_weights = []
        
        for img_path in self.image_files:
            basename = os.path.basename(img_path)
            mask_path = os.path.join(masks_dir, basename)
            
            if os.path.exists(mask_path):
                self.valid_pairs.append((img_path, mask_path))
                
                # Calculate weight based on tea pixels
                if return_weights:
                    mask = np.array(Image.open(mask_path))
                    tea_pixels = np.isin(mask, [1,8,9,10]).sum()
                    total_pixels = mask.size
                    tea_ratio = tea_pixels / total_pixels
                    # Higher weight for images with more tea
                    weight = 1.0 + tea_ratio * 2.0
                    self.sample_weights.append(weight)
        
        print(f"Found {len(self.valid_pairs)} valid image-mask pairs in {images_dir}")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        image = np.array(Image.open(img_path).convert('L'))
        mask = np.array(Image.open(mask_path))
        
        tea_mask = np.isin(mask, [8,9]).astype(np.uint8)
        
        if self.transform:
            transformed = self.transform(image=image, mask=tea_mask)
            image = transformed['image']
            tea_mask = transformed['mask']
        
        return image, tea_mask.long(), os.path.basename(img_path)

# ============================================================================
# IMPROVED AUGMENTATION
# ============================================================================
def get_train_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        
        # Intensity (Tea-specific)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        
        # Normalize
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

def get_val_transforms(image_size=512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2()
    ])

# ============================================================================
# IMPROVED LOSS - FOCAL + DICE + IOU
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.75, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3)) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
    
    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        iou = self.iou_loss(pred, target)
        
        total = Config.CE_WEIGHT * focal + Config.DICE_WEIGHT * dice + Config.IOU_WEIGHT * iou
        return total, focal, dice, iou

# ============================================================================
# METRICS (Same)
# ============================================================================
def calculate_metrics(pred, target, num_classes=2):
    pred = torch.argmax(pred, dim=1)
    accuracy = (pred == target).float().mean().item()
    
    dice_scores = []
    iou_scores = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).float().sum().item()
        pred_sum = pred_cls.float().sum().item()
        target_sum = target_cls.float().sum().item()
        union = pred_sum + target_sum - intersection
        
        dice = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        dice_scores.append(dice)
        iou_scores.append(iou)
    
    return accuracy, dice_scores, iou_scores

# ============================================================================
# TRAINING FUNCTIONS (Updated)
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    all_dice_scores = []
    all_iou_scores = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss, focal, dice_loss, iou_loss = criterion(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        accuracy, dice_scores, iou_scores = calculate_metrics(outputs, masks)
        
        running_loss += loss.item()
        running_accuracy += accuracy
        all_dice_scores.append(dice_scores)
        all_iou_scores.append(iou_scores)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'accuracy': running_accuracy / num_batches,
        'dice_scores': np.mean(all_dice_scores, axis=0),
        'iou_scores': np.mean(all_iou_scores, axis=0)
    }

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    all_dice_scores = []
    all_iou_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            with autocast():
                outputs = model(images)
                loss, _, _, _ = criterion(outputs, masks)
            
            accuracy, dice_scores, iou_scores = calculate_metrics(outputs, masks)
            
            running_loss += loss.item()
            running_accuracy += accuracy
            all_dice_scores.append(dice_scores)
            all_iou_scores.append(iou_scores)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    num_batches = len(dataloader)
    return {
        'loss': running_loss / num_batches,
        'accuracy': running_accuracy / num_batches,
        'dice_scores': np.mean(all_dice_scores, axis=0),
        'iou_scores': np.mean(all_iou_scores, axis=0)
    }

# ============================================================================
# VISUALIZATION (Same as before)
# ============================================================================
def visualize_predictions(model, dataloader, device, save_dir, num_samples=15):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    colors = np.array([[128, 128, 128], [34, 139, 34]], dtype=np.uint8)
    samples_saved = 0
    
    with torch.no_grad():
        for images, masks, filenames in dataloader:
            if samples_saved >= num_samples:
                break
            
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            
            for i in range(len(images)):
                if samples_saved >= num_samples:
                    break
                
                img = (images[i, 0] * 0.5 + 0.5) * 255
                img = img.astype(np.uint8)
                mask_rgb = colors[masks[i]]
                pred_rgb = colors[preds[i]]
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                
                axes[0, 0].imshow(img, cmap='gray')
                axes[0, 0].set_title('Original', fontsize=12, fontweight='bold')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(img, cmap='gray')
                axes[0, 1].imshow(mask_rgb, alpha=0.5)
                axes[0, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[0, 1].axis('off')
                
                axes[1, 0].imshow(img, cmap='gray')
                axes[1, 0].imshow(pred_rgb, alpha=0.5)
                axes[1, 0].set_title('Prediction', fontsize=12, fontweight='bold')
                axes[1, 0].axis('off')
                
                axes[1, 1].imshow(img, cmap='gray')
                axes[1, 1].contour(masks[i], levels=[0.5], colors='red', linewidths=2)
                axes[1, 1].imshow(pred_rgb, alpha=0.4)
                axes[1, 1].set_title('Combined', fontsize=12, fontweight='bold')
                axes[1, 1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'pred_{samples_saved:03d}.png'), dpi=150)
                plt.close()
                
                samples_saved += 1

def plot_history(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['val_dice'], 'g-', label='Dice', linewidth=2)
    axes[2].plot(epochs, history['val_iou'], 'm-', label='IoU', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Metrics')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'history.png'), dpi=300)
    plt.close()

def update_excel_log(epoch, train_metrics, val_metrics, lr, save_path):
    """Update Excel file after each epoch"""
    
    # Create new row
    new_row = {
        'Epoch': epoch,
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Learning_Rate': lr,
        'Train_Loss': train_metrics['loss'],
        'Val_Loss': val_metrics['loss'],
        'Train_Accuracy': train_metrics['accuracy'],
        'Val_Accuracy': val_metrics['accuracy'],
        'Train_Dice_BG': train_metrics['dice_scores'][0],
        'Train_Dice_Tea': train_metrics['dice_scores'][1],
        'Train_Mean_Dice': np.mean(train_metrics['dice_scores']),
        'Val_Dice_BG': val_metrics['dice_scores'][0],
        'Val_Dice_Tea': val_metrics['dice_scores'][1],
        'Val_Mean_Dice': np.mean(val_metrics['dice_scores']),
        'Train_IoU_BG': train_metrics['iou_scores'][0],
        'Train_IoU_Tea': train_metrics['iou_scores'][1],
        'Train_Mean_IoU': np.mean(train_metrics['iou_scores']),
        'Val_IoU_BG': val_metrics['iou_scores'][0],
        'Val_IoU_Tea': val_metrics['iou_scores'][1],
        'Val_Mean_IoU': np.mean(val_metrics['iou_scores'])
    }
    
    # Load existing or create new
    if os.path.exists(save_path):
        df = pd.read_excel(save_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    # Save
    df.to_excel(save_path, index=False, engine='openpyxl')

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*80)
    print("Enhanced ENet Tea Segmentation - WITH IMPROVEMENTS")
    print("="*80)
    
    # Create directories
    for d in [Config.OUTPUT_DIR, Config.CHECKPOINT_DIR, Config.PREDICTIONS_DIR, Config.GRAPHS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = SegmentationDataset(
        Config.TRAIN_IMAGES_DIR,
        Config.TRAIN_MASKS_DIR,
        transform=get_train_transforms(Config.IMAGE_SIZE),
        return_weights=True
    )
    
    val_dataset = SegmentationDataset(
        Config.VAL_IMAGES_DIR,
        Config.VAL_MASKS_DIR,
        transform=get_val_transforms(Config.IMAGE_SIZE)
    )

    test_dataset = SegmentationDataset(
        Config.TEST_IMAGES_DIR,
        Config.TEST_MASKS_DIR,
        transform=get_val_transforms(Config.IMAGE_SIZE)  # Same as val
    )
    
    # Create dataloaders with weighted sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing ENet...")
    model = ENet(num_classes=Config.NUM_CLASSES, in_channels=Config.INPUT_CHANNELS)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    
    # Scheduler
    def warmup_lambda(epoch):
        if epoch < Config.WARMUP_EPOCHS:
            return epoch / Config.WARMUP_EPOCHS
        return 1.0
    
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS - Config.WARMUP_EPOCHS, eta_min=Config.MIN_LR)
    
    # Mixed precision
    scaler = GradScaler()
    
    # History
    history = {
        'train_loss': [], 'train_accuracy': [], 'train_dice': [], 'train_iou': [],
        'val_loss': [], 'val_accuracy': [], 'val_dice': [], 'val_iou': []
    }
    
    best_dice = 0.0
    patience_counter = 0
    start_epoch = 1
    
    # Resume from checkpoint
    if Config.RESUME_CHECKPOINT and os.path.exists(Config.RESUME_CHECKPOINT):
        print(f"\nResuming from: {Config.RESUME_CHECKPOINT}")
        checkpoint = torch.load(Config.RESUME_CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        history = checkpoint.get('history', history)
        print(f"Resuming from epoch {start_epoch}, Best Dice: {best_dice:.4f}\n")
    
    print("="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    # Training loop
    for epoch in range(start_epoch, Config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{Config.NUM_EPOCHS}")
        print("-" * 80)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        if epoch <= Config.WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_dice'].append(train_metrics['dice_scores'][1])
        history['train_iou'].append(train_metrics['iou_scores'][1])
        
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_dice'].append(val_metrics['dice_scores'][1])
        history['val_iou'].append(val_metrics['iou_scores'][1])

        mean_val_iou = np.mean(val_metrics['iou_scores'])  # Calculate mean
        
        # Print summary
        print("\n" + "="*80)
        print(f"EPOCH {epoch} SUMMARY")
        print("="*80)
        print(f"{'Metric':<25} {'Train':<15} {'Validation':<15}")
        print("-"*80)
        print(f"{'Loss':<25} {train_metrics['loss']:<15.4f} {val_metrics['loss']:<15.4f}")
        print(f"{'Accuracy':<25} {train_metrics['accuracy']:<15.4f} {val_metrics['accuracy']:<15.4f}")
        print(f"{'Dice (Tea)':<25} {train_metrics['dice_scores'][1]:<15.4f} {val_metrics['dice_scores'][1]:<15.4f}")
        print(f"{'Mean Dice':<25} {np.mean(train_metrics['dice_scores']):<15.4f} {np.mean(val_metrics['dice_scores']):<15.4f}")
        print(f"{'IoU (Tea)':<25} {train_metrics['iou_scores'][1]:<15.4f} {val_metrics['iou_scores'][1]:<15.4f}")
        print(f"{'Mean IoU':<25} {np.mean(train_metrics['iou_scores']):<15.4f} {np.mean(val_metrics['iou_scores']):<15.4f}")
        print(f"{'Learning Rate':<25} {current_lr:<15.6f}")
        print("="*80)

        update_excel_log(epoch, train_metrics, val_metrics, current_lr, Config.EXCEL_LOG)
        
        # Save checkpoint
        if epoch % Config.CHECKPOINT_FREQ == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_dice': best_dice
            }, checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Save best model
        current_dice = val_metrics['dice_scores'][1]
        if current_dice > best_dice:
            best_dice = current_dice
            patience_counter = 0
            
            best_path = os.path.join(Config.OUTPUT_DIR, 'best_enet_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_dice': best_dice
            }, best_path)
            print(f"\n✓ New best model saved! Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            print(f"\nNo improvement. Patience: {patience_counter}/{Config.PATIENCE}")
        
        # Early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

        # Save final checkpoint
        final_checkpoint = os.path.join(Config.OUTPUT_DIR, 'last_enet_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_dice': best_dice
        }, final_checkpoint)
        print(f"\nFinal checkpoint saved: {final_checkpoint}")

    # Final evaluation
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best Validation Dice: {best_dice:.4f}")
    print(f"Total Epochs: {epoch}")
    print("="*80 + "\n")
    # Load best model
    print("Loading best model for evaluation...")
    checkpoint = torch.load(os.path.join(Config.OUTPUT_DIR, 'best_enet_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final TEST evaluation (changed from validation)
    print("\nEvaluating on TEST set...")
    final_metrics = validate_epoch(model, test_loader, criterion, device)  # Changed val_loader to test_loader

    print("\n" + "="*80)
    print("FINAL TEST METRICS")  # Changed title
    print("="*80)

    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Dice (Background): {final_metrics['dice_scores'][0]:.4f}")
    print(f"Dice (Tea): {final_metrics['dice_scores'][1]:.4f}")
    print(f"Mean Dice: {np.mean(final_metrics['dice_scores']):.4f}")
    print(f"IoU (Background): {final_metrics['iou_scores'][0]:.4f}")
    print(f"IoU (Tea): {final_metrics['iou_scores'][1]:.4f}")
    print(f"Mean IoU: {np.mean(final_metrics['iou_scores']):.4f}")
    print("="*80 + "\n")

    # Log final test metrics to Excel
    print("\nSaving final test metrics to Excel...")
    test_row = {
        'Epoch': 'FINAL_TEST',
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Test_Loss': final_metrics['loss'],
        'Test_Accuracy': final_metrics['accuracy'],
        'Test_Dice_BG': final_metrics['dice_scores'][0],
        'Test_Dice_Tea': final_metrics['dice_scores'][1],
        'Test_IoU_BG': final_metrics['iou_scores'][0],
        'Test_IoU_Tea': final_metrics['iou_scores'][1],
        'Train_Mean_IoU': np.mean(train_metrics['iou_scores']),
        'Val_Mean_IoU': np.mean(val_metrics['iou_scores'])
    }

    if os.path.exists(Config.EXCEL_LOG):
        df = pd.read_excel(Config.EXCEL_LOG)
        df = pd.concat([df, pd.DataFrame([test_row])], ignore_index=True)
        df.to_excel(Config.EXCEL_LOG, index=False, engine='openpyxl')
    
    # Generate visualizations
    print(f"Generating {Config.NUM_VIS_SAMPLES} visualizations...")
    visualize_predictions(model, test_loader, device, Config.PREDICTIONS_DIR, Config.NUM_VIS_SAMPLES)  # Changed val_loader to test_loader
    
    # Plot history
    print("\nPlotting training history...")
    plot_history(history, Config.GRAPHS_DIR)
    
    print("\n" + "="*80)
    print("ALL OUTPUTS SAVED")
    print("="*80)
    print(f"Best Model: {Config.OUTPUT_DIR}/best_enet_model.pth")
    print(f"Checkpoints: {Config.CHECKPOINT_DIR}/")
    print(f"Predictions: {Config.PREDICTIONS_DIR}/")
    print(f"Graphs: {Config.GRAPHS_DIR}/")
    print("="*80 + "\n")
    
    print("✓ Training completed successfully!")

if __name__ == '__main__':
    main()