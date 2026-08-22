import os
import time
import logging
import argparse
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from mkunet_network import MK_UNet, MK_UNet_ShallowDec
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, AvgMeter, cal_params_flops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== MULTI-CLASS LOSS =====================
def multi_class_loss(pred, mask, num_classes, ce_weight=1.0, dice_weight=1.0):
    """
    Combined CrossEntropy + Dice loss for multi-class segmentation.
    pred: [B, C, H, W] logits
    mask: [B, 1, H, W] or [B, H, W] class indices (long)
    """
    target = mask.squeeze(1).long()  # [B, H, W]
    
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(pred, target, reduction='mean')
    
    # Dice Loss (per-class, then averaged)
    pred_softmax = F.softmax(pred, dim=1)
    dice_loss = 0.0
    valid_classes = 0
    for c in range(num_classes):
        pred_c = pred_softmax[:, c]
        mask_c = (target == c).float()
        # Skip classes not present in this batch
        if mask_c.sum() == 0 and pred_c.sum() < 1e-6:
            continue
        intersection = (pred_c * mask_c).sum()
        union = pred_c.sum() + mask_c.sum()
        dice_loss += 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        valid_classes += 1
    
    if valid_classes > 0:
        dice_loss = dice_loss / valid_classes
    
    return ce_weight * ce_loss + dice_weight * dice_loss


# ===================== METRICS =====================
def compute_multiclass_metrics(pred_mask, gt_mask, num_classes):
    """
    Computes per-class Dice and IoU.
    pred_mask: [H, W] predicted class IDs
    gt_mask: [H, W] ground truth class IDs
    Returns dict of per-class metrics.
    """
    results = {}
    for c in range(num_classes):
        pred_c = (pred_mask == c).float()
        gt_c = (gt_mask == c).float()
        
        smooth = 1e-6
        intersection = (pred_c * gt_c).sum()
        
        dice = (2.0 * intersection + smooth) / (pred_c.sum() + gt_c.sum() + smooth)
        iou = (intersection + smooth) / (pred_c.sum() + gt_c.sum() - intersection + smooth)
        
        results[c] = {'dice': dice.item(), 'iou': iou.item()}
    
    return results


def compute_area_percentages(pred_mask, num_classes):
    """
    Computes the percentage of area occupied by each class.
    pred_mask: [H, W] predicted class IDs
    """
    total_pixels = pred_mask.numel()
    percentages = {}
    for c in range(num_classes):
        count = (pred_mask == c).sum().item()
        percentages[c] = (count / total_pixels) * 100.0
    return percentages


# ===================== TEST / VALIDATION =====================
def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size, 
        shuffle=False, split='test', color_image=opt.color_image,
        num_classes=opt.num_classes,
        preload=opt.preload, num_workers=opt.num_workers,
        pin_memory=opt.pin_memory, persistent_workers=True
    )
    
    all_dice = {c: [] for c in range(opt.num_classes)}
    all_iou = {c: [] for c in range(opt.num_classes)}
    total_images = 0
    
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack       
            images = images.to(device)
            gts = gts.to(device).long()

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress
            
            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # Resize prediction to original size
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_class = torch.argmax(pred_resized.squeeze(0), dim=0)  # [H, W]
                
                # Resize GT to original size
                g = gts[i].unsqueeze(0).float()
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze().long()

                # Per-class metrics
                metrics = compute_multiclass_metrics(pred_class, gt_resized, opt.num_classes)
                for c in range(opt.num_classes):
                    all_dice[c].append(metrics[c]['dice'])
                    all_iou[c].append(metrics[c]['iou'])
                
                total_images += 1

    # Average per-class metrics
    mean_dice = sum(np.mean(all_dice[c]) for c in range(opt.num_classes)) / opt.num_classes
    mean_iou = sum(np.mean(all_iou[c]) for c in range(opt.num_classes)) / opt.num_classes
    
    # Garbage collect validation tensors
    import gc
    if 'images' in locals():
        del images, gts, predictions
    gc.collect()
    torch.cuda.empty_cache()
    
    return mean_dice, mean_iou, total_images


# ===================== TRAIN =====================
def train(train_loader, model, optimizer, epoch, opt, model_name, scaler):
    model.train()
    global best, test_dice_at_best_val, total_train_time, dict_plot
    
    epoch_start = time.time()
    loss_record = AvgMeter()
    size_rates = [0.75, 1.0, 1.25] if opt.multiscale else [1.0]
    total_step = len(train_loader)

    for i, (images, gts) in enumerate(train_loader, start=1):
        images_orig = images.to(device)
        gts_orig = gts.to(device)

        for rate in size_rates:            
            optimizer.zero_grad()
            
            if rate != 1:
                trainsize = int(round(opt.img_size * rate / 32) * 32)
                images_scaled = F.interpolate(images_orig, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts_scaled = F.interpolate(gts_orig.float(), size=(trainsize, trainsize), mode='nearest').long()
            else:
                images_scaled = images_orig
                gts_scaled = gts_orig.long()
            
            # Forward pass under autocast
            with torch.cuda.amp.autocast(enabled=opt.amp):
                out = model(images_scaled)
                seg_pred = out[0] if isinstance(out, list) else out
                loss = multi_class_loss(seg_pred, gts_scaled, opt.num_classes)

                # Classification loss (if model has classification head)
                if isinstance(out, list) and len(out) > 1:
                    cls_pred = out[1]  # [B, num_classes-1]
                    # Derive multi-label targets from segmentation masks
                    target_seg = gts_scaled.squeeze(1).long()  # [B, H, W]
                    cls_target = torch.zeros(target_seg.size(0), opt.num_classes - 1, device=target_seg.device)
                    for c in range(1, opt.num_classes):
                        cls_target[:, c - 1] = (target_seg == c).flatten(1).any(dim=1).float()
                    cls_loss = F.binary_cross_entropy_with_logits(cls_pred, cls_target)
                    loss = loss + 0.5 * cls_loss

            # Backward pass under autocast scaler
            if opt.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_gradient(optimizer, opt.clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                clip_gradient(optimizer, opt.clip)
                optimizer.step()
            
            if rate == 1:
                loss_record.update(loss.item(), opt.batchsize)
                
        if i % 100 == 0 or i == total_step:
            print(f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{total_step:04d}], '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}, Loss: {loss_record.show():.4f}')
        
    total_train_time += (time.time() - epoch_start)
    
    # Save Last
    save_path = opt.train_save
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-last.pth"))

    # Validation and Testing
    epoch_results = {}
    for ds in ['test', 'val']:
        d_dice, d_iou, _ = test(model, opt.test_path, ds, opt)
        epoch_results[ds] = d_dice
        logging.info(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        print(f'Epoch: {epoch}, Dataset: {ds}, Dice: {d_dice:.4f}, IoU: {d_iou:.4f}')
        dict_plot[ds].append(d_dice)

    # Check if Best Validation Dice
    if epoch_results['val'] > best:
        logging.info(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        print(f"### Best Model Saved (Dice improved from {best:.4f} to {epoch_results['val']:.4f}) ###")
        best = epoch_results['val']
        test_dice_at_best_val = epoch_results['test']
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))

    # Garbage collection and cache clearing
    import gc
    del images_orig, gts_orig, loss
    if 'images_scaled' in locals():
        del images_scaled, gts_scaled
    if 'out' in locals():
        del out, seg_pred
    gc.collect()
    torch.cuda.empty_cache()


def save_checkpoint(model, optimizer, scheduler, epoch, best, test_dice_at_best_val,
                    total_train_time, dict_plot, save_path, model_name, drive_path=None):
    """Save full checkpoint so training can be resumed later."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best': best,
        'test_dice_at_best_val': test_dice_at_best_val,
        'total_train_time': total_train_time,
        'dict_plot': dict_plot,
    }
    ckpt_path = os.path.join(save_path, f"{model_name}-checkpoint.pth")
    torch.save(checkpoint, ckpt_path)
    print(f"  💾 Checkpoint saved: epoch {epoch} -> {ckpt_path}")
    
    # Auto-save to Google Drive if path is provided
    if drive_path:
        os.makedirs(drive_path, exist_ok=True)
        for fname in [f"{model_name}-checkpoint.pth", f"{model_name}-best.pth", f"{model_name}-last.pth"]:
            src = os.path.join(save_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(drive_path, fname))
        print(f"  ☁️  Synced to Drive: {drive_path}")
    

if __name__ == '__main__':
    dataset_name = 'Plantation'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet_ShallowDec_L', 
                        choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_ShallowDec', 'MK_UNet_ShallowDec_L'])    
    parser.add_argument('--num_classes', type=int, default=16, help='Number of classes (15 plantation types + 1 background)')
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256, help='Input image size (should match tile size)')
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str, default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path', type=str, default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save', type=str, default='') 
    parser.add_argument('--num_runs', type=int, default=3, help='Number of training runs')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from (e.g., ./model_pth/<RUN_ID>/<RUN_ID>-checkpoint.pth)')
    parser.add_argument('--drive_save_path', type=str, default='', help='Google Drive path to auto-save checkpoints (e.g., /content/drive/MyDrive/MK-UNet/model_pth/)')
    
    # Optimization and speedup arguments
    parser.add_argument('--amp', action='store_true', default=True, help='Enable Mixed Precision training')
    parser.add_argument('--no_amp', action='store_false', dest='amp', help='Disable Mixed Precision training')
    parser.add_argument('--multiscale', action='store_true', help='Enable multi-scale training [0.75, 1.0, 1.25]')
    parser.add_argument('--preload', action='store_true', default=True, help='Preload all dataset images into RAM')
    parser.add_argument('--no_preload', action='store_false', dest='preload', help='Disable Preloading of dataset')
    parser.add_argument('--compile', action='store_true', help='Compile the model using torch.compile (requires PyTorch 2.0+)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='DataLoader pin memory')
    parser.add_argument('--no_pin_memory', action='store_false', dest='pin_memory', help='Disable pin memory')
    
    opt = parser.parse_args()

    # Network configuration mapping
    NET_CONFIGS = {
        'MK_UNet_T':            ([4, 8, 16, 24, 32],       MK_UNet),
        'MK_UNet_S':            ([8, 16, 32, 48, 80],       MK_UNet),
        'MK_UNet':              ([16, 32, 64, 96, 160],     MK_UNet),
        'MK_UNet_ShallowDec':   ([16, 32, 64, 96, 160],     MK_UNet_ShallowDec),
        'MK_UNet_ShallowDec_L': ([64, 128, 256, 384, 512],  MK_UNet_ShallowDec),  # L encoder + shallow decoder
    }

    chosen_net = opt.network
    if chosen_net not in NET_CONFIGS:
        print(f"WARNING: Network '{chosen_net}' not found. Defaulting to 'MK_UNet_ShallowDec_L'.")
        chosen_net = 'MK_UNet_ShallowDec_L'

    # ===================== RESUME MODE =====================
    if opt.resume and os.path.isfile(opt.resume):
        print(f"\n🔄 RESUMING from checkpoint: {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location='cpu')
        start_epoch = checkpoint['epoch'] + 1
        best = checkpoint['best']
        test_dice_at_best_val = checkpoint['test_dice_at_best_val']
        total_train_time = checkpoint['total_train_time']
        dict_plot = checkpoint['dict_plot']

        # Extract run_id from checkpoint filename
        ckpt_fname = os.path.basename(opt.resume)  # e.g., <RUN_ID>-checkpoint.pth
        run_id = ckpt_fname.replace('-checkpoint.pth', '')
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)

        # Build model and load saved weights
        channels, ModelClass = NET_CONFIGS[chosen_net]
        model = ModelClass(num_classes=opt.num_classes, in_channels=3, channels=channels, enable_cls=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if opt.compile:
            print("  ⚡ Compiling model using torch.compile()...")
            model = torch.compile(model)

        scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"  ✅ Resuming from epoch {start_epoch}/{opt.epoch}")
        print(f"  ✅ Best Val Dice so far: {best:.4f}")
        print(f"  ✅ Total train time so far: {total_train_time:.1f}s")

        # Determine drive save path for this run
        drive_run_path = ''
        if opt.drive_save_path:
            drive_run_path = os.path.join(opt.drive_save_path, run_id)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, 
            shuffle=True, augmentation=opt.augmentation, split='train', 
            color_image=opt.color_image, num_classes=opt.num_classes,
            preload=opt.preload, num_workers=opt.num_workers,
            pin_memory=opt.pin_memory, persistent_workers=True
        )

        for epoch in range(start_epoch, opt.epoch + 1):
            train(train_loader, model, optimizer, epoch, opt, run_id, scaler)
            scheduler.step()
            save_checkpoint(model, optimizer, scheduler, epoch, best, test_dice_at_best_val,
                            total_train_time, dict_plot, opt.train_save, run_id, drive_run_path)

        summary = (f"\n{'='*40}\nFINAL RESULTS (RESUMED): {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)

    # ===================== FRESH TRAINING =====================
    else:
        for run in range(1, opt.num_runs + 1):
            dict_plot = {'val': [], 'test': []}
            best = 0.0
            test_dice_at_best_val = 0.0
            total_train_time = 0

            timestamp = time.strftime('%H%M%S')
            run_id = (f"{dataset_name}_{chosen_net}_bs{opt.batchsize}_lr{opt.lr}_"
                          f"e{opt.epoch}_cls{opt.num_classes}_run{run}_t{timestamp}")
            opt.train_save = f'./model_pth/{run_id}/'
            
            os.makedirs('logs', exist_ok=True)
            os.makedirs(opt.train_save, exist_ok=True)
            
            logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                                format='[%(asctime)s] %(message)s', force=True)

            # Build model
            channels, ModelClass = NET_CONFIGS[chosen_net]
            model = ModelClass(num_classes=opt.num_classes, in_channels=3, channels=channels, enable_cls=True)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            if opt.compile:
                print("  ⚡ Compiling model using torch.compile()...")
                model = torch.compile(model)

            scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

            print(f"\n{'='*50}")
            print(f"Run {run}/{opt.num_runs} | Network: {chosen_net} | Channels: {channels}")
            print(f"Classes: {opt.num_classes} | Image Size: {opt.img_size}")
            print(f"{'='*50}")
            
            cal_params_flops(model, opt.img_size, logging)
            optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

            # Determine drive save path for this run
            drive_run_path = ''
            if opt.drive_save_path:
                drive_run_path = os.path.join(opt.drive_save_path, run_id)

            train_loader = get_loader(
                image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
                batchsize=opt.batchsize, trainsize=opt.img_size, 
                shuffle=True, augmentation=opt.augmentation, split='train', 
                color_image=opt.color_image, num_classes=opt.num_classes,
                preload=opt.preload, num_workers=opt.num_workers,
                pin_memory=opt.pin_memory, persistent_workers=True
            )

            for epoch in range(1, opt.epoch + 1):
                train(train_loader, model, optimizer, epoch, opt, run_id, scaler)
                scheduler.step()
                save_checkpoint(model, optimizer, scheduler, epoch, best, test_dice_at_best_val,
                                total_train_time, dict_plot, opt.train_save, run_id, drive_run_path)

            # FINAL SUMMARY
            summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                       f"Best Val Dice: {best:.4f}\n"
                       f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                       f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
            print(summary)
            logging.info(summary)
