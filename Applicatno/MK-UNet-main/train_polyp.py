import os
import time
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

# Project-specific imports
from mkunet_network import MK_UNet
from utils.dataloader_polyp import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter, cal_params_flops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def structure_loss(pred, mask, w=1):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (w * (wbce + wiou)).mean()

def dice_coefficient(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    total = predicted_flat.sum() + labels_flat.sum()
    return (2. * intersection + smooth) / (total + smooth)

def iou(predicted, labels):
    if predicted.device != labels.device:
        labels = labels.to(predicted.device)
    smooth = 1e-6
    predicted_flat = predicted.contiguous().view(-1)
    labels_flat = labels.contiguous().view(-1)
    intersection = (predicted_flat * labels_flat).sum()
    union = predicted_flat.sum() + labels_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def test(model, path, dataset, opt):
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image,
        num_classes=1, preload=opt.preload, num_workers=opt.num_workers,
        pin_memory=opt.pin_memory, persistent_workers=True
    )
    
    DSC, IOU, total_images = 0.0, 0.0, 0
    with torch.no_grad():
        for pack in test_loader:
            images, gts, original_shapes, _ = pack       
            images = images.to(device)
            gts = gts.to(device).float()

            ress = model(images)
            # Take the primary output
            predictions = ress[0] if isinstance(ress, list) else ress
            
            for i in range(len(images)):
                # Note: original_shapes in some loaders is [W, H], in others [H, W]
                # We ensure it matches your specific data loader's return order
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Prediction Resize (Bilinear for soft maps)
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_resized = pred_resized.sigmoid().squeeze()
                
                # 2. Local Normalization
                pred_resized = (pred_resized - pred_resized.min()) / (pred_resized.max() - pred_resized.min() + 1e-8)
                
                # 3. GT Resize (NEAREST to maintain binary mask integrity)
                g = gts[i].unsqueeze(0)
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze()

                #print(pred_resized.shape, gt_resized.shape, g.shape)

                # 4. Binary Thresholding
                input_binary = (pred_resized >= 0.5).float()
                target_binary = (gt_resized >= 0.2).float() 
 
                total_images += 1

                DSC += dice_coefficient(input_binary, target_binary).item()
                IOU += iou(input_binary, target_binary).item()

    # Garbage collect validation tensors
    import gc
    if 'images' in locals():
        del images, gts, predictions
    gc.collect()
    torch.cuda.empty_cache()

    return DSC / total_images, IOU / total_images, total_images

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
                gts_scaled = F.interpolate(gts_orig, size=(trainsize, trainsize), mode='nearest')
            else:
                images_scaled = images_orig
                gts_scaled = gts_orig
            
            # Forward pass under autocast
            with torch.cuda.amp.autocast(enabled=opt.amp):
                out = model(images_scaled)
                loss = structure_loss(out[0] if isinstance(out, list) else out, gts_scaled)
            
            # Backward pass under scaler
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
        test_dice_at_best_val = epoch_results['test'] # Track test dice at peak val
        torch.save(model.state_dict(), os.path.join(save_path, f"{model_name}-best.pth"))

    # Garbage collection and cache clearing
    import gc
    del images_orig, gts_orig, loss
    if 'images_scaled' in locals():
        del images_scaled, gts_scaled
    if 'out' in locals():
        del out
    gc.collect()
    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    # Initial defaults
    dataset_name = 'ClinicDB' #'ColonDB'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='MK_UNet', 
                        choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_M', 'MK_UNet_L'])    
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.0005) # base learning rate is 0.0005 for CosineAnnealingLR and 0.0001 for no scheduler
    parser.add_argument('--batchsize', type=int, default=8)
    parser.add_argument('--test_batchsize', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--decay_rate', type=float, default=0.1)
    parser.add_argument('--decay_epoch', type=int, default=300)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--augmentation', default=True)
    parser.add_argument('--train_path', type=str, default=f'./data/polyp/target/{dataset_name}/train/')
    parser.add_argument('--test_path', type=str, default=f'./data/polyp/target/{dataset_name}/')
    parser.add_argument('--train_save', type=str, default='') 
    
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
        'MK_UNet_T': [4, 8, 16, 24, 32],
        'MK_UNet_S': [8, 16, 32, 48, 80],
        'MK_UNet':   [16, 32, 64, 96, 160],
        'MK_UNet_M': [32, 64, 128, 192, 320],
        'MK_UNet_L': [64, 128, 256, 384, 512]
    }

    # Handling Spelling Mistakes or Invalid Choices
    # We use a case-insensitive match or check if it exists in our dictionary
    chosen_net = opt.network
    if chosen_net not in NET_CONFIGS:
        print(f"WARNING: Network '{chosen_net}' not found. Defaulting to 'MK_UNet'.")
        chosen_net = 'MK_UNet'

    for run in [1,2,3,4,5]:
        dict_plot = {'val': [], 'test': []}
        best = 0.0
        test_dice_at_best_val = 0.0
        total_train_time = 0

        timestamp = time.strftime('%H%M%S')
        run_id = (f"{dataset_name}_{chosen_net}_bs{opt.batchsize}_lr{opt.lr}_"
                      f"e{opt.epoch}_aug{opt.augmentation}_run{run}_t{timestamp}")
        opt.train_save = f'./model_pth/{run_id}/'
        
        os.makedirs('logs', exist_ok=True)
        os.makedirs(opt.train_save, exist_ok=True)
        
        logging.basicConfig(filename=f'logs/train_log_{run_id}.log', level=logging.INFO, 
                            format='[%(asctime)s] %(message)s', force=True)

        if opt.network != chosen_net:
            logging.warning(f"User input '{opt.network}' was invalid. Fallback to '{chosen_net}' used.")

        # Build model
        channels = NET_CONFIGS[chosen_net]
        model = MK_UNet(num_classes=1, in_channels=3, channels=channels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        if opt.compile:
            print("  ⚡ Compiling model using torch.compile()...")
            model = torch.compile(model)

        scaler = torch.cuda.amp.GradScaler(enabled=opt.amp)

        print(f"Network: {chosen_net} | Channels: {channels}")
        cal_params_flops(model, opt.img_size, logging)
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.epoch, eta_min=1e-6)

        train_loader = get_loader(
            image_root=f'{opt.train_path}/images/', gt_root=f'{opt.train_path}/masks/',
            batchsize=opt.batchsize, trainsize=opt.img_size, 
            shuffle=True, augmentation=opt.augmentation, split='train', color_image=opt.color_image,
            preload=opt.preload, num_workers=opt.num_workers,
            pin_memory=opt.pin_memory, persistent_workers=True
        )

        for epoch in range(1, opt.epoch + 1):
            #adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train(train_loader, model, optimizer, epoch, opt, run_id, scaler)
            scheduler.step()
        # FINAL SUMMARY
        
        summary = (f"\n{'='*40}\nFINAL RESULTS: {run_id}\n"
                   f"Best Val Dice: {best:.4f}\n"
                   f"Test Dice at Best Val: {test_dice_at_best_val:.4f}\n"
                   f"Total Train Time: {total_train_time:.2f}s\n{'='*40}")
        print(summary)
        logging.info(summary)
        
        
