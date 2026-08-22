import os
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import cv2
import argparse
from tqdm import tqdm

# Project-specific imports
from mkunet_network import MK_UNet, MK_UNet_ShallowDec
from utils.dataloader_polyp import get_loader


# ===================== DEFAULT CLASS MAP =====================
# Update this with your actual class names
CLASS_MAP = {
    0: 'Background',
    1: 'A', 2: 'B', 3: 'C', 4: 'D',
    5: 'E', 6: 'F', 7: 'G', 8: 'H',
    9: 'I', 10: 'J', 11: 'K', 12: 'L',
    13: 'M', 14: 'N'
}


def compute_multiclass_metrics(pred_mask, gt_mask, num_classes):
    """Per-class Dice and IoU."""
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
    """Percentage of area occupied by each class."""
    total_pixels = pred_mask.numel()
    percentages = {}
    for c in range(num_classes):
        count = (pred_mask == c).sum().item()
        percentages[c] = round((count / total_pixels) * 100.0, 2)
    return percentages


def test(model, path, dataset, opt, save_base=None):
    """Evaluates the model, saves prediction masks, and computes area percentages."""
    data_path = os.path.join(path, dataset)
    image_root = f'{data_path}/images/'
    gt_root = f'{data_path}/masks/'
    model.eval()
    
    test_loader = get_loader(
        image_root=image_root, gt_root=gt_root, 
        batchsize=opt.test_batchsize, trainsize=opt.img_size,
        shuffle=False, split='test', color_image=opt.color_image,
        num_classes=opt.num_classes
    )
    
    detailed_results = []
    all_dice = {c: [] for c in range(opt.num_classes)}
    total_images = 0

    with torch.no_grad():
        for pack in tqdm(test_loader, desc=f"Inference on {dataset}"):
            images, gts, original_shapes, names = pack       
            images = images.cuda()
            gts = gts.cuda().long()

            ress = model(images)
            predictions = ress[0] if isinstance(ress, list) else ress
            cls_logits = ress[1] if isinstance(ress, list) and len(ress) > 1 else None
            
            for i in range(len(images)):
                h_orig, w_orig = int(original_shapes[0][i]), int(original_shapes[1][i])
                
                # 1. Resize prediction
                p = predictions[i].unsqueeze(0)
                pred_resized = F.interpolate(p, size=(h_orig, w_orig), mode='bilinear', align_corners=False)
                pred_class = torch.argmax(pred_resized.squeeze(0), dim=0)  # [H, W]
                
                # 2. Resize GT
                g = gts[i].unsqueeze(0).float()
                gt_resized = F.interpolate(g, size=(h_orig, w_orig), mode='nearest').squeeze().long()

                # 3. Per-class metrics
                metrics = compute_multiclass_metrics(pred_class, gt_resized, opt.num_classes)
                
                # 4. Area percentages
                area_pct = compute_area_percentages(pred_class, opt.num_classes)

                # 5. Build result row
                row = {'Name': names[i]}
                for c in range(opt.num_classes):
                    cls_name = CLASS_MAP.get(c, f'Class_{c}')
                    row[f'Dice_{cls_name}'] = round(metrics[c]['dice'], 4)
                    row[f'IoU_{cls_name}'] = round(metrics[c]['iou'], 4)
                    row[f'Area%_{cls_name}'] = area_pct[c]
                
                # Mean dice across non-background classes
                fg_dices = [metrics[c]['dice'] for c in range(1, opt.num_classes)]
                row['Mean_Dice_FG'] = round(np.mean(fg_dices), 4) if fg_dices else 0.0

                # 5b. Classification predictions (if available)
                if cls_logits is not None:
                    cls_probs = torch.sigmoid(cls_logits[i])  # [num_classes-1]
                    cls_pred_binary = (cls_probs > 0.5).int()
                    # GT: which foreground classes exist in this tile
                    for c in range(1, opt.num_classes):
                        cls_name = CLASS_MAP.get(c, f'Class_{c}')
                        row[f'Cls_Pred_{cls_name}'] = cls_pred_binary[c - 1].item()
                        row[f'Cls_Prob_{cls_name}'] = round(cls_probs[c - 1].item(), 4)
                        row[f'Cls_GT_{cls_name}'] = int((gt_resized == c).any().item())

                detailed_results.append(row)
                total_images += 1

                for c in range(opt.num_classes):
                    all_dice[c].append(metrics[c]['dice'])

                # 6. Save colored prediction mask
                if save_base:
                    # Save as class-ID mask
                    pred_img = pred_class.cpu().numpy().astype(np.uint8)
                    cv2.imwrite(os.path.join(save_base, names[i]), pred_img)
                    
                    # Save a colorized version for visualization
                    color_mask = colorize_mask(pred_img, opt.num_classes)
                    color_name = names[i].rsplit('.', 1)[0] + '_color.png'
                    cv2.imwrite(os.path.join(save_base, color_name), color_mask)

    # Overall mean dice (foreground only)
    mean_fg_dice = np.mean([np.mean(all_dice[c]) for c in range(1, opt.num_classes)])
    mean_all_dice = np.mean([np.mean(all_dice[c]) for c in range(opt.num_classes)])
    
    return mean_fg_dice, mean_all_dice, detailed_results


def colorize_mask(mask, num_classes):
    """Creates a colorized RGB visualization of a class-ID mask."""
    # Generate distinct colors for each class
    np.random.seed(42)
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Background = black
    for c in range(1, num_classes):
        colors[c] = np.random.randint(50, 255, size=3)
    
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        color_img[mask == c] = colors[c]
    
    return color_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, help='ID of the run to test')
    parser.add_argument('--network', type=str, default='MK_UNet_ShallowDec_L',
                        choices=['MK_UNet_T', 'MK_UNet_S', 'MK_UNet', 'MK_UNet_ShallowDec', 'MK_UNet_ShallowDec_L'])
    parser.add_argument('--num_classes', type=int, default=32)
    parser.add_argument('--dataset_name', type=str, default='Plantation')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--test_batchsize', type=int, default=1)
    parser.add_argument('--color_image', default=True)
    parser.add_argument('--test_path', type=str, default='./data/polyp/target/')
    parser.add_argument('--class_map_file', type=str, default=None, help='Optional JSON file with class name mapping')
    opt = parser.parse_args()

    # Load custom class map if provided
    if opt.class_map_file and os.path.exists(opt.class_map_file):
        with open(opt.class_map_file, 'r') as f:
            loaded = json.load(f)
            CLASS_MAP = {int(k): v for k, v in loaded.items()}

    # --- Paths ---
    save_base = f'./predictions_plantation/{opt.run_id}/{opt.dataset_name}/{opt.split}'
    os.makedirs(save_base, exist_ok=True)
    os.makedirs('results_plantation', exist_ok=True)
    model_path = os.path.join(f'./model_pth/{opt.run_id}/', f'{opt.run_id}-best.pth')

    opt.test_path = f'{opt.test_path}/{opt.dataset_name}/'

    # --- Model Loading ---
    NET_CONFIGS = {
        'MK_UNet_T':            ([4, 8, 16, 24, 32],       MK_UNet),
        'MK_UNet_S':            ([8, 16, 32, 48, 80],       MK_UNet),
        'MK_UNet':              ([16, 32, 64, 96, 160],     MK_UNet),
        'MK_UNet_ShallowDec':   ([16, 32, 64, 96, 160],     MK_UNet_ShallowDec),
        'MK_UNet_ShallowDec_L': ([64, 128, 256, 384, 512],  MK_UNet_ShallowDec),
    }
    
    channels, ModelClass = NET_CONFIGS.get(opt.network, NET_CONFIGS['MK_UNet_ShallowDec_L'])
    model = ModelClass(num_classes=opt.num_classes, in_channels=3, channels=channels, enable_cls=True).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    # --- Run Inference ---
    mean_fg_dice, mean_all_dice, results = test(model, opt.test_path, opt.split, opt, save_base=save_base)

    # --- Save Detailed Results to Excel ---
    df = pd.DataFrame(results)
    
    # Add average row
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['Name'] = 'AVERAGE'
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    
    excel_name = f'results_plantation/Results_{opt.run_id}_{opt.dataset_name}_{opt.split}.xlsx'
    df.to_excel(excel_name, index=False)

    # --- Print Summary ---
    print(f"\n{'='*60}")
    print(f"RESULTS for {opt.run_id}")
    print(f"{'='*60}")
    print(f"Mean Foreground Dice: {mean_fg_dice:.4f}")
    print(f"Mean All-Class Dice:  {mean_all_dice:.4f}")
    print(f"\n--- Area Percentages (averaged across all images) ---")
    
    for c in range(opt.num_classes):
        cls_name = CLASS_MAP.get(c, f'Class_{c}')
        area_col = f'Area%_{cls_name}'
        if area_col in df.columns:
            avg_area = df[area_col].iloc[:-1].mean()  # Exclude AVERAGE row
            print(f"  {cls_name:15s}: {avg_area:6.2f}%")
    
    print(f"\nExcel report saved to: {excel_name}")
    print(f"Predictions saved to: {save_base}/")
