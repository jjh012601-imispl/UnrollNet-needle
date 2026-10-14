"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import h5py
import torch
import matplotlib.pyplot as plt
import os


def custom_collate_fn(batch):
    """P matrix는 numpy로 유지"""
    images = torch.stack([torch.from_numpy(item['image']) for item in batch])
    labels = torch.stack([torch.from_numpy(item['label']) for item in batch])
    
    P_list = [item['P'] for item in batch]
    P_T_list = [item['P_T'] for item in batch]
    
    return {
        'image': images,
        'label': labels,
        'P': P_list,
        'P_T': P_T_list
    }

def visualize_results(teacher_model, val_loader, device, epoch, save_dir='./visualizations'):
    teacher_model.eval()
    os.makedirs(save_dir, exist_ok=True)
    epoch_dir = os.path.join(save_dir, f'epoch_{epoch+1}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        
        val_img = batch['image'].to(device)
        val_gt = batch['label'].to(device)
        val_P = batch['P']
        
        val_dataset = val_loader.dataset
        P_list = [torch.tensor(val_P[i], dtype=torch.float32).to(device) for i in range(len(val_P))] if val_P is not None else None
        
        # Forward
        dc_output, [net_output_list, dc_output_list] = teacher_model(val_img, val_img, P_list)
        
        # Unroll iteration 개수 확인
        num_iters = len(net_output_list)
        
        # 시각화할 샘플 수
        num_viz = min(5, val_img.shape[0])
        
        for i in range(num_viz):
            img_name = val_dataset.image_list[i]
            
            # 동적으로 subplot 생성
            fig, axes = plt.subplots(num_iters, 4, figsize=(16, 4*num_iters))
            
            # 1차원 배열 처리 (iter=1일 경우)
            if num_iters == 1:
                axes = axes.reshape(1, -1)
            
            # 각 iteration에 대해 시각화
            for iter_idx in range(num_iters):
                # Original Image
                img_data = val_img[i, 0].cpu().numpy()
                axes[iter_idx, 0].imshow(img_data, cmap='gray', vmin=0, vmax=1)
                axes[iter_idx, 0].set_title(f'Iter {iter_idx+1}: Original\n{img_name}')
                axes[iter_idx, 0].axis('off')
                
                # Ground Truth
                gt_data = val_gt[i].cpu().numpy()
                axes[iter_idx, 1].imshow(gt_data, cmap='gray', vmin=0, vmax=1)
                axes[iter_idx, 1].set_title(f'Iter {iter_idx+1}: GT')
                axes[iter_idx, 1].axis('off')
                
                # R Output (before DC)
                r_data = net_output_list[iter_idx][i, 0].cpu().numpy()
                r_pred = (r_data > 0.5).astype(float)
                axes[iter_idx, 2].imshow(r_pred, cmap='gray', vmin=0, vmax=1)
                axes[iter_idx, 2].set_title(f'Iter {iter_idx+1}: R Output')
                axes[iter_idx, 2].axis('off')
                
                # DC Output (after DC)
                dc_data = dc_output_list[iter_idx][i, 0].cpu().numpy()
                dc_pred = (dc_data > 0.5).astype(float)
                axes[iter_idx, 3].imshow(dc_pred, cmap='gray', vmin=0, vmax=1)
                axes[iter_idx, 3].set_title(f'Iter {iter_idx+1}: DC Output')
                axes[iter_idx, 3].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f'sample_{i+1}_{img_name}'), 
                       dpi=100, bbox_inches='tight')
            plt.close()
    
    print(f'Visualizations saved to {epoch_dir}')

def compute_dice(pred, target, smooth=1e-6):
    """
    Dice Score 계산
    Args:
        pred : (B, H, W) - 이진 예측 마스크
        target : (B, H, W) - 이진 실제 마스크
    """
    
    intersection = (pred * target).sum(dim=(1,2))
    union = pred.sum(dim=(1,2)) + target.sum(dim=(1,2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

