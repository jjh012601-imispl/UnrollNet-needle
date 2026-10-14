import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.UnrollNet_seg import UnrollNet
from data.dataset import NeedleDataset
from models.utils import custom_collate_fn, compute_dice

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Config:
    base_dir       = '/data1/jihyun/needle'
    matches_file   = '/data1/jihyun/needle/top1_matches.txt'
    p_matrices_dir = '/data1/jihyun/needle/P_matrices'
    save_dir       = './checkpoints_teacher'
    unroll_iter    = 5
    iter_cg        = 10
    batch_size     = 4
    num_epochs     = 100
    lr             = 1e-4
    patience       = 30
    criterion      = nn.MSELoss()

opt = Config()
criterion = opt.criterion


def train_epoch(model, labeled_loader, optimizer, device):
    """Supervised training"""
    model.train()

    total_loss     = 0
    total_sup_loss = 0

    for batch_idx, labeled_batch in enumerate(labeled_loader):
        optimizer.zero_grad()
        sup_loss = 0.0

        labeled_img = labeled_batch['image'].to(device)
        labeled_gt  = labeled_batch['label'].to(device)
        labeled_P   = labeled_batch['P']

        P_list = ([torch.tensor(labeled_P[i], dtype=torch.float32).to(device)
                   for i in range(len(labeled_P))]
                  if labeled_P is not None else None)

        _, labeled_output_list = model(labeled_img, labeled_img, P_list)
        R_output_list  = labeled_output_list[0]
        DC_output_list = labeled_output_list[1]

        for i in range(opt.unroll_iter):
            sup_loss += (criterion(R_output_list[i].squeeze(1),  labeled_gt)
                       + criterion(DC_output_list[i].squeeze(1), labeled_gt))

        loss = sup_loss

        total_loss     += loss.item()
        total_sup_loss += sup_loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 60 == 0:
            print(f'  Batch [{batch_idx}/{len(labeled_loader)}] '
                  f'Total: {loss.item():.4f}  Sup: {sup_loss.item():.4f}')

    n = len(labeled_loader)
    return total_loss / n, total_sup_loss / n


def validate(model, val_loader, device):
    model.eval()
    total_loss         = 0
    total_dice_r_list  = []
    total_dice_dc_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            val_img = batch['image'].to(device)
            val_gt  = batch['label'].to(device)
            val_P   = batch['P']

            P_list = ([torch.tensor(val_P[i]).to(device) for i in range(len(val_P))]
                      if val_P is not None else None)

            dc_output, [net_output_list, dc_output_list] = model(val_img, val_img, P_list)
            num_iters = len(net_output_list)

            if batch_idx == 0:
                total_dice_r_list  = [0.0] * num_iters
                total_dice_dc_list = [0.0] * num_iters

            loss = criterion((dc_output.squeeze(1) > 0.5).float(), val_gt)
            total_loss += loss.item()

            for idx in range(num_iters):
                pred_r  = (net_output_list[idx].squeeze(1) > 0.5).float()
                pred_dc = (dc_output_list[idx].squeeze(1)  > 0.5).float()
                total_dice_r_list[idx]  += compute_dice(pred_r,  val_gt)
                total_dice_dc_list[idx] += compute_dice(pred_dc, val_gt)

    n = len(val_loader)
    avg_dice_r_list  = [d / n for d in total_dice_r_list]
    avg_dice_dc_list = [d / n for d in total_dice_dc_list]

    final_dice_dc = avg_dice_dc_list[-1]
    final_dice_r  = avg_dice_r_list[-1]

    print(f'Validation - Loss: {total_loss/n:.4f}  '
          f'Dice(R): {final_dice_r:.4f}  Dice(DC): {final_dice_dc:.4f}')

    return total_loss / n, final_dice_r, final_dice_dc, avg_dice_r_list, avg_dice_dc_list


def main():
    os.makedirs(opt.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # ── Datasets ──────────────────────────────────────────
    print('Loading datasets...')

    labeled_dataset = NeedleDataset(
        base_dir=opt.base_dir,
        split='train',
        num=None,
        matches_file=opt.matches_file,
        p_matrices_dir=opt.p_matrices_dir,
        use_augmented=True   # train/original_aug 사용
    )

    val_dataset = NeedleDataset(
        base_dir=opt.base_dir,
        split='val',
        matches_file=opt.matches_file,
        p_matrices_dir=opt.p_matrices_dir,
        use_augmented=True   # val/original_aug 사용
    )

    labeled_loader = DataLoader(labeled_dataset, batch_size=opt.batch_size,
                                shuffle=True,  collate_fn=custom_collate_fn)
    val_loader     = DataLoader(val_dataset,    batch_size=opt.batch_size,
                                shuffle=False, collate_fn=custom_collate_fn)

    print(f'  Train: {len(labeled_dataset)}  Val: {len(val_dataset)}')

    # ── Model ─────────────────────────────────────────────
    model     = UnrollNet(in_channels=1, unroll_iter=opt.unroll_iter,
                          iter_cg=opt.iter_cg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    best_dice_dc     = 0.0
    best_dice_r      = 0.0
    best_loss        = float('inf')
    patience_counter = 0
    best_epoch       = 0

    training_log = []

    print('\nStarting training...')
    for epoch in range(opt.num_epochs):
        print(f'\n===== Epoch {epoch+1}/{opt.num_epochs} =====')

        train_loss, sup_loss = train_epoch(model, labeled_loader, optimizer, device)
        print(f'Train  total={train_loss:.4f}  sup={sup_loss:.4f}')

        val_loss, final_dice_r, final_dice_dc, dice_r_list, dice_dc_list = validate(
            model, val_loader, device)

        improved = False

        # DC 기준 best model 저장
        if final_dice_dc > best_dice_dc:
            best_dice_dc     = final_dice_dc
            best_dice_r      = final_dice_r
            improved         = True
            patience_counter = 0
            best_epoch       = epoch

            dice_str   = f'{int(final_dice_dc * 10000)}'
            model_name = f'best_model_DC_{dice_str}.pth'
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),   # ← model_state_dict 통일
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice_dc':         best_dice_dc,
                'best_dice_r':          best_dice_r,
                'dice_r_list':          dice_r_list,
                'dice_dc_list':         dice_dc_list,
            }, os.path.join(opt.save_dir, model_name))
            print(f'✓ Saved DC model: {model_name}')

        # R 기준 best model 저장
        if final_dice_r > best_dice_r:
            dice_str   = f'{int(final_dice_r * 10000)}'
            model_name = f'best_model_R_{dice_str}.pth'
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice_r':          final_dice_r,
                'best_dice_dc':         final_dice_dc,
                'dice_r_list':          dice_r_list,
                'dice_dc_list':         dice_dc_list,
            }, os.path.join(opt.save_dir, model_name))
            print(f'✓ Saved R model: {model_name}')

        # Loss 기준 best model 저장
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch':            epoch,
                'model_state_dict': model.state_dict(),
                'best_loss':        best_loss,
                'best_dice_r':      final_dice_r,
                'best_dice_dc':     final_dice_dc,
            }, os.path.join(opt.save_dir, 'best_model_Loss.pth'))
            print(f'✓ Saved Loss model: {best_loss:.4f}')

        if not improved:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{opt.patience}')

        training_log.append({
            'epoch':      epoch + 1,
            'train_loss': float(train_loss),
            'sup_loss':   float(sup_loss),
            'val_loss':   float(val_loss),
            'dice_r':     float(final_dice_r),
            'dice_dc':    float(final_dice_dc),
        })
        with open(os.path.join(opt.save_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=4)

        if patience_counter >= opt.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print(f'\nTraining completed.')
    print(f'Best DC Dice: {best_dice_dc:.4f} @ epoch {best_epoch+1}')
    print(f'Best R  Dice: {best_dice_r:.4f}')
    print(f'Best Loss   : {best_loss:.4f}')


if __name__ == '__main__':
    main()