import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from models.UnrollNet_seg import UnrollNet
from data.dataset import NeedleDataset
from models.utils import custom_collate_fn, compute_dice
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# ── Config ────────────────────────────────────────────────────────────────────
class DefaultConfig(object):
    base_dir       = '/data1/jihyun/needle'
    matches_file   = '/data1/jihyun/needle/top1_matches.txt'
    p_matrices_dir = '/data1/jihyun/needle/P_matrices'

    unroll_iter = 5   # teacher/student 모두 5
    iter_cg     = 10

    batch_size  = 4
    num_workers = 8
    criterion   = nn.MSELoss()
    num_epochs  = 100
    lr          = 1e-4
    patience    = 30

opt = DefaultConfig()
criterion = opt.criterion


# ── Train / Val ───────────────────────────────────────────────────────────────
def train_epoch(student_model, teacher_model, labeled_loader, unlabeled_loader,
                optimizer, device):
    student_model.train()
    teacher_model.eval()

    total_loss        = 0
    total_sup_loss    = 0
    total_pseudo_loss = 0

    unlabeled_iter = iter(unlabeled_loader)

    for batch_idx, labeled_batch in enumerate(labeled_loader):
        optimizer.zero_grad()
        sup_loss    = 0.0
        pseudo_loss = 0.0

        # Labeled
        labeled_img = labeled_batch['image'].to(device)
        labeled_gt  = labeled_batch['label'].to(device)
        labeled_P   = labeled_batch['P']

        P_list = ([torch.tensor(labeled_P[i], dtype=torch.float32).to(device)
                   for i in range(len(labeled_P))]
                  if labeled_P is not None else None)

        _, labeled_output_list = student_model(labeled_img, labeled_img, P_list)
        R_output_list  = labeled_output_list[0]
        DC_output_list = labeled_output_list[1]

        for i in range(opt.unroll_iter):
            sup_loss += (criterion(R_output_list[i].squeeze(1),  labeled_gt)
                       + criterion(DC_output_list[i].squeeze(1), labeled_gt))

        # Unlabeled
        try:
            unlabeled_batch = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter  = iter(unlabeled_loader)
            unlabeled_batch = next(unlabeled_iter)

        unlabeled_img = unlabeled_batch['image'].to(device)
        unlabeled_P   = unlabeled_batch['P']

        unlabeled_P_list = ([torch.tensor(unlabeled_P[i]).to(device)
                             for i in range(len(unlabeled_P))]
                            if unlabeled_P is not None else None)

        # Teacher → soft pseudo label
        with torch.no_grad():
            _, teacher_output = teacher_model(unlabeled_img, unlabeled_img, unlabeled_P_list)
            soft_pseudo_label = teacher_output[0][-1].detach()

        # Student pseudo loss
        _, student_output = student_model(unlabeled_img, unlabeled_img, unlabeled_P_list)
        student_R = student_output[0][-1]
        pseudo_loss = criterion(student_R.squeeze(1), soft_pseudo_label.squeeze(1))

        loss = sup_loss + pseudo_loss * 0.7

        total_loss        += loss.item()
        total_sup_loss    += sup_loss.item()
        total_pseudo_loss += pseudo_loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % 60 == 0:
            print(f'  Batch [{batch_idx}/{len(labeled_loader)}] '
                  f'Total: {loss.item():.4f}  Sup: {sup_loss.item():.4f}  '
                  f'Pseudo: {pseudo_loss.item():.4f}')

    n = len(labeled_loader)
    return total_loss / n, total_sup_loss / n, total_pseudo_loss / n


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

    return (total_loss / n,
            avg_dice_r_list[-1], avg_dice_dc_list[-1],
            avg_dice_r_list,     avg_dice_dc_list)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs('./checkpoints_student', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # !! teacher = unroll_iter=5 supervised best model !!
    teacher_path = './checkpoints_teacher/best_model_DC_*.pth'

    # glob으로 실제 경로 탐색
    import glob
    candidates = glob.glob(teacher_path)
    if not candidates:
        print(f'Error: teacher checkpoint not found at {teacher_path}')
        return
    teacher_path = max(candidates,
                       key=lambda x: int(os.path.basename(x)
                                         .replace('best_model_DC_', '')
                                         .replace('.pth', '')))
    print(f'Teacher: {teacher_path}')

    # Datasets
    labeled_dataset = NeedleDataset(
        base_dir=opt.base_dir, split='train', num=None,
        matches_file=opt.matches_file,
        p_matrices_dir=opt.p_matrices_dir,
        use_augmented=True
    )
    unlabeled_dataset = NeedleDataset(
        base_dir=opt.base_dir, split='unlabeled',
        matches_file=None,
        p_matrices_dir=opt.p_matrices_dir
    )
    val_dataset = NeedleDataset(
        base_dir=opt.base_dir, split='val',
        matches_file=opt.matches_file,
        p_matrices_dir=opt.p_matrices_dir,
        use_augmented=True
    )

    labeled_loader   = DataLoader(labeled_dataset,   batch_size=opt.batch_size,
                                  shuffle=True,  collate_fn=custom_collate_fn)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=opt.batch_size,
                                  shuffle=True,  collate_fn=custom_collate_fn)
    val_loader       = DataLoader(val_dataset,       batch_size=opt.batch_size,
                                  shuffle=False, collate_fn=custom_collate_fn)

    print(f'Train(aug): {len(labeled_dataset)}  '
          f'Unlabeled: {len(unlabeled_dataset)}  '
          f'Val: {len(val_dataset)}')

    # Teacher (fixed, unroll_iter=5)
    teacher_model = UnrollNet(in_channels=1, unroll_iter=opt.unroll_iter,
                              iter_cg=opt.iter_cg).to(device)
    ckpt = torch.load(teacher_path)
    teacher_model.load_state_dict(ckpt['model_state_dict'])
    teacher_model.eval()
    teacher_model.requires_grad_(False)
    print(f'Teacher loaded (Val Dice: {ckpt.get("best_dice_dc", 0.0):.4f})')

    # Student (unroll_iter=5)
    student_model = UnrollNet(in_channels=1, unroll_iter=opt.unroll_iter,
                              iter_cg=opt.iter_cg).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=10, verbose=True
)

    best_dice_dc     = 0.0
    best_dice_r      = 0.0
    best_loss        = float('inf')
    patience_counter = 0
    best_epoch       = 0
    training_log     = []

    print('\nStarting self-training (unroll_iter=5)...')
    for epoch in range(opt.num_epochs):
        print(f'\n===== Epoch {epoch+1}/{opt.num_epochs} =====')

        train_loss, sup_loss, pseudo_loss = train_epoch(
            student_model, teacher_model,
            labeled_loader, unlabeled_loader,
            optimizer, device)
        print(f'Train  total={train_loss:.4f}  sup={sup_loss:.4f}  pseudo={pseudo_loss:.4f}')

        val_loss, final_dice_r, final_dice_dc, dice_r_list, dice_dc_list = validate(
            student_model, val_loader, device)
        print(f'Val    loss={val_loss:.4f}  Dice(R)={final_dice_r:.4f}  Dice(DC)={final_dice_dc:.4f}')
        scheduler.step(final_dice_dc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'LR: {current_lr:.2e}')

        improved = False

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
                'unroll_iter':          opt.unroll_iter,
                'iter_cg':              opt.iter_cg,
                'model_state_dict':     student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice_dc':         best_dice_dc,
                'best_dice_r':          best_dice_r,
                'dice_r_list':          dice_r_list,
                'dice_dc_list':         dice_dc_list,
            }, os.path.join('./checkpoints_student', model_name))
            print(f'✓ Saved: {model_name}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch':            epoch,
                'unroll_iter':      opt.unroll_iter,
                'model_state_dict': student_model.state_dict(),
                'best_loss':        best_loss,
                'best_dice_r':      final_dice_r,
                'best_dice_dc':     final_dice_dc,
            }, './checkpoints_student/best_model_Loss.pth')
            print(f'✓ Saved Loss model: {best_loss:.4f}')

        if not improved:
            patience_counter += 1
            print(f'No improvement. Patience: {patience_counter}/{opt.patience}')

        training_log.append({
            'epoch':       epoch + 1,
            'train_loss':  float(train_loss),
            'sup_loss':    float(sup_loss),
            'pseudo_loss': float(pseudo_loss),
            'val_loss':    float(val_loss),
            'dice_r':      float(final_dice_r),
            'dice_dc':     float(final_dice_dc),
        })
        with open('./checkpoints_student/training_log.json', 'w') as f:
            json.dump(training_log, f, indent=4)

        if patience_counter >= opt.patience:
            print(f'\nEarly stopping at epoch {epoch+1}')
            break

    print(f'\nDone. Best Dice(DC)={best_dice_dc:.4f} @ epoch {best_epoch+1}')
    print(f'      Best Dice(R) ={best_dice_r:.4f}')


if __name__ == '__main__':
    main()