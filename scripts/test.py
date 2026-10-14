import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import cv2
from models.UnrollNet_seg import UnrollNet
from data.dataset import NeedleDataset
from models.utils import custom_collate_fn, compute_dice

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# ── Config ────────────────────────────────────────────────────────────────────
class DefaultConfig(object):
    base_dir       = '/data1/jihyun/needle'
    matches_file   = '/data1/jihyun/needle/top1_matches_with_all4.txt'
    p_matrices_dir = '/data1/jihyun/needle/P_matrices'
    unroll_iter    = 5
    iter_cg        = 10
    batch_size     = 4

opt = DefaultConfig()

# ── 평가할 모델 / 저장 위치: 명령행 인자로 지정 ─────────────────────────────────
#   student : python test5.py --model <student.pth> --out ./eval_student --mode "SSL student"
#   teacher : python test5.py --model <teacher.pth> --out ./eval_teacher --mode "supervised teacher"
#   시간 측정: python test5.py --model <pth> --out ./eval_bench --no_vis --bench
import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument('--model', default='/data1/jihyun/SSDU_selftraining2/checkpoints_unrolliter_supervised/unroll_iter_5/best_model_DC_5977.pth')
_ap.add_argument('--out',   default='./eval_teacher')
_ap.add_argument('--mode',  default='supervised teacher')
_ap.add_argument('--no_vis', action='store_true', help='예측 시각화 저장 생략 (N 민감도처럼 여러 번 돌릴 때)')
_ap.add_argument('--bench',  action='store_true', help='단계별 추론 시간 측정 (batch=1, ms/sample)')
_args = _ap.parse_args()
MODEL_PATH, OUT_DIR, MODE = _args.model, _args.out, _args.mode
SPLIT = 'test'          # 평가 split (test = 15장)


# ============================================================
#  Metrics
# ============================================================

def compute_iou(pred, target, smooth=1e-6):
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    return ((intersection + smooth) / (union + smooth)).mean().item()


def compute_iou_per_sample(pred, target, smooth=1e-6):
    """샘플별 IoU 리스트 반환 (std 계산용)"""
    intersection = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.cpu().tolist()


def compute_hd95_batch(pred, target):
    pred_np   = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    hd95_list    = []
    failed_count = 0
    for i in range(pred_np.shape[0]):
        pred_pts   = np.argwhere(pred_np[i]   > 0)
        target_pts = np.argwhere(target_np[i] > 0)
        if len(pred_pts) == 0 or len(target_pts) == 0:
            failed_count += 1
            continue
        try:
            dist_fwd = np.array([np.min(np.linalg.norm(target_pts - p, axis=1))
                                  for p in pred_pts])
            dist_bwd = np.array([np.min(np.linalg.norm(pred_pts - t, axis=1))
                                  for t in target_pts])
            hd95_list.append(max(np.percentile(dist_fwd, 95),
                                  np.percentile(dist_bwd, 95)))
        except Exception:
            failed_count += 1
    return hd95_list, failed_count


# ============================================================
#  Needle fitting
# ============================================================

def fit_needle_line_hough(mask_tensor, pixel_spacing=0.1):
    mask  = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=50, maxLineGap=20)
    if lines is None:
        return None, None, False
    longest = max(lines, key=lambda l: np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]))
    x1, y1, x2, y2 = longest[0]
    angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    mask_np = mask_tensor.cpu().numpy()
    pts = np.argwhere(mask_np > 0)
    if len(pts) == 0:
        return None, None, False
    tip_idx   = np.argmax(pts[:, 0])
    tip_point = (int(pts[tip_idx, 1]), int(pts[tip_idx, 0]))
    return angle_deg, tip_point, True


def fit_needle_line_ransac(mask_tensor, pixel_spacing=0.1):
    from sklearn.linear_model import RANSACRegressor
    mask = mask_tensor.cpu().numpy()
    pts  = np.argwhere(mask > 0)
    if len(pts) < 10:
        return None, None, False
    y_c = pts[:, 0]; x_c = pts[:, 1]
    ransac = RANSACRegressor(residual_threshold=2.0, max_trials=1000, random_state=42)
    try:
        ransac.fit(x_c.reshape(-1, 1), y_c)
    except Exception:
        return None, None, False
    angle_deg = np.degrees(np.arctan(ransac.estimator_.coef_[0]))
    tip_idx   = np.argmax(y_c)
    tip_point = (int(x_c[tip_idx]), int(y_c[tip_idx]))
    return angle_deg, tip_point, True


def compute_needle_metrics(pred_mask, gt_mask, pixel_spacing=0.1):
    gt_ang,   gt_tip,   gt_ok   = fit_needle_line_hough(gt_mask,   pixel_spacing)
    if not gt_ok:
        gt_ang, gt_tip, gt_ok   = fit_needle_line_ransac(gt_mask,  pixel_spacing)
    pred_ang, pred_tip, pred_ok = fit_needle_line_hough(pred_mask, pixel_spacing)
    if not pred_ok:
        pred_ang, pred_tip, pred_ok = fit_needle_line_ransac(pred_mask, pixel_spacing)
    if not gt_ok or not pred_ok:
        return {'success': False, 'trajectory_error_deg': None,
                'tip_error_mm': None, 'tip_error_pixels': None,
                'gt_angle': None, 'pred_angle': None,
                'gt_tip': None,   'pred_tip': None}
    traj_raw   = abs(gt_ang - pred_ang)
    d          = traj_raw % 180
    traj_err   = min(d, 180 - d)          # 직선 방향 모호성(180도) 제거
    tip_err_px = np.linalg.norm(np.array(gt_tip) - np.array(pred_tip))
    return {'success': True,
            'trajectory_error_deg': float(traj_err),
            'trajectory_error_raw_deg': float(traj_raw),
            'tip_error_mm':         float(tip_err_px * pixel_spacing),
            'tip_error_pixels':     float(tip_err_px),
            'gt_angle': gt_ang, 'pred_angle': pred_ang,
            'gt_tip': gt_tip,   'pred_tip':   pred_tip}


# ============================================================
#  요약 (실패 샘플 처리 + 그룹별)
# ============================================================

def _ms(a):
    return {'mean': float(np.mean(a)), 'std': float(np.std(a))} if len(a) else None


def summarize_samples(ps, pixel_spacing=0.1, img_size=256):
    """실패 샘플(빈 예측 / 직선 피팅 실패)을 평균에서 빼지 않고 패널티로 채운 값도 같이 계산.
       *_valid      : 성공한 샘플만 평균 (기존 방식)
       *_penalized  : 실패 샘플 = 패널티(HD95 = 이미지 대각선, Traj = 90 deg, Tip = 대각선*spacing)
    """
    if not ps:
        return None
    PEN_HD   = img_size * np.sqrt(2)
    PEN_TRAJ = 90.0
    PEN_TIP  = PEN_HD * pixel_spacing
    hd = [s['hd95_dc'] for s in ps]
    tr = [s['trajectory_error_deg'] for s in ps]
    tp = [s['tip_error_mm'] for s in ps]
    return {
        'n':               len(ps),
        'dice_dc':         _ms([s['dice_dc'] for s in ps]),
        'iou_dc':          _ms([s['iou_dc'] for s in ps]),
        'hd95_valid':      _ms([x for x in hd if x is not None]),
        'hd95_penalized':  _ms([x if x is not None else PEN_HD for x in hd]),
        'num_empty_pred':  sum(x is None for x in hd),
        'traj_valid':      _ms([x for x in tr if x is not None]),
        'traj_penalized':  _ms([x if x is not None else PEN_TRAJ for x in tr]),
        'tip_valid':       _ms([x for x in tp if x is not None]),
        'tip_penalized':   _ms([x if x is not None else PEN_TIP for x in tp]),
        'num_fit_fail':    sum(x is None for x in tr),
    }


def print_summary(title, sm):
    if sm is None:
        print(f'{title}: (no samples)'); return
    f = lambda d, fmt='{:.4f}': (fmt.format(d['mean']) + ' +/- ' + fmt.format(d['std'])) if d else '-'
    n = sm['n']
    print(f'--- {title}  (n={n}) ---')
    print(f'  Dice     : {f(sm["dice_dc"])}')
    print(f'  IoU      : {f(sm["iou_dc"])}')
    print(f'  HD95     : valid {f(sm["hd95_valid"], "{:.2f}")}  ({n - sm["num_empty_pred"]}/{n})  |  '
          f'penalized {f(sm["hd95_penalized"], "{:.2f}")}  (empty pred {sm["num_empty_pred"]})')
    print(f'  Traj(deg): valid {f(sm["traj_valid"], "{:.2f}")}  ({n - sm["num_fit_fail"]}/{n})  |  '
          f'penalized {f(sm["traj_penalized"], "{:.2f}")}')
    print(f'  Tip (mm) : valid {f(sm["tip_valid"], "{:.2f}")}  |  penalized {f(sm["tip_penalized"], "{:.2f}")}')


# ============================================================
#  테스트
# ============================================================

def test_single_model(model, test_loader, device, threshold=0.5, pixel_spacing=0.1):
    model.eval()
    all_dice_dc = []; all_dice_r = []
    all_iou_dc  = []   # ← per-sample IoU (std 계산용)
    all_hd95    = []
    all_traj    = []; all_tip   = []
    hd95_failed = 0;  total_s   = 0
    per_sample  = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img = batch['image'].to(device)
            gt  = batch['label'].to(device)
            P   = batch['P']
            P_list = ([torch.tensor(P[i], dtype=torch.float32).to(device)
                       for i in range(len(P))] if P is not None else None)

            _, [net_out, dc_out] = model(img, img, P_list)

            pred_r  = (net_out[-1].squeeze(1) > threshold).float()
            pred_dc = (dc_out[-1].squeeze(1)  > threshold).float()

            all_dice_r.append(compute_dice(pred_r,  gt))
            all_dice_dc.append(compute_dice(pred_dc, gt))

            # ── IoU: 샘플별로 extend ────────────────────────
            all_iou_dc.extend(compute_iou_per_sample(pred_dc, gt))

            h_list, h_fail = compute_hd95_batch(pred_dc, gt)
            all_hd95.extend(h_list)
            hd95_failed += h_fail
            total_s     += pred_dc.shape[0]

            for i in range(pred_dc.shape[0]):
                dataset_idx = batch_idx * test_loader.batch_size + i
                img_name = (test_loader.dataset.image_list[dataset_idx]
                            if dataset_idx < len(test_loader.dataset.image_list)
                            else f'sample_{dataset_idx}')

                s_dice_dc = compute_dice(pred_dc[i:i+1], gt[i:i+1])
                s_dice_r  = compute_dice(pred_r[i:i+1],  gt[i:i+1])
                s_iou_dc  = compute_iou(pred_dc[i:i+1],  gt[i:i+1])

                h_s, _ = compute_hd95_batch(pred_dc[i:i+1], gt[i:i+1])
                s_hd95 = float(h_s[0]) if h_s else None

                nm = compute_needle_metrics(pred_dc[i], gt[i], pixel_spacing)
                if nm['success']:
                    all_traj.append(nm['trajectory_error_deg'])
                    all_tip.append(nm['tip_error_mm'])

                per_sample.append({
                    'image':                img_name,
                    'dice_dc':              float(s_dice_dc),
                    'dice_r':               float(s_dice_r),
                    'iou_dc':               float(s_iou_dc),
                    'hd95_dc':              s_hd95,
                    'trajectory_error_deg': float(nm['trajectory_error_deg']) if nm['success'] else None,
                    'tip_error_mm':         float(nm['tip_error_mm'])         if nm['success'] else None,
                    'needle_fit_success':   nm['success'],
                })

    def _s(arr):
        return {'mean': float(np.mean(arr)), 'std': float(np.std(arr))}

    return {
        'dice_r':           _s(all_dice_r),
        'dice_dc':          _s(all_dice_dc),
        'iou_dc':           _s(all_iou_dc),   # ← per-sample 기반 std
        'hd95_dc':          {'mean': float(np.mean(all_hd95)), 'std': float(np.std(all_hd95)),
                             'num_valid': len(all_hd95), 'num_failed': hd95_failed,
                             'total': total_s} if all_hd95 else None,
        'trajectory_error': _s(all_traj) if all_traj else None,
        'tip_error_mm':     _s(all_tip)  if all_tip  else None,
        'num_needle_fits':  len(all_traj),
        'summary_all':      summarize_samples(per_sample, pixel_spacing),
        'per_sample':       per_sample,
    }


# ============================================================
#  추론 시간 측정 (batch=1, ms/sample)
# ============================================================

def benchmark_timing(model, dataset, device, N, n_warm=10, threshold=0.5, pixel_spacing=0.1):
    """단계별 샘플당 시간:
         (a) P 행렬 CPU->GPU 전송
         (b) 네트워크 forward (N 회 unrolled R-A, CG 포함)
         (c) 예측 마스크 궤적/팁 추정 (Hough, 실패 시 RANSAC)  ※ GT 피팅은 제외
       미리 계산된 P 파일을 읽어 쓰므로 SVD(P 계산) 시간은 포함되지 않음.
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    model.eval()
    sync = torch.cuda.synchronize if device.type == 'cuda' else (lambda: None)
    t_xfer, t_fwd, t_post = [], [], []
    with torch.no_grad():
        for k, batch in enumerate(loader):
            img = batch['image'].to(device)
            sync(); t0 = time.perf_counter()
            P = batch['P']
            P_list = ([torch.tensor(P[i], dtype=torch.float32).to(device) for i in range(len(P))]
                      if P is not None else None)
            sync(); t1 = time.perf_counter()
            _, [net_out, dc_out] = model(img, img, P_list)
            sync(); t2 = time.perf_counter()
            pred = (dc_out[-1].squeeze(1) > threshold).float()[0]
            _, _, ok = fit_needle_line_hough(pred, pixel_spacing)
            if not ok:
                fit_needle_line_ransac(pred, pixel_spacing)
            t3 = time.perf_counter()
            if k >= n_warm:                                   # 워밍업 제외
                t_xfer.append((t1 - t0) * 1e3)
                t_fwd.append((t2 - t1) * 1e3)
                t_post.append((t3 - t2) * 1e3)
    f = lambda a: f'{np.mean(a):.2f} +/- {np.std(a):.2f}'
    print(f'\n[Timing] batch=1, n={len(t_fwd)} (warm-up {n_warm} 제외), ms/sample, device={device}')
    print(f'  P transfer (CPU->GPU)        : {f(t_xfer)}')
    print(f'  Network forward (N={N})        : {f(t_fwd)}')
    print(f'  Traj/tip estimation (pred)   : {f(t_post)}')
    print('  * SVD(P 계산), 데이터 로딩은 미포함')
    return {'n': len(t_fwd), 'n_warmup': n_warm, 'unroll_iter': N,
            'p_transfer_ms': {'mean': float(np.mean(t_xfer)), 'std': float(np.std(t_xfer))},
            'forward_ms':    {'mean': float(np.mean(t_fwd)),  'std': float(np.std(t_fwd))},
            'postproc_ms':   {'mean': float(np.mean(t_post)), 'std': float(np.std(t_post))}}


# ============================================================
#  시각화
# ============================================================

def visualize_predictions(model, test_loader, device, save_dir,
                          threshold=0.5, pixel_spacing=0.1):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    overlay_dir = os.path.join(save_dir, 'overlays')
    os.makedirs(overlay_dir, exist_ok=True)

    sample_idx = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            img = batch['image'].to(device)
            gt  = batch['label'].to(device)
            P   = batch['P']
            P_list = ([torch.tensor(P[i], dtype=torch.float32).to(device)
                       for i in range(len(P))] if P is not None else None)

            _, [net_out, dc_out] = model(img, img, P_list)

            for i in range(img.shape[0]):
                dataset_idx = batch_idx * test_loader.batch_size + i
                if dataset_idx >= len(test_loader.dataset.image_list):
                    break

                img_name  = test_loader.dataset.image_list[dataset_idx]
                base_name = os.path.splitext(img_name)[0]

                orig = img[i, 0].cpu().numpy()
                gt_m = gt[i].cpu().numpy()
                r_p  = (net_out[-1][i, 0].cpu().numpy() > threshold).astype(float)
                dc_p = (dc_out[-1][i, 0].cpu().numpy()  > threshold).astype(float)
                H, W = orig.shape

                dice_r  = compute_dice(torch.tensor(r_p, device=device).unsqueeze(0),
                                       torch.tensor(gt_m, device=device).unsqueeze(0))
                dice_dc = compute_dice(torch.tensor(dc_p, device=device).unsqueeze(0),
                                       torch.tensor(gt_m, device=device).unsqueeze(0))

                nm = compute_needle_metrics((dc_out[-1][i, 0] > threshold).float(),
                                            gt[i], pixel_spacing)

                fig = plt.figure(figsize=(32, 7))
                gs  = fig.add_gridspec(1, 5, left=0.03, right=0.97,
                                       top=0.92, bottom=0.08, wspace=0.12)
                axes = [fig.add_subplot(gs[0, c]) for c in range(5)]
                length = 200

                axes[0].imshow(orig, cmap='gray')
                axes[0].set_title('Original', fontsize=17, fontweight='bold', pad=12)
                axes[0].axis('off')

                axes[1].imshow(orig, cmap='gray', alpha=0.7)
                axes[1].imshow(gt_m, cmap='Reds', alpha=0.3)
                axes[1].set_xlim(0, W); axes[1].set_ylim(H, 0)
                if nm['success'] and nm['gt_tip'] is not None:
                    x1, y1 = nm['gt_tip']
                    gt_ang = np.radians(nm['gt_angle'])
                    x2 = int(x1 - length * np.cos(gt_ang))
                    y2 = int(y1 - length * np.sin(gt_ang))
                    axes[1].plot([x1, x2], [y1, y2], 'g-', linewidth=3, label='GT')
                    axes[1].plot(x1, y1, 'go', markersize=14)
                    axes[1].legend(loc='upper right', fontsize=10)
                axes[1].set_title('Ground Truth', fontsize=17, fontweight='bold', pad=12)
                axes[1].axis('off')

                axes[2].imshow(orig, cmap='gray', alpha=0.7)
                axes[2].imshow(r_p, cmap='Blues', alpha=0.3)
                axes[2].set_xlim(0, W); axes[2].set_ylim(H, 0)
                axes[2].set_title(f'R Output\nDice:{dice_r:.4f}', fontsize=17,
                                  fontweight='bold', pad=12)
                axes[2].axis('off')

                axes[3].imshow(orig, cmap='gray', alpha=0.7)
                axes[3].imshow(dc_p, cmap='Purples', alpha=0.3)
                axes[3].set_xlim(0, W); axes[3].set_ylim(H, 0)
                if nm['success'] and nm['pred_tip'] is not None:
                    x1, y1 = nm['pred_tip']
                    pa = np.radians(nm['pred_angle'])
                    x2 = int(x1 - length * np.cos(pa))
                    y2 = int(y1 - length * np.sin(pa))
                    axes[3].plot([x1, x2], [y1, y2], 'y-', linewidth=3, label='Pred')
                    axes[3].plot(x1, y1, 'yo', markersize=14)
                    axes[3].legend(loc='upper right', fontsize=10)
                axes[3].set_title(f'DC Output\nDice:{dice_dc:.4f}', fontsize=17,
                                  fontweight='bold', pad=12)
                axes[3].axis('off')

                overlay = np.stack([orig] * 3, axis=-1)
                overlay[(dc_p == 1) & (gt_m == 1)] = [1, 1, 0]
                overlay[(dc_p == 1) & (gt_m == 0)] = [1, 0, 0]
                overlay[(dc_p == 0) & (gt_m == 1)] = [0, 1, 0]
                axes[4].imshow(overlay)
                axes[4].set_xlim(0, W); axes[4].set_ylim(H, 0)
                if nm['success']:
                    if nm['gt_tip']:
                        x1, y1 = nm['gt_tip']
                        ga = np.radians(nm['gt_angle'])
                        axes[4].plot([x1, int(x1 - length*np.cos(ga))],
                                     [y1, int(y1 - length*np.sin(ga))],
                                     'g-', linewidth=3, label='GT')
                        axes[4].plot(x1, y1, 'go', markersize=14)
                    if nm['pred_tip']:
                        x1, y1 = nm['pred_tip']
                        pa = np.radians(nm['pred_angle'])
                        axes[4].plot([x1, int(x1 - length*np.cos(pa))],
                                     [y1, int(y1 - length*np.sin(pa))],
                                     'c-', linewidth=3, label='Pred')
                        axes[4].plot(x1, y1, 'co', markersize=14)
                    axes[4].text(0.02, 0.98,
                                 f"Traj:{nm['trajectory_error_deg']:.2f}°\n"
                                 f"Tip:{nm['tip_error_mm']:.2f}mm",
                                 transform=axes[4].transAxes,
                                 color='white', fontsize=12, fontweight='bold',
                                 va='top',
                                 bbox=dict(boxstyle='round,pad=0.5',
                                           facecolor='black', alpha=0.85))
                    axes[4].legend(loc='upper right', fontsize=10)
                else:
                    axes[4].text(0.02, 0.98, 'Needle fit failed',
                                 transform=axes[4].transAxes,
                                 color='red', fontsize=12, fontweight='bold', va='top',
                                 bbox=dict(boxstyle='round,pad=0.5',
                                           facecolor='black', alpha=0.85))
                axes[4].set_title('Overlay (TP/FP/FN)', fontsize=17,
                                  fontweight='bold', pad=12)
                axes[4].axis('off')

                fig.text(0.5, 0.02, img_name, ha='center',
                         fontsize=15, fontweight='bold')
                plt.savefig(os.path.join(save_dir,
                                         f'{sample_idx+1:03d}_{base_name}.png'),
                            dpi=150, bbox_inches='tight')
                plt.close(fig)

                gt_ov = np.stack([orig] * 3, axis=-1)
                gt_ov[:, :, 1] = np.maximum(gt_ov[:, :, 1], gt_m)
                cv2.imwrite(os.path.join(overlay_dir,
                                         f'{sample_idx+1:03d}_{base_name}_gt.png'),
                            cv2.cvtColor((gt_ov * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

                pred_ov = np.stack([orig] * 3, axis=-1)
                tp = (dc_p == 1) & (gt_m == 1)
                fp = (dc_p == 1) & (gt_m == 0)
                pred_ov[tp] = np.maximum(pred_ov[tp], [0, 1, 0])
                pred_ov[fp] = np.maximum(pred_ov[fp], [1, 0, 0])
                cv2.imwrite(os.path.join(overlay_dir,
                                        f'{sample_idx+1:03d}_{base_name}_pred.png'),
                            cv2.cvtColor((pred_ov * 255).astype(np.uint8),
                                         cv2.COLOR_RGB2BGR))

                sample_idx += 1

    print(f'Saved {sample_idx} visualizations -> {save_dir}')


# ============================================================
#  Main
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    ckpt_dir      = OUT_DIR
    vis_dir       = os.path.join(OUT_DIR, 'test_visualizations')
    os.makedirs(ckpt_dir, exist_ok=True)
    pixel_spacing = 0.1

    # 직접 pth 경로 지정
    model_path = MODEL_PATH

    if not os.path.isfile(model_path):
        print(f'Error: checkpoint not found: {model_path}')
        return
    print(f'Loading: {model_path}')

    test_dataset = NeedleDataset(
        base_dir=opt.base_dir,
        split=SPLIT,
        matches_file=opt.matches_file,
        p_matrices_dir=opt.p_matrices_dir,
        use_augmented=False
    )
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size,
                             shuffle=False, collate_fn=custom_collate_fn)
    print(f'Test samples: {len(test_dataset)}')

    ckpt = torch.load(model_path, map_location=device)
    N_ck = ckpt.get('unroll_iter', opt.unroll_iter)           # 체크포인트의 N 사용
    model = UnrollNet(in_channels=1, unroll_iter=N_ck, iter_cg=opt.iter_cg).to(device)
    key = next((k for k in ['model_state_dict', 'student_state_dict', 'state_dict'] if k in ckpt), None)
    assert key is not None, f'state_dict 키를 찾을 수 없음: {list(ckpt.keys())}'
    print(f'checkpoint key: {key}  (unroll_iter={N_ck})')
    model.load_state_dict(ckpt[key])

    m = test_single_model(model, test_loader, device, pixel_spacing=pixel_spacing)

    SEP = '=' * 70
    print(f'\n{SEP}')
    print(f'TEST RESULTS  (unroll_iter={N_ck}, {MODE}, split={SPLIT})  |  model: {os.path.basename(model_path)}')
    print(SEP)
    print(f'R  Dice : {m["dice_r"]["mean"]:.4f} +/- {m["dice_r"]["std"]:.4f}')
    print_summary(f'ALL  ({SPLIT})', m['summary_all'])
    print('* valid = 성공 샘플만 평균 / penalized = 실패 샘플에 패널티(HD95 대각선, Traj 90deg, Tip 대각선*0.1mm)')
    print(SEP)

    timing = benchmark_timing(model, test_dataset, device, N_ck,
                              pixel_spacing=pixel_spacing) if _args.bench else None

    result_out = {
        'model_path':  model_path,
        'unroll_iter': N_ck,
        'iter_cg':     opt.iter_cg,
        'mode':        MODE,
        'split':       SPLIT,
        'timing':      timing,
        'metrics':     m
    }
    with open(os.path.join(ckpt_dir, 'test_results.json'), 'w') as f:
        json.dump(result_out, f, indent=4)
    print(f'Results saved -> {ckpt_dir}/test_results.json')

    if not _args.no_vis:
        visualize_predictions(model, test_loader, device,
                              save_dir=vis_dir,
                              pixel_spacing=pixel_spacing)
    print('\nDONE.')


if __name__ == '__main__':
    main()