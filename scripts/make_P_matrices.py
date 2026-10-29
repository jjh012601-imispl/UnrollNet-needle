"""Generate background-subspace matrices P (rank-2 SVD of the first T' frames per sequence).

Usage:
  python scripts/make_P_matrices.py --spatials /path/to/Spatials --out /path/to/P_matrices
"""
import argparse
import numpy as np
import scipy.io
from PIL import Image
from pathlib import Path
from scipy.ndimage import rotate
from tqdm import tqdm


def compute_and_save_P_matrix(frame_list, output_path, rank=2):
    x = np.array(frame_list).astype(np.float32) / 255.0
    casorati = x.reshape(len(frame_list), -1).T            # (H*W, T')
    U, S, Vt = np.linalg.svd(casorati, full_matrices=False)
    P = U[:, :rank]                                          # (H*W, rank)
    scipy.io.savemat(output_path, {'P': P, 'P_T': P.T})


def load_frames(folder, num_frames, size):
    frames = []
    for i in range(1, num_frames + 1):
        p = folder / f'Spatial_{i:03d}.jpg'
        if not p.exists():
            print(f'Warning: {p} does not exist. Skipping folder.')
            return None
        frames.append(np.array(Image.open(p).convert('L').resize((size, size))))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spatials', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--angles', type=int, nargs='+', default=[0, -40, -25, 25, 40])
    ap.add_argument('--num_frames', type=int, default=20)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--rank', type=int, default=2)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    folders = sorted([f for f in Path(a.spatials).iterdir() if f.is_dir()])
    print(f'Processing {len(folders)} folders, angles={a.angles}')
    for folder in tqdm(folders):
        frames = load_frames(folder, a.num_frames, a.size)
        if frames is None:
            continue
        for ang in a.angles:
            if ang == 0:
                f, fr = out / f'{folder.name}_P.mat', frames
            else:
                f = out / f'{folder.name}_P_rot{ang:+d}.mat'
                fr = [rotate(x, ang, reshape=False, mode='constant', cval=0, order=1) for x in frames]
            compute_and_save_P_matrix(fr, f, a.rank)
    print('All done!')


if __name__ == '__main__':
    main()
