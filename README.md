# Label-Scarce Ultrasound Biopsy Needle Identification using Dynamics-Guided Prior Modeling

Official code for the semi-supervised dynamics-based unrolling framework for ultrasound biopsy needle segmentation
(*Diagnostics*, MDPI; manuscript ID: diagnostics-4599565).

The network (**UnrollNet**) unrolls N iterations that alternate between a U-Net regularization layer (R) and a
null-space projection-based Analytic Model layer (A, solved with conjugate gradient). A supervised teacher
(N = 5, frozen) generates soft pseudo-labels for unlabeled frames, and a randomly initialized student with the same
architecture is trained with a supervised loss on labeled data and a pseudo loss on unlabeled data
(L = L_sup + λ_p · L_pseudo, λ_p = 0.7).

## Repository structure

```
.
├── models/
│   ├── UnrollNet_seg.py      # UnrollNet (R + A layers, N unrolls)
│   ├── unet.py               # U-Net regularizer
│   ├── basic_module.py
│   └── utils.py              # collate function, Dice
├── data/
│   └── dataset.py            # NeedleDataset
├── scripts/
│   ├── train_teacher.py      # supervised UnrollNet (teacher, N = 5)
│   ├── train_student.py      # self-training student (N = 5)
│   └── test.py               # evaluation (Dice, IoU, HD95, trajectory/tip error)
├── requirements.txt
└── README.md
```

## Environment

Tested with PyTorch 2.6.0, CUDA 12.4, one NVIDIA RTX 3090 (24 GB), Ubuntu 18.04.6.

```bash
conda create -n unrollnet python=3.9 -y && conda activate unrollnet
pip install -r requirements.txt
```

## Data

The ultrasound data were collected from renal biopsy procedures at two institutions (Philips iU22, curved array
transducer) under IRB approval and **are not included in this repository**. They are available from the
corresponding authors upon reasonable request. The scripts expect the following layout under `data/raw/`
(change `base_dir`, `matches_file`, and `p_matrices_dir` in the scripts if you use another location):

```
data/raw/
├── train/             # labeled training frames (+ pre-computed rotation augmentation)
├── val/               # labeled validation frames
├── test/              # labeled test frames
├── unlabeled/         # unlabeled frames (used only by the student)
├── P_matrices/        # background-subspace matrices P (SVD of the first T' = 20 frames per sequence)
└── top1_matches.txt   # frame-to-P-matrix matching file
```

Labeled data: 181 frames (148 train / 18 validation / 15 test, split at the video level) and 2,653 unlabeled frames.
Training frames are augmented by rotations of ±25° and ±40° (5× the original number).

## Training

Run all commands from the repository root.

```bash
# 1) Teacher: supervised UnrollNet (N = 5, 10 CG steps)  ->  ./checkpoints_teacher/
python scripts/train_teacher.py

# 2) Student: self-training with the frozen teacher     ->  ./checkpoints_student/
python scripts/train_student.py
```

Training settings (both stages): Adam, learning rate 1e-4, batch size 4, up to 100 epochs, early stopping with
patience 30 on the validation Dice of the final A-layer output, MSE loss with deep supervision over all intermediate
R and A outputs. The student additionally uses ReduceLROnPlateau (factor 0.5, patience 10), is randomly initialized,
and uses λ_p = 0.7. `train_student.py` loads the latest `best_model_DC_*.pth` in `./checkpoints_teacher/` as the teacher.

## Evaluation

```bash
python scripts/test.py --model ./checkpoints_student/best_model_DC_XXXX.pth --out ./eval
```

Metrics: Dice, IoU, HD95, trajectory angle error (Hough transform with RANSAC fallback), and tip-position error
(pixel spacing 0.1 mm/pixel). Inference time is measured for the network forward pass only (batch size 1).

## Main results (15 test frames, mean ± SD)

| Method | Dice | IoU | HD95 (px) | Traj. error | Tip error |
|---|---|---|---|---|---|
| UnrollNet (sup.) | 0.7298 ± 0.0518 | 0.5808 ± 0.1131 | 13.76 ± 12.91 | 1.20° ± 0.94° | 1.08 mm ± 0.77 mm |
| **Ours (SSL student)** | **0.7429 ± 0.0489** | **0.5967 ± 0.0917** | 10.69 ± 8.72 | **0.66° ± 0.63°** | 0.95 mm ± 0.74 mm |

## Citation

```bibtex
@article{TODO,
  title   = {Label-Scarce Ultrasound Biopsy Needle Identification using Dynamics-Guided Prior Modeling},
  author  = {TODO},
  journal = {Diagnostics},
  year    = {2026}
}
```

## License

TODO (e.g., MIT for the code). The data are not covered by this license.
