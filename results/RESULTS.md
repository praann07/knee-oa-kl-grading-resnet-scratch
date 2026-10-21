# Knee OA KL-grade classification - final results (v5, from scratch)

Generated 2026-10-21 07:48. 5 classes (KL 0-4). **No pretrained weights**: random initialisation; the only pre-training is self-supervised rotation prediction on our own training images (no labels, no outside data).

Models: 4 trained (3 architectures x seeds); **4 selected on validation kappa**: d_s2, d_s1, d_s3, d_s4. Per-class bias tuned on validation (macro_f1). Nothing was selected or tuned on test.

## Data

- train 4,911 / validation 867 (85/15 split of the 5,778-image train folder)
- test: **4008** images = test (1656, the official split) + auto_test + val folders

## Results

| set | n | accuracy | balanced acc | macro F1 | weighted F1 | quad. kappa | per-class F1 (G0..G4) |
|---|---|---|---|---|---|---|---|
| all test, raw | 4008 | 65.82% | 0.684 | 0.672 | 0.658 | 0.819 | 0.76 / 0.33 / 0.63 / 0.81 / 0.83 |
| all test, bias-tuned (final) | 4008 | 65.67% | 0.661 | 0.665 | 0.648 | 0.813 | 0.77 / 0.31 / 0.59 / 0.81 / 0.84 |
| official test, raw | 1656 | 66.61% | 0.689 | 0.677 | 0.665 | 0.833 | 0.77 / 0.33 / 0.65 / 0.82 / 0.83 |
| official test, bias-tuned (final) | 1656 | 65.94% | 0.657 | 0.661 | 0.650 | 0.825 | 0.78 / 0.29 / 0.60 / 0.81 / 0.82 |
| all test, all 4 models raw | 4008 | 65.82% | 0.684 | 0.672 | 0.658 | 0.819 | 0.76 / 0.33 / 0.63 / 0.81 / 0.83 |

Previous versions on the same 4,008 images: v1 46.1% / kappa 0.416, v2 31.1% / 0.460, v3 51.0% / 0.579, v4 54.5% (raw) / 0.625, 55.3% (tuned) / 0.681.

## Comparison with published work (numbers as reported by the authors; test sets differ)

| work | pretrained? | test | result |
|---|---|---|---|
| Ensemble of 8 CNNs, Scientific Reports 2023 | yes (ImageNet) | 5-fold CV | 76.9% accuracy |
| EfficientNetB5 baseline, Diagnostics 2025 | yes | official 1,656 | 66.4% accuracy, macro F1 0.602 |
| ResNet-101 / Xception (cited in that paper) | yes | same dataset | 69% / 67.8% |
| from-scratch CNN, GitHub (afzall003) | no | official 1,656 | balanced acc 0.644, kappa 0.753 |
| **this work (from scratch)** | no | official 1,656 | **65.9%** accuracy, balanced acc 0.657, kappa 0.825 (final); raw 66.6% / 0.833 |

The 82% reported in the 2025 CleanLab paper is scored against *relabelled* test labels and is not comparable (on the original labels the improvement was not significant, McNemar p = 0.076).

## Honest notes

- G1 (Doubtful) and G2 (Minimal) remain the hardest grades; most errors are to a neighbouring grade.
- Bias tuning trades some accuracy for better F1/kappa on G1-G2; both raw and tuned are reported.
- Research prototype, not a medical device.

## Files

`knee_oa_model.pt` + `predict.py` (laptop inference), `predictions_all_test.csv`, `results.json`, plots, `demo_images/`, `backup/all_model_weights.zip`.
