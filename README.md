# 🌾 U-Net for Three-Class Semantic Segmentation

A U-Net implemented in TensorFlow-Keras that segments field images into **background, weed and wheat**, with experiments on network depth, loss function and data augmentation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neelamkhan3/YOUR-REPO-NAME/blob/main/YOUR-NOTEBOOK-NAME.ipynb)

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Technologies](#-technologies-used)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation and Usage](#-installation-and-usage)
- [Results](#-results)
- [Figures](#-figures)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

## 🎯 Overview
This project builds a U-Net for pixel-level classification of field images into three classes. Dice scores are computed with `sklearn.metrics.f1_score` on a held-out test set.

**Objectives**
1. Build a standard U-Net and overfit it on a few images as a sanity check.
2. Find how many encoder/decoder levels are enough.
3. Compare cross-entropy with a composite cross-entropy + Dice loss.
4. Compare training with and without data augmentation.

## ✨ Features
- Configurable-depth U-Net (1 to 5 levels)
- Composite loss: cross-entropy + Dice
- Paired image/mask augmentation (flips and 90° rotations)
- Reproducible experiments (seed reset before every run)
- Training curves, confusion matrix and example predictions

## 🛠️ Technologies Used
| Technology | Purpose |
|---|---|
| Python 3 | Programming language |
| TensorFlow / Keras | Model building and training |
| NumPy, Pillow | Data loading and preprocessing |
| scikit-learn | `train_test_split` and `f1_score` (Dice) |
| Matplotlib, pandas | Plots and result tables |
| Google Colab (T4 GPU) | Training environment |

## 📊 Dataset
- 936 RGB images (480x352) with single-channel masks: 0 = background, 1 = weed, 2 = wheat
- Pixel share: background 65.87%, weed 14.43%, wheat 19.71%
- Resized to 256x256; images scaled to [0, 1]; masks resized with nearest-neighbour
- Split: 748 training / 188 test images (`train_test_split`, 80/20, `random_state=42`); 10% of training data is used for validation
- The dataset is **not stored in this repository**. The notebook downloads it from Google Drive with `gdown`.
- Dataset Link:  https://drive.google.com/file/d/1t5Z4ekEWc1jmblaEoh88mtQgMQAhrQjm/view?usp=drive_link 

## 🧠 Model Architecture
```
Input (256x256x3)
   ↓  [2x Conv3x3 + BatchNorm + ReLU] → skip connection →  MaxPool   (repeated per level, filters double)
   ↓  Bottleneck (+ 10% dropout)
   ↓  [Transposed Conv ↑ → Concatenate skip → 2x Conv3x3 + BatchNorm + ReLU]   (repeated per level)
   ↓  1x1 Conv, softmax (3 classes)
Output (256x256x3 class probabilities)
```
- **Base width:** 16 filters, doubled at each level (the sanity check uses the standard 4 levels with 64 base filters)
- **Optimizer:** Adam, learning rate 1e-3, batch size 4
- **Training:** 20 epochs, early stopping (patience 5), learning-rate reduction on plateau
- **Losses:** cross-entropy, and cross-entropy + soft Dice (equal weights)

## 🚀 Installation and Usage
**In Google Colab (recommended)**
1. Click the "Open in Colab" badge above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all cells in order. The dataset downloads automatically.

**Notebook steps:** setup and seeds → download data → pair images and masks → preprocessing and 80/20 split → U-Net builder → sanity check → Dice evaluation → depth, loss and augmentation experiments → curves, confusion matrix and predictions.

## 📈 Results
**Sanity check:** standard U-Net on 4 images reached a training loss of 0.003 and a Dice of 0.992 (after BatchNorm recalibration).

**Test-set macro Dice (`f1_score`)**

| Configuration | Macro Dice | Background | Weed | Wheat |
|---|---|---|---|---|
| Depth 1, CE | 0.8701 | 0.9618 | 0.7856 | 0.8628 |
| Depth 2, CE | 0.8750 | 0.9614 | 0.7934 | 0.8701 |
| Depth 3, CE | 0.8765 | 0.9606 | 0.7958 | 0.8730 |
| Depth 4, CE | 0.8775 | 0.9617 | 0.7962 | 0.8746 |
| Depth 5, CE | 0.8767 | 0.9614 | 0.7953 | 0.8736 |
| Depth 3, CE + Dice | 0.8761 | 0.9623 | 0.7960 | 0.8699 |
| Depth 3, CE + Dice, no augmentation | 0.8775 | 0.9614 | 0.7971 | 0.8739 |
| Depth 3, CE + Dice, with augmentation | 0.8759 | 0.9619 | 0.7955 | 0.8702 |

**Key findings**
- ✅ Three encoder/decoder levels are enough; deeper models add parameters, not accuracy.
- ✅ Dice loss and augmentation gave no measurable gain in this setup.
- ✅ Best macro Dice: **0.8775** (depth 4, CE).
- ⚠️ Weed is the hardest class (Dice about 0.80, recall 71.6%).
- ⚠️ Most differences are about 0.002, similar to run-to-run noise, so they should not be over-interpreted.

## 📁 Project Structure
```
YOUR-REPO-NAME/
├── YOUR-NOTEBOOK-NAME.ipynb   # Main notebook
├── README.md
├── requirements.txt           # Python dependencies
├── LICENSE
├── .gitignore
└── figures/
    ├── training_curves.png
    ├── example_predictions.png
    └── confusion_matrix.png
```

## 🔧 Troubleshooting
**Out-of-memory error:** reduce `BATCH_SIZE` (try 2), or use a smaller base width (`base_filters=16`).

**Slow training:** make sure a GPU runtime is selected. Disabling `enable_op_determinism` in `set_seed` can speed training up, at the cost of exact reproducibility.

## 💡 Future Improvements
- [ ] Repeat key comparisons with several seeds (mean ± std)
- [ ] Train longer (most runs were still improving at epoch 20)
- [ ] Class weighting or focal loss to improve weed recall
- [ ] Higher input resolution without distorting the aspect ratio
- [ ] Stronger augmentation (brightness, noise, elastic distortion)
- [ ] Try pretrained encoders (e.g. ResNet or EfficientNet backbones)

## 👤 Author
**Neelam Khan**
- GitHub: https://github.com/neelamkhan3
- LinkedIn: www.linkedin.com/in/neelamkhan3

## 📄 License
Released under the MIT License. See the `LICENSE` file for details. The dataset has its own terms and is not included here.
