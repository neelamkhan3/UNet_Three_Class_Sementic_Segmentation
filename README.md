# U-Net for Three-Class Semantic Segmentation (Background / Weed / Wheat)

Implementation of a U-Net in TensorFlow-Keras (Google Colab) that segments
field images into three classes: background, weed and wheat.
Dice scores are computed with `sklearn.metrics.f1_score`.

Open In Colab: https://colab.research.google.com/drive/1rMhQYxPKD8qYsgjrRGO4-hmpGs7hIx7W?usp=drive_link

- ## Dataset
- 936 RGB images (480x352) with single-channel masks (0 = background, 1 = weed, 2 = wheat)
- Pixel share: background 65.87%, weed 14.43%, wheat 19.71%
- The dataset is **not stored in this repository**. The notebook downloads it
  automatically from Google Drive using `gdown`
- Open the Drive link:  https://drive.google.com/file/d/1t5Z4ekEWc1jmblaEoh88mtQgMQAhrQjm/view?usp=d
    rive_link
- Dataset provided as part of the course assignment.

## What was done
1. **Sanity check:** a standard U-Net (4 levels, 64 filters) overfits 4 images
   (training loss 0.003, inference Dice 0.992 after BatchNorm recalibration).
2. **Depth study:** 1 to 5 encoder/decoder levels.
3. **Loss comparison:** cross-entropy vs cross-entropy + Dice loss.
4. **Augmentation:** flips and 90-degree rotations vs none.
5. **Extra analysis:** training curves, confusion matrix, example predictions.

All experiments use the same seed (reset before every run), the same validation
split and 20 epochs.

## Results (test set, macro Dice)

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

## Key findings
- Three encoder/decoder levels are enough; deeper models add parameters, not accuracy.
- Dice loss and augmentation gave no measurable improvement in this setup.
- Weed is the hardest class (Dice about 0.80, recall 71.6%); background and
  wheat are recognized well.
- Differences between most configurations are about 0.002, similar to the
  run-to-run noise, so they should not be over-interpreted.

## Limitations
- Single seed per configuration; 20-epoch training budget (most runs were still improving).
- Images resized to 256x256, which changes the aspect ratio.

## Files
- `unet_three_class_sementic_segmentation.ipynb`: full notebook
- `figures/`: training curves, example predictions, confusion matrix

## How to run
Open the notebook in Google Colab, select a GPU runtime
(Runtime, Change runtime type, T4 GPU) and run all cells in order.

## Requirements
Python 3, TensorFlow 2.x, NumPy, Pillow, scikit-learn, matplotlib, pandas, gdown
