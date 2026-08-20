# Chest CT Classification — Portfolio Reconstruction

This project reconstructs a 2024–2025 deep-learning exercise that explored four-class classification of chest CT images using transfer learning.

> **Portfolio / educational use only.** This is not a diagnostic model, has not been clinically validated, and must not be used to diagnose, stage, or guide treatment for lung cancer.

## What the historical project did

The recovered Colab work used the public **Chest CT-Scan images Dataset** and an ImageNet-pretrained ResNet50. The dataset contains 1,000 2D JPG/PNG images across four categories: adenocarcinoma, large-cell carcinoma, squamous-cell carcinoma, and normal. Its supplied split contains 613 training images, 72 validation images, and 315 test images.

The compact historical notebook trained on the 613-image training split and evaluated on the 72-image validation split. Training accuracy increased to about 88.8%, while the final training epoch reported 22.2% validation accuracy and a subsequent validation evaluation reported about 29.2%. The untouched 315-image test split was not used for a final evaluation.

Those results indicate substantial overfitting and are preserved here as an engineering lesson, **not** as evidence of useful clinical performance.

## Important audit findings

- The historical ResNet50 backbone was trainable end-to-end from the start despite the small dataset.
- Inputs were rescaled to `[0, 1]` rather than using the ImageNet ResNet50 preprocessing contract.
- The model used `Flatten()` on convolutional feature maps, creating a much larger classifier head than necessary.
- Validation generators used Keras's default `shuffle=True`. Later classification-report code in the larger notebook compared shuffled predictions with the generator's fixed class array, so those per-class reports are not trustworthy.
- The compact notebook never reported an untouched test-set result.
- The expanded “Fixed” notebook contains duplicated setup code and malformed edits, so it is retained only as historical source evidence, not as the canonical public implementation.

See [`docs/AUDIT.md`](docs/AUDIT.md) for the reconstruction notes.

## Reconstructed approach

The public code demonstrates a safer and more reproducible experiment design:

1. Validate the expected dataset structure and split counts before training.
2. Keep validation and test ordering deterministic.
3. Use `tf.keras.applications.resnet50.preprocess_input`.
4. Freeze the ImageNet backbone for initial transfer learning.
5. Use global average pooling instead of flattening the entire feature map.
6. Evaluate with accuracy, balanced accuracy, macro-F1, and a confusion matrix.
7. Reserve the supplied test split for one final evaluation after model/threshold choices are frozen.

## Dataset

Source: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

The Kaggle data card currently identifies the database as **Open Database** while the image contents remain **© Original Authors**. This repository therefore does **not** redistribute the CT images. Download the dataset from its source and point the code to your local `Data/` directory.

Expected supplied split:

| Split | Images |
| --- | ---: |
| train | 613 |
| valid | 72 |
| test | 315 |
| **total** | **1,000** |

## Repository structure

```text
src/chest_ct_portfolio/
  dataset.py       # split/class normalization and dataset validation
  evaluation.py    # deterministic classification metrics
  modeling.py      # corrected ResNet50 transfer-learning reference

tests/
  test_dataset.py
  test_evaluation.py

docs/
  AUDIT.md

notebooks/
  evaluation_demo.ipynb
```

## Install and test

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

python -m pip install -e ".[dev]"
pytest
```

To install the optional current TensorFlow training stack:

```bash
python -m pip install -e ".[ml]"
```

## What this project demonstrates

- medical-imaging-aware ML review,
- transfer-learning methodology,
- dataset-contract validation,
- reproducible evaluation,
- identification of overfitting and label-order bugs,
- and responsible separation of an educational experiment from clinical claims.
