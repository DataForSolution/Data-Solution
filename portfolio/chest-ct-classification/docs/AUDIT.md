# Chest CT CNN reconstruction audit

## Overall assessment

**Historical model: needs revision before any performance claim.**  
**Reconstructed portfolio code: suitable for an educational engineering portfolio after verification.**

## Source artifacts reviewed

Google Drive contained a dedicated `Chest_CT_Scan_CNN_Project` folder with:

- `Chest_CT-Scan_CNN.ipynb`
- `Chest_CT-Scan_CNN_Fixed.ipynb`

A later Colab notebook, `chest_ct_scan_model.ipynb`, provided a much smaller and clearer training attempt and is treated as the best historical summary of the intended model.

GitHub also contained older root-level chest-CT notebooks. Those legacy copies are not treated as canonical because they are partial/duplicated development artifacts.

## Dataset and provenance

The experiment used the Kaggle `mohamedhanyyy/chest-ctscan-images` dataset. Its public data card describes four categories and train/valid/test folders. A secondary dataset catalog reports the supplied split as 613 train, 72 validation, and 315 test images (1,000 total).

The Kaggle license display states: Database: Open Database; Contents: © Original Authors. The reconstruction therefore links to the source but does not redistribute image contents.

## Historical methodology findings

### 1. Strong overfitting signal

The compact ResNet50 run increased training accuracy from roughly 35% in epoch 1 to roughly 89% in epoch 10 while validation accuracy stayed low and unstable. The final epoch showed about 22% validation accuracy; a later `model.evaluate()` call reported about 29% on the 72-image validation set.

The gap is too large to present the model as generalizing reliably.

### 2. The test set was not used for a final untouched evaluation

The dataset contains a separate test directory with 315 images, but the compact historical notebook only created train and validation generators. A final held-out test metric was therefore not established.

### 3. Transfer-learning setup was too aggressive for the sample size

The historical model inserted a full ImageNet ResNet50 into a Sequential model without freezing the backbone. Fine-tuning the entire network immediately on 613 training images increases overfitting risk and can destroy useful pretrained features.

The reconstruction freezes the backbone for the initial phase and makes later fine-tuning an explicit separate decision.

### 4. Input preprocessing did not match the pretrained ResNet50 contract

The historical generator used `rescale=1/255`. The reconstructed reference model applies `tf.keras.applications.resnet50.preprocess_input`, matching the pretrained model's expected preprocessing.

### 5. Classifier head was unnecessarily large

The historical model used `Flatten()` after ResNet50 convolutional features. The reconstructed model uses `GlobalAveragePooling2D()`, reducing the parameter count and overfitting pressure.

### 6. Expanded-notebook classification reports were vulnerable to label-order mismatch

The historical `flow_from_directory()` validation generator did not set `shuffle=False`, so Keras defaulted to shuffling. Later prediction/classification-report code compared predictions to `validation_generator.classes`, whose order is the underlying file order rather than the shuffled prediction order. Those per-class metrics and confusion matrices cannot be trusted unless ordering is made deterministic and the generator is reset appropriately.

### 7. The “Fixed” notebook is not a clean canonical artifact

The fixed notebook contains 106 code cells, no narrative markdown, duplicated data-extraction/setup blocks, and malformed automated edits such as a broken Colab-import replacement. It is useful as historical evidence but not suitable as the public implementation.

## Responsible interpretation

The dataset consists of 2D JPG/PNG images rather than original DICOM studies, and the reconstruction does not have verified patient-level identifiers or acquisition metadata. Therefore it cannot establish patient-level split independence, scanner/site generalization, or resistance to image-processing artifacts.

The portfolio should describe this work as an **educational chest-CT image classification experiment**. It should not claim cancer diagnosis, clinical sensitivity/specificity, staging capability, or patient-care readiness.

## Correct evaluation sequence for future work

1. Validate image counts, class folders, duplicates, and patient-level independence where identifiers permit.
2. Freeze the supplied test split before model selection.
3. Train only on the training split.
4. Use the validation split for architecture/hyperparameter decisions.
5. Freeze the final pipeline.
6. Evaluate once on the untouched test split with deterministic ordering.
7. Report accuracy, balanced accuracy, macro-F1, class-wise sensitivity/recall, confusion matrix, and confidence intervals.
8. For clinically meaningful work, move to well-curated DICOM data with patient-level labels, site/scanner metadata, and external validation.
