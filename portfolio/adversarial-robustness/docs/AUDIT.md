# Adversarial robustness reconstruction audit

## Overall assessment

**Historical assignment:** useful evidence of adversarial-ML experimentation, but the added batch-sweep results are not reliable enough to publish as benchmark measurements.

**Reconstructed project:** suitable for a portfolio when presented as reproducibility, evaluation, and robustness-engineering work rather than as proof that the tested defences make ImageNet robust.

## Source/version map

The Google Drive `Colab Notebooks` folder contains a dense September 2024 development sequence:

1. `attack_defence_imagenet_v2.5.ipynb`
2. `Data-attack_defence_imagenet_v2.5.ipynb`
3. `DATA-450-ASSIGNMENT-V-2_Adversarial_Attacks.ipynb`
4. `ON-Progress- Adversarial_Attacks.ipynb`
5. `Data-450-0_Modified_attack_defence_imagenet_v2.5.ipynb`
6. `Data-450-02-Modified_attack_defence_imagenet_v2.ipynb`
7. `Data-450-03-Updated_attack_defence_imagenet_v2.ipynb`
8. `Data-450_updated_attack_defence_imagenet_v2.5_Final.ipynb`
9. `Data-450_updated_attack_defence_imagenet_with_beagle.ipynb`

The gazelle "Final" file is 6,422,446 bytes and the later beagle variant is 5,791,482 bytes. The beagle notebook is the latest assignment-specific variant by modification time, but neither is treated as a clean canonical executable because correctness depends on notebook state and model preprocessing.

Drive also contains `Adversarial_Attacks.ipynb` from April 2025. Its first heading is "Tutorial 10: Adversarial attacks" and its notebook links point to `phlippe/uvadlc_notebooks`. It is third-party tutorial/reference material and must not be represented as original work.

The public `DataForSolution/Data-Solution` repository already contains legacy root-level `Adversarial_Attacks.ipynb` and `Assign_Adversarial_Attacks.ipynb`. They are retained as legacy evidence pending the later repository-cleanup phase.

## Provenance of the DATA 450 starter notebook

The assignment text states that `attack_defence_imagenet_v2` was copied and extended from the Adversarial Robustness Toolbox example. The starter notebook itself describes:

- wrapping a Keras ResNet50 classifier in ART,
- PGD evasion attacks,
- Spatial Smoothing,
- an adaptive attack against the defended classifier,
- and an Ed Herranz extension using additional defences.

The portfolio reconstruction therefore does not copy this base notebook into the new project as original code.

## Assignment-specific work recovered

The later notebooks add or modify:

- gazelle and beagle subject selection,
- Spatial Smoothing parameter experiments,
- Feature Squeezing bit-depth experiments,
- JPEG-compression parameter experiments,
- attempted Gaussian-defence analysis,
- and repeated comparisons across clean/adversarial samples.

Those ideas are the basis of the rebuilt evaluation harness.

## High-impact correctness findings

### 1. Batch sweep bypassed the ResNet50 preprocessing contract

The original starter path created a custom `ResNet50Preprocessor` and passed it into the ART classifier so that ResNet50 received its expected channel/order/mean preprocessing.

The assignment-specific batch sweep later created:

```python
art_model = KerasClassifier(model=model, clip_values=(0, 255))
```

without that preprocessor and then generated/adjudicated attacks on raw `images`.

That is not equivalent to the earlier classifier pipeline. The resulting ImageNet class IDs and confidences from Feature Squeezing, Spatial Smoothing, and JPEG sweeps cannot be treated as reliable benchmark outputs.

### 2. Gazelle notebook depended on stale execution state

In the gazelle "Final" notebook, the Feature Squeezing cell has execution count 39, while the cells that define the new `art_model` and `adversarial_examples` have execution counts 29 and 31 and appear *after* it in notebook order.

This means the saved notebook did not demonstrate a clean top-to-bottom execution path; the cell relied on objects already present in the Colab kernel.

### 3. The Gaussian sweep was not implemented

Both late notebooks import `GaussianAugmentation`, initialize `gaussian_results = []`, but only implement the JPEG loop. No parameterized Gaussian result table is produced.

The reconstruction does not claim that requirement was completed.

### 4. Selected-image and all-image experiments were mixed

The starter flow operates on a selected `x_art` image and an adversarial `x_art_adv`; later added code switches to the entire 16-image `images` batch and a separately generated `adversarial_examples` object.

That is a different experiment. A comparison must state whether it is single-image targeted PGD, all-image untargeted PGD, or an adaptive attack against a defended pipeline.

### 5. `eps=0.1` has a different meaning on a `[0,255]` input scale

The added all-image PGD uses `clip_values=(0, 255)` and `eps=0.1`. On that scale the maximum per-pixel change is only 0.1 intensity units, whereas the starter notebook used `eps=5`.

Attack budgets must be reported in the same units as the model input. Comparing outcomes across these configurations without normalization is misleading.

### 6. Clean utility was not summarized as a first-class metric

Defences can recover adversarial examples while also damaging clean predictions. The historical tables list classes/confidences, but the workflow lacks a stable aggregate definition for:

- clean accuracy before defence,
- adversarial accuracy before defence,
- clean accuracy after defence,
- adversarial accuracy after defence,
- attack success among originally correct examples,
- and recovery among attacked examples.

The reconstructed metrics make those denominators explicit.

### 7. Adaptive evaluation remains essential

ART's own defence documentation warns about defence limitations. Input transformations can appear useful against a fixed attack yet fail against an attacker that differentiates through or otherwise adapts to the defended pipeline.

The reconstructed README therefore avoids calling Spatial Smoothing, JPEG, Feature Squeezing, or noise "solutions."

## Current dependency context

As checked during reconstruction, the latest ART release available on PyPI is 1.20.1. ART continues to provide PGD plus Feature Squeezing, JPEG Compression, Spatial Smoothing, and related preprocessors.

The core portfolio tests do not require ART; ART is optional for reproducing framework-specific experiments.

## What a rigorous future ImageNet experiment should do

1. Freeze the threat model before evaluation:
   - targeted vs. untargeted,
   - norm,
   - epsilon,
   - step size,
   - iterations,
   - restarts.
2. Use one correct preprocessing contract for clean, attacked, and defended inputs.
3. Verify clean baseline predictions before attack.
4. Evaluate attacks only with clearly defined denominators.
5. Sweep defence parameters on a validation subset, not on the final test set.
6. Freeze the defence configuration.
7. Evaluate an adaptive attack against the final defended pipeline.
8. Report clean-accuracy cost and robust-accuracy gain together.
9. Use more than 16 convenience images for any general robustness claim.
10. Preserve exact library/model/weight versions and random seeds.
