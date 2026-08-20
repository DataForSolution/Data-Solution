# CIFAR-10 generated-image analysis audit

## Assessment

The historical GAN and CNN implementations are tutorial-derived and should not be presented as original portfolio implementations. The later assignment-specific classifier analysis is the part worth preserving, after correcting its interpretation and reproducibility issues.

## Recovered versions

Google Drive contains two `Data-450-Assign-4B_CIFAR10GAN2_v2.ipynb` files (337,093 and 4,669,476 bytes) with identical cell source; the size difference is saved notebook output/state. A later `Data-450-Assign-4B_CIFAR10GAN-Complete-1_.ipynb` (1,826,729 bytes, modified October 3, 2024) adds the assignment-specific analysis. A separate `cifar10_fake_images_analysis.ipynb` is a random-data experiment rather than a run of the saved GAN.

The old GitHub Pages repository also contains multiple CIFAR project-page copies and a `projects/CIFAR10GAN_Final_.ipynb`. Those remain legacy artifacts pending the later cleanup phase.

## Provenance

The notebooks explicitly credit Jason Brownlee's Machine Learning Mastery tutorials for the DCGAN and CIFAR-10 CNN code. The reconstructed portfolio therefore implements only an independent analysis layer around classifier outputs.

## Findings

### Classifier confidence was mislabeled as realism

The completed notebook selects generated images with the largest maximum softmax value and calls them the “most real.” Maximum classifier probability measures classifier confidence, not perceptual realism or generative fidelity. The reconstruction uses the term `classifier_confidence` and does not turn it into a realism claim.

### The separate random-data notebook does not contain classifier probabilities

That notebook creates random image arrays and `np.random.rand(100, 10)` scores, then describes those scores as softmax probabilities. The rows are not normalized to sum to one. Its confidence summaries are therefore not valid probability-based evidence and are excluded from the reconstruction.

### Empty-class handling was unsafe

The random-data notebook sets no seed and immediately computes minimum, mean, and maximum for Deer and Bird subsets. A run with no predicted samples for one of those classes could fail. The reconstructed class summary handles zero-count classes explicitly.

### Some images were rescaled twice for display

The completed notebook converts generator output from `[-1,1]` to `[0,1]`, then later visualization cells apply `(X + 1) / 2` again. That second conversion shifts an already normalized image into `[0.5,1]`. The reconstruction uses an explicit source-range conversion helper and validates bounds.

### Inference was repeated one image at a time

The historical analysis repeatedly calls the classifier for individual samples, sometimes more than once for the same image. The reconstruction accepts one batched probability matrix and derives all summaries from it.

### Historical confidence values are not a GAN benchmark

The saved completed run reported high classifier confidence for generated samples predicted as Deer and Bird. Those numbers describe a fixed classifier's response; they do not establish realism, correct class semantics, or overall generator quality.

### Exact historical rerun is currently incomplete

The completed notebook loads `generator_model_200.h5` and `final_model.h5`. Those artifacts were not discoverable in the connected Drive during reconstruction, so the exact generated batch cannot be reproduced from the available files alone.

## Better future evaluation

A stronger GAN study should preserve exact weights/configuration/seeds, generate a fixed evaluation sample, use batched classifier diagnostics only as a semantic proxy, and add proper generative-quality checks such as FID/KID, diversity or mode-collapse analysis, and nearest-neighbor/memorization checks. Classifier confidence should never be equated with realism.
