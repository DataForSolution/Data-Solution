# Pima diabetes ML reconstruction audit

## Source inventory

Reviewed Drive artifacts:

- `DATA440SVMIndianDiabeAssign1.ipynb`
- `SVM_Indian_Diabetic.ipynb`
- `Data440Assi2IndiDiabNN.ipynb`
- `Data440AssinEX1aSVM.ipynb`
- `data440EX1BIndianDiabetic_datasets.ipynb`

The small `data440EX1B...` file simply demonstrates loading German Credit, diabetes, and Spambase from OpenML. `Data440AssinEX1aSVM.ipynb` is based on the unrelated Mice Protein dataset and is not part of the canonical diabetes model comparison.

## Dataset provenance

The diabetes notebooks explicitly load OpenML dataset 37 and identify it as the Pima Indians Diabetes Database. OpenML describes 768 instances, eight predictors plus class, with original ownership attributed to the National Institute of Diabetes and Digestive and Kidney Diseases.

Published analyses describe the sample as 768 women at least 21 years old of Pima Indian heritage, including 268 positive and 500 negative cases. This population boundary must remain visible in any portfolio description.

## SVM notebook lineage

After accounting for one extra package-install cell in the later notebook, approximately 87.5% of the cell sequence is identical between the two large SVM files. Their saved kernel results are identical:

| Kernel | Saved test accuracy |
| --- | ---: |
| linear | 0.7857 |
| rbf | 0.7662 |
| poly | 0.7532 |

For binary SVC, `decision_function_shape='ovo'` versus `'ovr'` does not create meaningfully different binary classifiers, which is why the saved accuracies are identical within each kernel.

### Test-set model selection

The notebook loops over kernels and prints performance on `X_test`, making the test set part of configuration comparison. The same split should not then be treated as an untouched estimate of generalization.

### No feature standardization

SVMs are sensitive to feature scale, while Pima features range from small pedigree-function values to insulin values in the hundreds. The reconstructed pipeline standardizes after train-only imputation.

### Zero measurements treated as real values

The historical `dropna()` removes conventional nulls but does not address zero-valued glucose, blood pressure, skin thickness, insulin, or BMI. Published analyses document these as physiologically implausible/missing measurements.

### Redundant age feature

The SVM notebook creates categorical `age_group` dummy columns while retaining continuous `age`. This is legal but redundant and adds arbitrary bin boundaries. The public baseline keeps the original eight features.

## MLP notebook

The neural-network notebook loads the same OpenML dataset.

### Preprocessing leakage

The final preprocessing cell performs:

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, ... = train_test_split(X, ...)
```

The scaler therefore learns means/standard deviations from the future test rows. The reconstruction puts `StandardScaler` after the split inside a `Pipeline`.

### Test set reused for architecture selection

The notebook evaluates default MLP, `(30,30,30)` ReLU, and `(20,20,20)` tanh variants on the same test set. Saved accuracies include approximately:

- default MLP: 0.714
- `(30,30,30)` ReLU: 0.688
- `(20,20,20)` tanh: 0.747

Those are useful learning observations but not an unbiased final model comparison.

## Better evaluation contract

The reconstruction:

- uses OpenML-37 schema names explicitly;
- treats only clinically implausible zero measurements as missing;
- learns median imputation and scaling from training data only;
- uses stratified train/validation/test partitions;
- reserves the final test split until after model choice;
- reports sensitivity and specificity so performance on the positive class is not hidden by the 500/268 class imbalance;
- keeps model claims educational rather than diagnostic.

## Reproducibility note

The public source does not redistribute the raw dataset. It documents OpenML data ID 37 and keeps data-dependent code separate from the testable preprocessing/evaluation contract.
