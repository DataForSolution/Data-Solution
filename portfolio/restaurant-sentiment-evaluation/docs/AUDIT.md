# Restaurant-review sentiment reconstruction audit

## Historical source

Reviewed:

- `DATA-460_Assign2_Updated_alt.ipynb`

The notebook contains 64 cells and combines:

- restaurant-review EDA,
- manual regex/stopword/stemming preprocessing,
- `CountVectorizer`,
- a 13-model classifier comparison,
- AdaBoost-focused metrics,
- sample-string predictions,
- and a VADER baseline.

The public reconstruction is independently written around scikit-learn/NLTK interfaces rather than copying the notebook's large commented code blocks.

## Assignment-template residue

Several cells retain explicit assignment instructions/placeholders such as “TEXT 1”, “TEXT 7”, and “finish this sentence”. The notebook is therefore best treated as a completed classroom workflow rather than publication-ready source.

## Vocabulary leakage

The notebook performs:

```python
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
```

before K-fold evaluation.

The vocabulary is therefore learned from every review, including the text later treated as held-out in each fold. Text preprocessing itself may be deterministic, but vocabulary selection is a learned preprocessing step and belongs inside the CV pipeline.

The reconstruction places TF-IDF vectorization inside every estimator pipeline.

## Cross-validation design

The broad comparison declares 11-fold ordinary `KFold` with shuffling. Binary sentiment labels should be kept proportionally represented with `StratifiedKFold`.

The notebook later uses a 5-fold `StratifiedKFold` for AdaBoost, which is better, but the globally fitted vocabulary remains.

## Model-zoo selection

The notebook evaluates 13 classifiers on the same CV scheme and highlights high test accuracy. There is no separate nested or final evaluation stage after choosing among those models.

The public project deliberately narrows the candidate set to three standard sparse-text classifiers and describes CV scores as model-selection evidence, not untouched final-test performance.

## Last-fold model state

Inside the AdaBoost CV loop, the same estimator object is repeatedly fitted. After the loop it remains fitted to the final fold's training subset.

The later `predict_sentiment(...)` helper accepts that estimator and the globally fitted vectorizer, then predicts new strings without first refitting a final pipeline on the complete development dataset.

The reconstructed workflow includes an explicit final-refit helper.

## Stateful report labeling

A later cell appends another classification report and prints its index using the existing `classification_reports` list. Saved output labels it `Fold 12` despite the earlier declared 11-fold loop. This is notebook-state residue rather than evidence of a planned 12th fold.

## Saved model-comparison outputs

The historical 11-fold loop reports approximately:

- MultinomialNB: train 94.05%, test 79.40%
- RandomForest: train 99.53%, test 80.20%
- GradientBoosting: train 85.91%, test 78.20%
- AdaBoost: train 82.10%, test 76.70%
- Bagging: train 97.14%, test 77.50%
- ExtraTrees: train 99.53%, test 79.50%
- SVC: train 97.33%, test 80.80%

The large gaps for Random Forest, Extra Trees, Bagging, and SVC are useful overfitting signals, but the globally fitted vocabulary weakens the evaluation.

The later stratified AdaBoost run reports about 82.27% training accuracy and 76.40% testing accuracy.

## VADER boundary

The notebook's VADER function maps:

- compound > 0 → positive,
- compound <= 0 → negative.

Neutral compound score zero is therefore forced into the negative class. VADER is useful as a transparent rule-based baseline but is not trained on the restaurant dataset and should not be treated as directly equivalent to supervised cross-validation.

## Public dataset substitution

The original notebook expects `/content/Data-460_Assign_2-Restaurant_Reviews.csv`, but that CSV is not present in the connected Drive inventory and its provenance/license are not documented in the notebook.

The reconstruction therefore targets UCI dataset 331, *Sentiment Labelled Sentences*, using the Yelp subset. UCI documents 500 positive and 500 negative Yelp sentences and licenses the dataset under CC BY 4.0.

This substitution is explicit and avoids republishing an untraceable classroom CSV.

## Better evaluation sequence

1. Parse/validate the licensed Yelp labels.
2. Stratify folds.
3. Fit vectorizer only on each training fold.
4. Generate out-of-fold predictions.
5. Compare a small declared candidate set.
6. Choose one final estimator.
7. Refit it on all development text.
8. Use a separate final dataset/domain for a stronger generalization claim.
