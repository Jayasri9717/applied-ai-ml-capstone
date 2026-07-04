# Part 2 — Supervised Machine Learning: Regression & Classification

## 1. Overview

This part builds two predictive models on top of `cleaned_data.csv` from
Part 1:

- **Regression:** predict `annual_charges` (continuous) using `LinearRegression`
  and `Ridge`.
- **Classification:** predict `smoker` status (binary) using `LogisticRegression`.

### Label definitions

- **`y_reg`** = `annual_charges` — the continuous target already established
  in Part 1.
- **`y_clf`** = `smoker`, binarized to 1 (yes) / 0 (no). We deliberately did
  **not** use the median-split of `annual_charges` suggested as the default
  option in the brief, because that split comes out **exactly 750/750** on
  this dataset — a perfectly balanced target with nothing to demonstrate for
  the required class-imbalance-handling task. The brief explicitly allows
  "another natural binary column in the dataset," and `smoker` is exactly
  that: a genuinely imbalanced (~22% positive) binary column that makes the
  imbalance-handling and threshold-tuning tasks meaningful.

### Two feature matrices — one important leakage-avoidance decision

We built **two different feature matrices**, not one shared `X`, to avoid
leakage in *both* directions:

- **`X_reg`** excludes only `annual_charges` (its own target). Critically, it
  **includes `smoker`** as a feature — `smoker` is not the regression target,
  and it is in fact the single strongest driver of `annual_charges`
  identified in Part 1. Excluding it would artificially and unrealistically
  cripple the regression model's ability to predict cost.
- **`X_clf`** excludes **both** `annual_charges` and `smoker`, since `smoker`
  *is* the classification target here — leaving it in `X_clf` would let the
  model trivially "predict" its own label with ~100% accuracy, which would
  be a textbook case of target leakage.

## 2. How to Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute part2_ml_models.ipynb --output part2_ml_models.ipynb
```

Or open `part2_ml_models.ipynb` in Jupyter and run all cells top-to-bottom.
`cleaned_data.csv` (carried over from Part 1) must be in the same folder.
Produces `plots/roc_curve.png`.

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`,
`matplotlib` (see `requirements.txt`).

## 3. Categorical Encoding Decisions (Task 2)

| Column | Encoding | Justification |
|---|---|---|
| `exercise_freq` | **Ordinal label encoding** (none=0, low=1, moderate=2, high=3) | This column has a genuine natural order — "high" exercise frequency is objectively more than "moderate," which is more than "low," which is more than "none." Encoding it as ordered integers preserves this real ranking, letting the model use it as a single meaningful numeric scale. |
| `sex`, `smoker` (regression only), `region` | **One-hot encoding**, `drop_first=True` | These categories have **no inherent order** — there is no sense in which "male" > "female," or "northeast" > "southwest." Label-encoding them as arbitrary integers would falsely imply an ordinal/numeric relationship (e.g. that "southeast" is "twice" "northwest"), which the model could spuriously learn to exploit. One-hot encoding avoids this by giving each category its own independent binary indicator column, with one category dropped as the reference level to avoid multicollinearity (the "dummy variable trap"). |

`exercise_freq` carried ~25% missing values from Part 1 (it exceeded the 20%
null threshold there and was intentionally left un-imputed). Since ordinal
encoding requires every row to have a category, we **mode-filled** it here
(mode = `"low"`) before mapping to integers.

## 4. Leak-Free Train-Test Split & Scaling (Task 3)

Both feature matrices were split 80/20 with `random_state=42`. For each,
`StandardScaler` was **fit only on the training split**, then used to
transform both the training and test splits.

**Why fitting the scaler on the full dataset would be leakage:** if the
scaler were fit on `X` before splitting, its learned mean and standard
deviation would be computed using data that includes the test set. The model
would then be trained on features that have already been standardized using
information about the test set's distribution — effectively letting
information from the "unseen" test set leak into training. This inflates
apparent performance in a way that would not hold up on genuinely new data,
which defeats the purpose of a held-out test set.

## 5. Regression Results (Task 4)

| Model | MSE | R² |
|---|---|---|
| Linear Regression | 47,595,098.38 | **0.5928** |
| Ridge (alpha=1.0) | 47,608,932.27 | 0.5927 |

**Coefficients (Linear Regression, sorted by absolute value):**

| Feature | Coefficient |
|---|---|
| `smoker_yes` | **+7,149.50** |
| `prior_conditions` | +1,193.76 |
| `children` | +709.02 |
| `exercise_freq` | −675.32 |
| `age` | +513.60 |
| `bmi` | +328.21 |
| `region_northwest` | −171.03 |
| `region_southwest` | −167.48 |
| `region_southeast` | +92.15 |
| `sex_male` | −2.65 |

**Top 3 features by absolute coefficient:** `smoker_yes`, `prior_conditions`,
`children`.

**Interpreting the coefficients:** because all features were standardized
(scaled to mean 0, standard deviation 1) before fitting, each coefficient
represents the change in predicted `annual_charges` associated with a
**one-standard-deviation increase** in that feature, holding all other
features constant. A **large positive coefficient** (e.g. `smoker_yes` =
+7,149.50) means that being a smoker is associated with roughly $7,150 higher
predicted annual charges compared to a non-smoker, all else equal — this
matches the ~3.5x cost gap for smokers visually confirmed in Part 1's box
plot. A **large negative coefficient** (e.g. `exercise_freq` = −675.32) means
that moving one level higher on the exercise-frequency scale (e.g. from "low"
to "moderate") is associated with roughly $675 **lower** predicted charges,
consistent with more exercise being protective against higher medical costs
in this dataset.

### Ridge vs. Linear Regression

The MSE and R² for Ridge (alpha=1.0) are nearly identical to plain Linear
Regression (R² 0.5927 vs 0.5928), and the individual coefficients are only
marginally smaller in magnitude across the board (e.g. `smoker_yes` shrinks
from 7,149.50 to 7,143.52). **Why Ridge can produce a different coefficient
profile than OLS:** Ridge adds an L2 penalty term (`alpha * sum(coef^2)`) to
the loss function being minimized, which discourages any single coefficient
from growing too large — it trades a small amount of bias for a reduction in
variance, shrinking all coefficients toward zero proportionally to their
size. The **`alpha` parameter controls the strength of this penalty**: higher
alpha shrinks coefficients more aggressively (more bias, less variance,
better protection against overfitting on noisy or collinear features), while
alpha=0 recovers plain OLS exactly. Here, the near-identical results indicate
this dataset's features aren't strongly collinear or noisy enough for
alpha=1.0 to meaningfully change the fitted model — a stronger penalty (much
larger alpha) would be needed to see a bigger divergence.

## 6. Classification Results (Task 5)

### Class imbalance

`y_clf_train` value counts: **932 non-smokers (0), 268 smokers (1)** — the
minority class share is **22.3%**, below the 35% threshold, so imbalance
handling was required.

**Chosen strategy: `class_weight='balanced'`** in `LogisticRegression`,
rather than SMOTE. Reasoning: SMOTE synthesizes new minority-class rows by
interpolating between real ones in feature space, which can generate
biologically implausible combinations for a small, mixed
categorical/numeric feature set like this one (e.g. blending BMI and age
values in ways that don't correspond to any real employee profile).
`class_weight='balanced'` instead reweights the existing, real training
rows so the loss function penalizes misclassifying the minority class more
heavily — no synthetic rows are created, and it is simpler and equally
effective here.

**Before/after comparison** (for illustration, we also ran SMOTE on the same
split to show the contrast):

| | Class 0 (no) | Class 1 (yes) |
|---|---|---|
| **Before** (raw training counts) | 932 | 268 |
| **After (if SMOTE were used instead)** | 932 | 932 |

`class_weight='balanced'` does not change the row counts shown above (it
reweights the loss internally); the "before" counts remain the actual
training data used.

### Confusion matrix & classification report

```
Confusion Matrix:
[[138  99]
 [ 29  34]]

              precision    recall  f1-score   support
           0       0.83      0.58      0.68       237
           1       0.26      0.54      0.35        63
    accuracy                           0.57       300
```

### ROC curve & AUC

**AUC = 0.5586** (see `plots/roc_curve.png`).

**Precision formula:** Precision = TP / (TP + FP)
**Recall formula:** Recall = TP / (TP + FN)

**Which metric matters more here?** For this specific task — predicting
smoker status from demographic/lifestyle data — **recall** on the smoker
class is arguably more important than precision, if the intended use case is
something like flagging at-risk employees for a wellness program: missing an
actual smoker (a false negative) means a genuinely at-risk person is not
offered support, which is a more costly miss than mistakenly flagging a
non-smoker for a voluntary program (a false positive).

**What the AUC means here:** an AUC of 0.5586 is only modestly better than
0.50 (pure random guessing), meaning the model has **weak ability to
separate smokers from non-smokers** using the available features (age, BMI,
number of children, region, prior conditions, exercise frequency). **This is
an honest and realistic finding, not a flaw in the pipeline:** in this
dataset, smoking status was generated largely independently of the other
demographic and lifestyle attributes, so there is genuinely little signal
for a classifier to find. This is a useful, realistic lesson: not every
binary target is well-predicted by whatever features happen to be available,
and a weak AUC is itself a legitimate, reportable result rather than
something to "fix" by tuning further.

## 7. Task 5b — Decision-Threshold Sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.2100 | 1.0000 | 0.3471 |
| 0.40 | 0.2133 | 0.9683 | 0.3496 |
| 0.50 | 0.2556 | 0.5397 | 0.3469 |
| 0.60 | 0.2353 | 0.0635 | 0.1000 |
| 0.70 | 0.0000 | 0.0000 | 0.0000 |

**Precision** = TP / (TP + FP). **Recall** = TP / (TP + FN).

**Threshold maximizing F1: 0.40** (F1 = 0.3496), very marginally ahead of
0.30 (F1 = 0.3471) — both are effectively tied given how flat the F1 curve
is between 0.30 and 0.50.

**Which metric matters more, and which direction to move the threshold:**
as argued above, **recall** is the more important metric for a wellness-flagging
use case. Since **lowering** the classification threshold increases recall
(more borderline cases get classified as "smoker"), we would lower the
threshold toward 0.30–0.40 to prioritize catching more true smokers. **The
cost of doing this** is a corresponding drop in precision (more false
positives — non-smokers incorrectly flagged), visible directly in the table:
at threshold 0.30, recall is a near-perfect 1.00, but precision falls to
0.21, meaning roughly 4 in 5 people flagged as "smoker" at that threshold are
actually not. This is a real trade-off: whether it's worth accepting many
false positives to almost never miss a true smoker depends entirely on how
costly a missed case is in the intended application.

## 8. Task 6 — Regularization Experiment

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.2556 | 0.5397 | 0.5586 |
| C=0.01 (strong L2) | 0.2403 | 0.4921 | 0.5681 |

**What `C` controls:** in scikit-learn's `LogisticRegression`, `C` is the
**inverse** of the regularization strength — a smaller `C` means a
*stronger* L2 penalty on the coefficients (more shrinkage toward zero, more
bias, less variance), while a larger `C` allows coefficients to grow larger
to fit the training data more closely. Here, reducing `C` from 1.0 to 0.01
(much stronger regularization) produced a **very slightly worse** precision
and recall, but a **very slightly better** AUC (0.5681 vs 0.5586). Given how
close all these numbers are, and how weak the underlying signal is overall
(see Section 6), this difference is small enough that it may not reflect a
real, reliable improvement — which is exactly what the bootstrap analysis
below investigates directly.

## 9. Task 6b — Bootstrap Confidence Interval for AUC Difference

Using 500 bootstrap resamples of the test set (`np.random.choice` with
replacement, seeded with `np.random.default_rng(42)`):

- **Mean AUC difference (C=1.0 − C=0.01):** **−0.0095**
- **95% CI:** **[−0.0188, 0.0003]**
- **Does the interval exclude zero?** **No** — the interval spans from
  slightly negative to slightly positive, including 0.

**Interpretation:** because the 95% confidence interval for the AUC
difference **includes zero**, we cannot conclude that either model (C=1.0 or
C=0.01) reliably outperforms the other on this dataset — the small observed
difference in Section 8 could plausibly be due to sampling noise in this
particular train/test split rather than a genuine, consistent advantage for
either regularization strength. This reinforces the honest conclusion from
Section 6: with such weak overall signal in the classification task, minor
differences between regularization settings are not statistically
meaningful here.

## 10. Files in this Folder

```
part2/
├── cleaned_data.csv           # carried over from Part 1
├── part2_ml_models.ipynb      # all preprocessing/modeling/evaluation code (outputs cleared)
├── plots/
│   └── roc_curve.png
├── requirements.txt
└── README.md                  # this file
```
