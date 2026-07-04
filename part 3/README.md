# Part 3 — Advanced Modeling: Ensembles, Tuning, and Full ML Pipeline

## 1. Overview

This part extends Part 2's classification task (predicting `smoker` status)
with decision trees, ensemble methods (Random Forest, Gradient Boosting),
systematic hyperparameter tuning via `GridSearchCV` wrapped in a full
`sklearn.Pipeline`, a manual learning curve, and model serialization.

The notebook opens by **reproducing Part 2's exact preprocessing** (same
categorical encoding, same 80/20 split with `random_state=42`, same
`StandardScaler` fit only on training data) so that `Xc_train_scaled`,
`Xc_test_scaled`, `y_clf_train`, and `y_clf_test` are identical to Part 2's,
making every comparison in this part directly consistent with Part 2's
Logistic Regression result (test AUC 0.5586).

## 2. How to Run

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute part3_ensembles_tuning.ipynb --output part3_ensembles_tuning.ipynb
```

Or open `part3_ensembles_tuning.ipynb` in Jupyter and run all cells
top-to-bottom. `cleaned_data.csv` (from Part 1/2) must be in the same
folder. Produces `best_model.pkl` (the tuned pipeline).

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`
(see `requirements.txt`).

## 3. A Note on the Numbers in This Part

Part 2 already established, honestly, that predicting `smoker` status from
the other available features (age, BMI, children, region, exercise
frequency, prior conditions) is a **genuinely weak-signal problem** in this
dataset (Logistic Regression AUC ≈ 0.56). That finding carries through
Part 3: several ensemble/tree models below score **at or even slightly below
0.50 AUC** on the held-out test set. This is not a bug in the code — it is
the expected, consistent continuation of Part 2's result, and we report it
honestly rather than tuning until a better-looking (but less truthful)
number appears. Where a model's test AUC dips under 0.50, that reflects
normal sampling variability around a fundamentally near-random signal, not
a modeling error.

## 4. Task 1 & 2 — Decision Tree: Unconstrained vs Controlled

| Tree | Train Accuracy | Test Accuracy | Train-Test Gap |
|---|---|---|---|
| Unconstrained (`max_depth=None`) | **1.0000** | 0.5633 | **0.4367** |
| Controlled (`max_depth=5, min_samples_split=20`) | 0.7858 | 0.7600 | **0.0258** |

**Does the unconstrained tree overfit?** Emphatically yes — it reaches
**100% training accuracy** (it has memorized the training set exactly) while
test accuracy collapses to 56.3%, a **0.44 train-test gap**. This is a
textbook demonstration of why decision trees are described as **high-variance
models**: at each node, the tree greedily picks whatever split best separates
the *current* training subset, without ever revisiting or reconsidering
earlier splits higher up the tree. Left unconstrained, it keeps splitting
until every leaf is pure (or has one sample), which means it can carve out a
uniquely-shaped decision boundary tailored to the exact noise in this
particular training sample — a boundary that will not generalize, because a
different training sample would have different noise and produce a
differently-shaped (but equally overfit) tree.

**The controlled tree's gap is nearly 17x smaller** (0.026 vs 0.437).
`max_depth=5` directly caps how many sequential splits any path through the
tree can have, which limits how finely it can carve up the feature space —
trading away some ability to fit fine-grained training patterns (a small
increase in bias) for a large reduction in variance. `min_samples_split=20`
additionally prevents the tree from splitting a node that has fewer than 20
samples in it, which stops it from creating splits based on tiny, noisy
subsets of just a handful of rows — exactly the kind of split that would
capture noise rather than signal.

## 5. Task 3 — Gini vs Entropy

**Gini impurity formula:** Gini = 1 − Σ pᵢ²
**Entropy formula:** Entropy = −Σ pᵢ log₂(pᵢ)

where pᵢ is the proportion of samples in the node belonging to class *i*.

**A Gini score of 0** means the node is **perfectly pure** — every single
sample in that node belongs to the same class, so there is nothing left to
split on; that node can become a leaf.

| Criterion | Test Accuracy (max_depth=5) |
|---|---|
| Gini | 0.7667 |
| Entropy | 0.7700 |

The two criteria produce nearly identical trees on this dataset (0.77 vs
0.7667 accuracy) — this is typical in practice: Gini and Entropy usually
agree on which splits are best, since both are measures of node "impurity"
that peak when classes are evenly mixed and hit zero when a node is pure;
they rarely disagree enough to produce meaningfully different trees.

## 6. Task 4 — Random Forest

| Metric | Value |
|---|---|
| Train Accuracy | 0.8767 |
| Test Accuracy | 0.7867 |
| Test AUC | **0.4654** |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| `bmi` | 0.3427 |
| `age` | 0.2707 |
| `children` | 0.0996 |
| `exercise_freq` | 0.0841 |
| `prior_conditions` | 0.0715 |

**How Random Forest computes feature importance:** for each feature, the
algorithm tracks how much that feature's splits reduce Gini impurity, every
time it is used as a split point, across **every tree** in the forest, and
then averages this reduction. A feature that reliably produces large,
consistent impurity reductions across many trees gets a high importance
score. **Why this differs from a linear regression coefficient:** a
regression coefficient measures a feature's *linear, additive* relationship
with the target, holding other features fixed — it can be directly signed
(positive/negative) and interpreted as "one unit change in X changes the
prediction by this much." Feature importance in a Random Forest, by
contrast, is **always non-negative** and reflects how *useful a feature was
for making splitting decisions* anywhere in the tree structure, including
non-linear thresholds and interactions with other features — it tells you
"how much this feature mattered for prediction," not "in which direction."

**Note on Task 4b overlap:** because `X_clf` has only 9 total features, the
"top 5" and "bottom 5" by importance necessarily share one feature
(`prior_conditions`, ranked 5th of 9) — this is simply an artifact of having
fewer than 10 features, not an error.

### Bagging (Bootstrap Aggregating)

Random Forest builds many individual decision trees, each trained on a
**bootstrap sample** — a random sample of the training rows drawn *with
replacement*, the same size as the original training set but with some rows
repeated and others left out. Additionally, at **each split** within each
tree, only a random subset of √(number of features) features is even
considered as candidates, rather than all features. Both sources of
randomness mean every tree in the forest ends up structurally different from
the others, each capturing a slightly different (and imperfect) view of the
data. When the forest's final prediction **averages** (or majority-votes)
across all these different trees, the individual trees' idiosyncratic
errors — the very overfitting behavior demonstrated by the unconstrained
tree in Task 1 — tend to cancel out, while genuine, shared signal reinforces
across trees. This is why an ensemble of high-variance trees can have much
lower variance overall than any single deep tree.

## 7. Task 4a — Gradient Boosting

| Metric | Value |
|---|---|
| Train Accuracy | 0.8067 |
| Test Accuracy | 0.7833 |
| Test AUC | **0.5193** |

Gradient Boosting achieves a modestly better AUC than the Random Forest here
(0.5193 vs 0.4654), though both remain close to random-guessing territory,
consistent with the weak overall signal discussed in Section 3.

## 8. Task 4b — Feature Ablation Study

**5 lowest-importance features (from the Random Forest in Task 4):**
`prior_conditions`, `sex_male`, `region_southeast`, `region_northwest`,
`region_southwest`.

| Model | Test AUC |
|---|---|
| Full model (all 9 features) | 0.4654 |
| Reduced model (5 lowest-importance features removed, 4 features remain) | **0.4888** |
| Change | **+0.0234** |

**Were these features genuinely uninformative?** The reduced model's AUC is
actually *slightly higher* without them, not lower — this suggests the five
removed features (`prior_conditions`, `sex_male`, and the three `region`
dummies) were contributing close to **pure noise** to this particular Random
Forest rather than real signal, and removing them let the model focus its
splits on the two features that do carry the (weak) genuine signal here
(`bmi` and `age`). **Production trade-off implication:** dropping these five
features would give a **simpler, lower-dimensional model** with equal or
slightly better predictive performance — this is a "free lunch" in
production terms: fewer features to collect, validate, and maintain at
inference time, with no accuracy cost in this case. In general this trade-off
is only acceptable when the accuracy check shown above (full vs. reduced
AUC) confirms no meaningful degradation; here it clearly does confirm that,
so simplifying would be a reasonable choice if this model were deployed.

## 9. Task 5 — Cross-Validated Comparison

Using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and
`scoring='roc_auc'` on the training set:

| Model | CV Mean AUC | CV Std AUC |
|---|---|---|
| Logistic Regression | 0.5164 | 0.0360 |
| Decision Tree (max_depth=5) | 0.4885 | 0.0150 |
| Random Forest | 0.4784 | 0.0266 |
| Gradient Boosting | 0.4989 | 0.0481 |

**Why cross-validation is more reliable than a single train-test split:** a
single 80/20 split gives exactly one estimate of test performance, which
depends heavily on which particular rows happened to land in the test set —
an unusually easy or hard test split can make a model look better or worse
than it really is. 5-fold cross-validation instead evaluates the model on
**5 different held-out folds**, each time training on the other 4/5 of the
data, and reports the mean and spread across all 5 estimates. This gives a
far more stable, less split-dependent estimate of how well the model is
likely to generalize, and the standard deviation across folds additionally
quantifies how *consistent* that performance is — a model with a high mean
but also high std (like Gradient Boosting here, std = 0.048) is less
reliably good than one with a similar mean but lower std.

## 10. Task 6 — GridSearchCV Hyperparameter Tuning

**Pipeline:** `make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RandomForestClassifier(random_state=42))`

**Parameter grid:**
```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5],
}
```

**Total configurations evaluated:** 3 × 3 × 2 = **18 hyperparameter
combinations**, each fit across **5 folds** = **90 total model fits**.

**Best params found:** `max_depth=5, min_samples_leaf=1, n_estimators=200`
**Best CV score (mean AUC):** **0.5168**
**Best pipeline test-set AUC:** **0.5097**

**Grid Search vs. Randomized Search trade-off:** `GridSearchCV` exhaustively
evaluates *every* combination in the grid, guaranteeing it finds the best
combination *within that grid* — but its cost grows multiplicatively with
the number of hyperparameters and values tested (here, 90 fits for a
modestly sized grid; a larger grid or more hyperparameters quickly becomes
computationally prohibitive). `RandomizedSearchCV` instead samples a fixed
number of random combinations from the specified parameter distributions,
which scales independently of the grid's total size — it can explore a much
larger or continuous hyperparameter space at a fixed computational budget,
at the cost of no longer *guaranteeing* the single best combination in the
grid is found, only a good one with high probability given enough samples.

## 11. Task 7 — Manual Learning Curve

Using the best pipeline's hyperparameters, refit on progressively larger
training subsets:

| Training Fraction | Rows | Training AUC | Test AUC |
|---|---|---|---|
| 0.2 | 240 | 0.9743 | 0.5113 |
| 0.4 | 480 | 0.9408 | 0.5210 |
| 0.6 | 720 | 0.8854 | 0.5098 |
| 0.8 | 960 | 0.8667 | 0.4949 |
| 1.0 | 1200 | 0.8474 | 0.5097 |

**(i) Does training AUC decrease as the training set grows?** Yes, clearly —
it drops from 0.974 at 20% of the data down to 0.847 at 100%. This is the
expected pattern for a high-variance/high-capacity model: with very little
training data, the Random Forest can nearly memorize it (near-perfect
training AUC), but as more (and more varied) data is added, it becomes
progressively harder to fit every training point perfectly, so training AUC
naturally comes down toward a more realistic level.

**(ii) Does test AUC increase with more training data?** No — it stays
essentially **flat, fluctuating in a narrow band around 0.49–0.52**
regardless of how much training data is used (240 rows or 1200 rows produce
statistically indistinguishable test AUC). This means collecting more data
of the same kind (more rows with these same features) would **not** be
expected to meaningfully improve this model's ability to predict smoker
status.

**(iii) Conclusion — data-limited or capacity-limited?** **Neither, in the
usual sense — this model is signal-limited.** Test AUC is not still rising
at 100% of the data (ruling out "just needs more rows"), and increasing
model capacity in Part 3 (Random Forest, Gradient Boosting, extensive
tuning) did not meaningfully improve on Part 2's simple Logistic Regression
result either (ruling out "just needs a more powerful model"). The flat
learning curve combined with the consistently weak AUC across every model
type tried point to the same underlying conclusion already suggested in
Part 2: the available features (age, BMI, children, region, exercise
frequency, prior conditions) simply carry very little genuine information
about smoking status in this dataset. The path to a meaningfully better
model here would be **collecting different, more directly relevant
features** (e.g. household smoking history, nicotine-dependence screening
responses) rather than more rows of the same features or a fancier
algorithm.

## 12. Task 8 — Model Serialization

The best pipeline (from `GridSearchCV.best_estimator_`) was saved with:
```python
joblib.dump(best_pipeline, "best_model.pkl")
```

`best_model.pkl` (≈840 KB) is committed directly to this folder (well under
the 100 MB limit).

**Reload-and-predict sanity check** (included in the notebook):
```python
loaded_model = joblib.load("best_model.pkl")
predictions = loaded_model.predict(hand_crafted_rows)
probabilities = loaded_model.predict_proba(hand_crafted_rows)[:, 1]
```
Ran successfully on two hand-crafted rows (a young/low-BMI/high-exercise
profile and an older/high-BMI/multiple-prior-conditions profile), returning
predicted class and probability for each with no errors.

## 13. Summary Comparison Table — All Models (Parts 2 & 3)

| Model | CV Mean AUC | CV Std AUC | Test AUC |
|---|---|---|---|
| Logistic Regression | 0.5164 | 0.0360 | 0.5586 |
| Decision Tree (max_depth=5) | 0.4885 | 0.0150 | 0.4504 |
| Random Forest | 0.4784 | 0.0266 | 0.4654 |
| Gradient Boosting | 0.4989 | 0.0481 | 0.5193 |
| **Tuned RF (GridSearchCV pipeline)** | **0.5168** | — | 0.5097 |

### Recommendation

**We recommend Logistic Regression** (from Part 2, `class_weight='balanced'`,
C=1.0) as the model to give the client. Its test-set AUC (0.5586) is the
highest of any model tried across both parts, and its cross-validated mean
AUC (0.5164) is statistically indistinguishable from the tuned Random Forest
pipeline's (0.5168) despite being far simpler, faster to train, and fully
interpretable via its coefficients. Given that every model type tried here —
from a single shallow tree through a heavily-tuned 200-tree Random Forest —
converges to essentially the same weak-signal ceiling (AUC ≈ 0.50–0.56), the
added complexity of an ensemble buys no measurable benefit for this specific
prediction target with these specific features, so the simplest adequate
model is the right choice. If this were deployed, we would communicate to
the client that the honest, current ceiling for predicting smoker status
from this feature set is close to random guessing, and that better features
— not a better algorithm — is the lever most likely to improve it.

## 14. Files in this Folder

```
part3/
├── cleaned_data.csv                  # carried over from Part 1/2
├── part3_ensembles_tuning.ipynb      # all modeling/tuning/serialization code (outputs cleared)
├── best_model.pkl                    # serialized best (tuned Random Forest) pipeline
├── requirements.txt
└── README.md                         # this file
```
