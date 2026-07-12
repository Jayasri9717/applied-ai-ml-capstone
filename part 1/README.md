# Part 1 — Data Acquisition, Cleaning, and Exploratory Analysis

## 1. Dataset Description

This project uses a **synthetic "Employee Health & Insurance Cost" dataset**
(`raw_data.csv`), generated programmatically by `generate_raw_data.py`
(1,540 rows × 9 columns before cleaning, 1,500 rows after duplicate removal).

**Why a synthetic dataset?** The brief allows any public structured CSV with
≥500 rows and ≥5 columns. Well-known public alternatives (e.g. the classic
"Medical Cost Personal Dataset" on Kaggle) are widely used for this exact kind
of assignment but are *too clean* — they contain zero missing values and no
duplicates, which would leave several required tasks (null-percentage
reporting, median imputation, duplicate-removal impact) with nothing
meaningful to demonstrate. Instead, `generate_raw_data.py` builds a dataset
from a believable underlying cost formula (age, BMI, smoker status, prior
conditions, number of children, and exercise frequency genuinely drive a
target `annual_charges` variable, plus a small share of catastrophic claims
and random noise) and **then deliberately injects realistic messiness**:
missing values at different rates per column, 40 duplicate rows, a column
stored with the wrong dtype, and skewed/outlier-heavy numeric columns. This
keeps every analysis genuine and reproducible without external downloads —
running `python generate_raw_data.py` regenerates `raw_data.csv` from
scratch with a fixed random seed (42), so results are fully deterministic
and verifiable by the grader.

### Columns

| Column | Type | Description |
|---|---|---|
| `age` | numeric | Employee age (18–64) |
| `sex` | categorical | male / female |
| `bmi` | numeric | Body mass index |
| `children` | numeric | Number of dependents |
| `smoker` | categorical | yes / no |
| `region` | categorical | northeast / northwest / southeast / southwest |
| `exercise_freq` | categorical | none / low / moderate / high |
| `prior_conditions` | numeric | Count of pre-existing health conditions |
| `annual_charges` | numeric (target) | Annual medical insurance cost billed ($) |

## 2. How to Run

```bash
pip install -r requirements.txt
python generate_raw_data.py        # produces raw_data.csv (only needed once; already committed)
jupyter nbconvert --to notebook --execute part1_eda_cleaning.ipynb --output part1_eda_cleaning.ipynb
```

Or simply open `part1_eda_cleaning.ipynb` in Jupyter and run all cells
top-to-bottom. This produces:
- `cleaned_data.csv` (used by Parts 2 and 3)
- `plots/*.png` (6 visualization files, listed below)

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn` (see
`requirements.txt`).

## 3. Cleaning Steps & Findings

### Null value analysis (Task 2)

| Column | Null % |
|---|---|
| bmi | 6.10% |
| children | 3.96% |
| exercise_freq | 25.06% |
| prior_conditions | 9.35% |
| (all others) | 0.00% |

Only **`exercise_freq` exceeds the 20% threshold** and is reported but left
un-imputed at this stage (median-fill is undefined for a categorical column;
it's a candidate for mode-fill or a dedicated "missing" category if used in
Part 2). The three numeric columns below the threshold (`bmi`, `children`,
`prior_conditions`) were filled with their **column median**.

**Why median, not mean?** Both `prior_conditions` (skew = 1.50) and
`children` (skew = 1.05) are right-skewed count variables — most employees
have zero or one, but a smaller number have several, which drags the mean
above what's "typical." The median is robust to this asymmetry, so it
better represents a typical row and avoids systematically over-imputing the
real value for most employees.

### Duplicate detection (Task 3)

**40 duplicate rows** were found and removed (1540 → 1500 rows). Comparing
null percentages before and after: every column's null % was numerically
unchanged or shifted by less than 0.1 percentage points (e.g. `exercise_freq`
moved from 25.06% to 25.00%) — the duplicate rows were a representative
random sample of the data, not a block carrying a disproportionate share of
missing values, so removing them did not meaningfully bias the missingness
pattern.

### Data type correction (Task 4)

`annual_charges` loaded as `object`/string dtype because ~35% of rows had a
stray `"$"` prefix (e.g. `"$8132.21"`), a common real-world data-entry
inconsistency. It was cleaned with `.str.replace("$", "")` and converted with
`pd.to_numeric(errors='coerce')`, producing **zero new nulls** — confirming
every value was numeric once the symbol was stripped.

The four repetitive string columns (`sex`, `smoker`, `region`,
`exercise_freq`) were converted to `category` dtype. Memory usage dropped
from **450,049 bytes to 66,790 bytes — an 85.2% reduction** — because
`category` dtype stores each unique label once and references it by integer
code rather than repeating the full string in every row.

## 4. Skewness (Task 5)

| Column | Skewness |
|---|---|
| `annual_charges` | **2.376** (highest absolute) |
| `prior_conditions` | 1.502 |
| `children` | 1.053 |
| `bmi` | 0.040 |
| `age` | -0.028 |

**`annual_charges`** has the highest absolute skewness at **2.38**, indicating
strong **positive (right) skew**: most employees cluster at lower cost values,
with a long tail of expensive outliers (smokers and catastrophic-claim cases)
pulling the distribution's mean well above its median. **Consequence for
imputation:** if nulls in this column were filled using the mean, the filled
values would systematically overstate typical cost, because the mean
($11,007) is more than 60% higher than the median ($6,821) here — exactly
why Task 8a chose the median for this column (see below).

## 5. Outlier Detection — IQR (Task 6)

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Outliers (n) | Outliers (%) |
|---|---|---|---|---|---|---|---|
| `annual_charges` | 4,629.11 | 12,475.43 | 7,846.32 | −7,140.37 | 24,244.91 | **173** | 11.53% |
| `bmi` | 25.225 | 32.91 | 7.685 | 13.70 | 44.44 | **8** | 0.53% |

**`annual_charges`:** 173 rows (11.5%) sit above the upper bound. These are
not data errors — they correspond mostly to smokers (whose costs are
structurally ~3.5x higher, see box plot below) plus a small number of
genuine catastrophic-claim cases built into the data-generating process.
**Handling strategy for Part 2:** these outliers will be **retained, not
dropped** — they represent real, predictable variation (driven by `smoker`
status) rather than noise, and a tree-based or log-transformed regression
model in Part 2 can learn this pattern rather than being thrown off by it.
Dropping them would bias the model toward underestimating costs for a real
and economically important subgroup.

**`bmi`:** Only 8 rows (0.5%) are outliers — a small, plausible share of
employees with unusually low or high BMI. These will also be **retained**,
as removing under 1% of rows of biologically realistic values would add risk
of bias for little benefit.

## 6. Visualizations

All plots are saved in `plots/` as `.png` files.

1. **`01_line_plot_charges.png`** — `annual_charges` sorted ascending by row
   index. Shows a smooth, gradually accelerating curve consistent with a
   right-skewed distribution: most rows sit on a flat low-cost plateau, and
   the curve sharply bends upward only in the final ~15% of sorted rows.

2. **`02_bar_mean_charges_by_region.png`** — Mean `annual_charges` by
   `region`. Northeast has the highest mean (~$11,563) and southwest the
   lowest (~$10,164); see Task 8c discussion below for whether this
   difference is meaningful.

3. **`03_histogram_most_skewed.png`** — Histogram (bins=20) of
   `annual_charges`, the most skewed column. The distribution is clearly
   **bimodal and right-skewed**: a tall peak around $3,000–8,000
   (non-smokers), a smaller secondary peak around $20,000–27,000 (smokers),
   and a long thin tail extending out past $80,000 from catastrophic-claim
   cases.

4. **`04_scatter_bmi_vs_charges.png`** — `bmi` vs `annual_charges`, colored
   by `smoker`. **Interpretation:** there are two visually distinct bands —
   smokers (orange) cluster at higher charges across nearly all BMI values,
   while non-smokers (blue) form a lower band that rises modestly with BMI
   once it crosses ~30 (the BMI-cost surcharge built into the data). The
   relationship between BMI and charges alone is weak-to-moderate and only
   becomes visually clear once smoker status is used to separate the two
   sub-populations — a textbook case where a third variable changes the
   apparent strength of a bivariate relationship.

5. **`05_boxplot_charges_by_smoker.png`** — `annual_charges` split by
   `smoker`. **Interpretation:** the median charge for smokers (~$23,000) is
   roughly **3.5x higher** than for non-smokers (~$6,500), with almost no
   overlap between the two boxes' interquartile ranges. Non-smokers show a
   long tail of upper outliers (catastrophic claims unrelated to smoking);
   smokers show a tighter, narrower spread overall but with their own high
   tail. This is by far the strongest single categorical driver of cost in
   the dataset.

6. **`06_correlation_heatmap_pearson.png`** — Pearson correlation matrix of
   all numeric columns. **Highest absolute correlation pair: `prior_conditions`
   and `annual_charges` (r = 0.114)** — a fairly weak linear correlation.
   **Causal discussion:** this is plausible as a partial causal link (more
   pre-existing conditions → more treatment → higher billed charges), but it
   is also very likely **confounded by `age`**, even though `age` itself
   shows almost no direct linear correlation with charges here (r = 0.038):
   older employees tend to accumulate more prior conditions, and `age` was
   also built into the cost formula independently. A more important caveat
   raised by this analysis: the single strongest cost driver in the entire
   dataset — `smoker` status — is **categorical** and therefore invisible to
   a numeric-only Pearson correlation matrix entirely, illustrating that a
   correlation heat map computed only on numeric columns can dramatically
   understate which features actually matter most for the target.

## 7. Task 8a — Mean vs Median Imputation Comparison

| Column | Mean | Median | Skew | Chosen statistic |
|---|---|---|---|---|
| `annual_charges` | 11,006.90 | 6,820.67 | +2.38 | **Median** |
| `prior_conditions` | 0.531 | 0.000 | +1.50 | **Median** |

Both columns are positively skewed, meaning a small number of large values
pull the mean upward, well above what is typical for most rows (most
strikingly for `annual_charges`: the mean is **61% higher** than the median).
Imputing with the mean would systematically inflate the assumed value for
any missing entry; the **median was chosen for both columns** as the more
representative central tendency. After imputation,
`df[col].isnull().sum()` confirms **0 remaining nulls** in both columns.

## 8. Task 8b — Spearman vs Pearson Correlation

| Pair | Pearson | Spearman | |Spearman − Pearson| |
|---|---|---|---|
| `prior_conditions` <-> `annual_charges` | 0.114 | 0.283 | **0.169** |
| `annual_charges` <-> `bmi` | 0.032 | 0.131 | **0.099** |
| `annual_charges` <-> `age` | 0.038 | 0.133 | **0.096** |

In all three top pairs, **|Spearman| > |Pearson|**, indicating the underlying
relationships are **monotonic but non-linear** rather than approximately
linear. This makes sense given how the data was generated: `prior_conditions`
contributes to cost as a step-wise multiplier (each additional condition adds
a fixed cost increment) rather than a smooth linear function, and `bmi`'s
effect only kicks in above a threshold (BMI > 30), producing a kinked,
non-linear-but-monotonic relationship that Pearson under-detects and Spearman
captures better. **For Part 2 feature selection, Spearman correlation will be
the primary guide**, since it more accurately reflects the true (non-linear)
strength of these relationships and will better flag which numeric features
are worth including, especially for tree-based models that can exploit
non-linear and threshold effects directly.

## 9. Task 8c — Grouped Aggregation

`groupby('region')['annual_charges'].agg(['mean', 'std', 'count'])`:

| Region | Mean | Std | Count |
|---|---|---|---|
| northeast | 11,562.97 | 10,285.69 | 416 |
| northwest | 11,293.46 | 11,217.35 | 340 |
| southeast | 10,911.84 | 11,073.19 | 399 |
| southwest | 10,163.94 | 9,170.53 | 345 |

- **Highest mean group:** northeast ($11,563)
- **Highest standard deviation group:** northwest ($11,217)
- **Within-group variance concern:** yes — every region's standard deviation
  is close to or larger than its own mean. This high within-group spread
  means `region` alone explains very little about an individual employee's
  cost; two employees in the same region can have wildly different charges
  depending on `smoker` status, `prior_conditions`, etc. `region` is not a
  reliable single predictor on its own.
- **Mean ratio (highest / lowest):** 11,562.97 / 10,163.94 = **1.14**. A
  ratio this close to 1.0 suggests `region` carries only **weak predictive
  signal** by itself — the roughly 14% spread between the highest- and
  lowest-mean regions is small compared to the within-region standard
  deviations (which are 80–100% of each group's own mean), confirming that
  region is a much weaker driver of cost than `smoker` status or
  `prior_conditions`.

## 10. Files in this Folder

```
part1/
├── generate_raw_data.py          # generates raw_data.csv (synthetic, seeded, reproducible)
├── raw_data.csv                  # raw input data (1540 rows, intentionally messy)
├── part1_eda_cleaning.ipynb      # all cleaning/EDA/visualization code (outputs cleared)
├── cleaned_data.csv              # output of the notebook; used in Parts 2 & 3
├── plots/                        # 6 saved PNG visualizations
├── requirements.txt
└── README.md                     # this file
```
