"""
generate_raw_data.py
---------------------
Generates a synthetic but realistic "Employee Health & Insurance Cost" dataset
(raw_data.csv) for use in Part 1 (EDA/cleaning) and Part 2 (ML) of the capstone.

This script is run ONCE to produce raw_data.csv, which is then committed to the
repo so the grader does not need internet access to reproduce the raw input.
The dataset is intentionally "messy" (missing values, duplicates, a numeric
column stored as text, outliers, skewed columns) to give every cleaning/EDA
task in the Part 1 spec something real to work on.

Run:
    python generate_raw_data.py
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1500


def main():
    # ---- Core demographic / lifestyle features -----------------------------
    age = RNG.integers(18, 65, size=N)

    sex = RNG.choice(["male", "female"], size=N)

    region = RNG.choice(
        ["northeast", "northwest", "southeast", "southwest"],
        size=N,
        p=[0.27, 0.23, 0.27, 0.23],
    )

    smoker = RNG.choice(["yes", "no"], size=N, p=[0.21, 0.79])

    children = RNG.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.40, 0.25, 0.18, 0.10, 0.05, 0.02])

    exercise_freq = RNG.choice(
        ["none", "low", "moderate", "high"], size=N, p=[0.30, 0.30, 0.25, 0.15]
    )

    # BMI: roughly normal, slightly right-skewed (a realistic population trait)
    bmi = RNG.normal(loc=28, scale=6, size=N)
    bmi = bmi + RNG.exponential(scale=2.0, size=N) * 0.6  # adds right skew
    bmi = np.clip(bmi, 14, 55)

    # prior_conditions: count of pre-existing conditions, right-skewed (Poisson)
    prior_conditions = RNG.poisson(lam=0.6, size=N)

    # ---- Target variable: annual_charges -----------------------------------
    # Built from a believable underlying formula + noise, so correlations in
    # Part 1/2 are genuine and explainable, not arbitrary.
    base = 1500
    age_effect = age * 45
    bmi_effect = np.where(bmi > 30, (bmi - 30) * 220, 0)
    smoker_effect = np.where(smoker == "yes", 18000, 0)
    children_effect = children * 450
    conditions_effect = prior_conditions * 1800
    exercise_map = {"none": 1400, "low": 700, "moderate": 0, "high": -600}
    exercise_effect = np.array([exercise_map[e] for e in exercise_freq])

    noise = RNG.normal(0, 1800, size=N)
    # A handful of genuine high-cost outliers (catastrophic claims) - realistic
    catastrophic = RNG.choice([0, 1], size=N, p=[0.97, 0.03]) * RNG.uniform(25000, 60000, size=N)

    annual_charges = (
        base
        + age_effect
        + bmi_effect
        + smoker_effect
        + children_effect
        + conditions_effect
        + exercise_effect
        + noise
        + catastrophic
    )
    annual_charges = np.clip(annual_charges, 500, None)

    df = pd.DataFrame(
        {
            "age": age,
            "sex": sex,
            "bmi": np.round(bmi, 2),
            "children": children,
            "smoker": smoker,
            "region": region,
            "exercise_freq": exercise_freq,
            "prior_conditions": prior_conditions,
            "annual_charges": np.round(annual_charges, 2),
        }
    )

    # ---- Introduce realistic messiness --------------------------------------

    # 1. Missing values at varying rates per column (some <20%, one >20% on purpose)
    def inject_nulls(col, frac):
        idx = RNG.choice(df.index, size=int(frac * N), replace=False)
        df.loc[idx, col] = np.nan

    inject_nulls("bmi", 0.06)                # 6%  -> below threshold, median-fill
    inject_nulls("prior_conditions", 0.09)   # 9%  -> below threshold, median-fill
    inject_nulls("children", 0.04)           # 4%  -> below threshold, median-fill
    inject_nulls("exercise_freq", 0.25)      # 25% -> ABOVE 20% threshold, report only

    # 2. annual_charges stored partly as messy text (forces dtype correction task)
    #    e.g. "13452.6" stays numeric-looking but column becomes dtype "object"
    #    because of stray formatting in a subset of rows.
    charges_str = df["annual_charges"].astype(str)
    messy_idx = RNG.choice(df.index, size=int(0.35 * N), replace=False)
    charges_str.loc[messy_idx] = charges_str.loc[messy_idx].apply(lambda x: f"${x}")
    df["annual_charges"] = charges_str  # now dtype object, needs pd.to_numeric cleanup

    # 3. Duplicate rows (simulate accidental double-entry from client system)
    dup_rows = df.sample(n=40, random_state=1)
    df = pd.concat([df, dup_rows], ignore_index=True)

    # 4. Shuffle row order so duplicates aren't trivially adjacent
    df = df.sample(frac=1, random_state=7).reset_index(drop=True)

    df.to_csv("raw_data.csv", index=False)
    print(f"raw_data.csv written: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.dtypes)


if __name__ == "__main__":
    main()
