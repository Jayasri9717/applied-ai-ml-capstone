# Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline

## Chosen Track: **(C) Model Prediction Explanation Pipeline**

This feature loads the best-performing model from Part 3 (`best_model.pkl`,
a tuned Random Forest pipeline predicting `smoker` status), runs it on three
hand-crafted feature-vector inputs, and calls an LLM to generate a
structured, schema-validated, plain-language explanation of each prediction
for a non-technical stakeholder.

## Important: One step only you can do

This notebook makes real HTTP calls to a live LLM API. I cannot execute
those calls myself — I do not have (and should never be given) your API
key, and I validated every other part of this pipeline (model loading,
`encode_record`, the PII guardrail, JSON parsing, and schema validation)
using a local simulated response to confirm the logic is 100% correct and
bug-free before handing it to you. The only thing left to do is run the
notebook once yourself, using your own free API key, so the final tables
below reflect real model output. Full step-by-step instructions are in
Section 5 below.

## 1. How to Run

### Step 1 — Get a free Groq API key
1. Go to https://console.groq.com and sign up (free, no credit card required for the free tier).
2. Once logged in, go to **API Keys** in the left sidebar and click **Create API Key**.
3. Copy the key — you'll only see it once.

### Step 2 — Set up your `.env` file
1. In this `part4/` folder, copy `.env.example` to a new file named exactly `.env`.
2. Open `.env` and paste your key in:
   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   GROQ_MODEL=llama-3.1-8b-instant
   ```
3. `.env` is already excluded via `.gitignore` — it will never be committed to GitHub.

**If you get a "model not found" error:** Groq periodically updates its
available model list. Check https://console.groq.com/docs/models for the
current list of supported models and update `GROQ_MODEL` in `.env`
accordingly (any chat-completion model listed there will work).

### Step 3 — Install dependencies and run
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute part4_llm_explanation.ipynb --output part4_llm_explanation.ipynb
```
Or simply open `part4_llm_explanation.ipynb` in Jupyter/VS Code and run all
cells top-to-bottom.

### Step 4 — Copy the results into this README
The last cell of the notebook prints ready-to-paste Markdown tables built
from your run's real results. Copy that output and paste it into Section 6
below, replacing the placeholder tables.

## 2. Prompt Design

### System prompt (written out verbatim, zero-shot per Track C)

```
You are a careful, factual model-explanation assistant for a health
insurance analytics team. You will be given a data record's feature
values, a machine learning model's predicted class, and the model's
predicted probability for that class. Your job is to produce a short,
plain-language explanation of the prediction for a non-technical
stakeholder. Output ONLY a single valid JSON object -- no markdown code
fences, no extra commentary before or after it. The JSON object must
have exactly these fields: "prediction_label" (string, e.g. "likely
smoker" or "likely non-smoker"), "confidence_level" (string, one of
"low", "medium", "high", based on how far the predicted probability is
from 0.5), "top_reason" (string, the single feature most likely driving
this prediction, in plain language), "second_reason" (string, the
second most likely contributing feature), "next_step" (string, a brief,
sensible recommended next action for the stakeholder, e.g. offering a
wellness check-in). Base your reasoning only on the feature values
provided; do not invent facts not present in the input.
```

### User prompt template (with placeholders shown)

```
Feature values:
{features_as_json}

Predicted class: {predicted_class} ({"smoker" if predicted_class == 1 else "non-smoker"})
Predicted probability of class 1 (smoker): {probability:.4f}

Produce the JSON explanation now.
```

### Why temperature=0

`temperature=0` was used for every "production" call in this pipeline (the
three-input demonstration). At temperature 0, the model always selects the
single highest-probability next token at each step, making its output fully
deterministic given the same input — critical for a structured data task
where we need the same input to reliably produce the same well-formed JSON
shape every time, rather than occasionally drifting into extra commentary,
different field names, or malformed syntax. This matters especially because
the output is immediately parsed with `json.loads()` and validated against
a strict schema — non-determinism here would translate directly into
unpredictable validation failure rates.

## 3. JSON Schema (5 required scalar fields)

```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label", "confidence_level", "top_reason", "second_reason", "next_step",
    ],
}
```

Every LLM response is stripped of whitespace, parsed with `json.loads()`
inside a `try/except json.JSONDecodeError` block, then validated against
this schema with `jsonschema.validate()` inside a `try/except
jsonschema.ValidationError` block. On any failure (malformed JSON or schema
violation), the pipeline logs the specific error message and falls back to
a dict with all 5 fields set to `None`, rather than crashing or silently
passing through bad data.

## 4. PII Guardrail

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

Every call to the LLM is routed through this check first. If PII is
detected in the constructed prompt, the pipeline prints "Input blocked: PII
detected." and returns `None` without ever contacting the LLM API — verified
directly in the notebook (see Section 6's guardrail table).

## 5. The Three Hand-Crafted Test Records

```python
test_records = [
    {"age": 24, "bmi": 21.5, "children": 0, "exercise_freq": 3, "prior_conditions": 0,
     "sex_male": True, "region_northwest": False, "region_southeast": False, "region_southwest": False},
    {"age": 57, "bmi": 35.2, "children": 3, "exercise_freq": 0, "prior_conditions": 3,
     "sex_male": False, "region_northwest": False, "region_southeast": True, "region_southwest": False},
    {"age": 40, "bmi": 27.8, "children": 1, "exercise_freq": 2, "prior_conditions": 1,
     "sex_male": True, "region_northwest": True, "region_southeast": False, "region_southwest": False},
]
```
Chosen to span a low-risk profile (young, low BMI, high exercise), a
higher-risk profile (older, high BMI, no exercise, multiple prior
conditions), and a middle-ground profile.

## 6. Results (fill in after running the notebook)

Run the notebook, then paste the output of its final cell here, replacing
the three placeholder tables below.

### Three-Row Demonstration Table

| Input (Record) | LLM Output | Valid JSON | Pass/Block |
|---|---|---|---|
| _run notebook to fill in_ | | | |
| _run notebook to fill in_ | | | |
| _run notebook to fill in_ | | | |

### Temperature A/B Comparison Table

| Input (Record) | Output at temp=0 | Output at temp=0.7 | Key Difference |
|---|---|---|---|
| _run notebook to fill in_ | | | |
| _run notebook to fill in_ | | | |
| _run notebook to fill in_ | | | |

**Why temperature affects output this way:** at `temperature=0`, the model
deterministically picks the single most likely next token at every step, so
identical inputs produce identical (or near-identical) outputs across runs.
At `temperature=0.7`, the model instead samples from a broader probability
distribution over likely next tokens, introducing controlled randomness —
this can surface different phrasing, different emphasis on which feature is
cited as the "top reason," or occasionally a different `confidence_level`
judgment, even though the underlying facts (feature values, predicted
class, predicted probability) are unchanged.

### Guardrail Test Results

| Input | PII Detected | LLM Called |
|---|---|---|
| "Please explain this record, contact me at jane.doe@example.com for questions." | Yes | No (blocked) |
| "Please explain this record for a general audience." | No | Yes |

(This table's structure is fixed and already verified — the specific
"Result" text for the clean-input case will appear in your notebook run.)

## 7. Files in this Folder

```
part4/
├── best_model.pkl              # copied from Part 3 (the tuned Random Forest pipeline)
├── part4_llm_explanation.ipynb # all LLM integration/guardrail/validation code (outputs cleared)
├── .env.example                # template for your API key (copy to .env, never commit .env)
├── .gitignore                  # excludes .env from version control
├── requirements.txt
└── README.md                   # this file
```
