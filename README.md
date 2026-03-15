# ⚗️ SynthForge — AI-Powered Synthetic Data Generator

**Generative AI Course Project | The Builder Track**  
**Deployed App:** https://parkeralanis-synthforge-gdsbt9nrr9ebupk8thgw7q.streamlit.app

---

## 📋 Abstract

SynthForge addresses the challenge of data scarcity in machine learning by using Anthropic's Claude API to generate realistic, statistically plausible tabular datasets from plain English descriptions. Unlike traditional synthetic data tools that require statisticians to manually define probability distributions and correlation matrices, SynthForge allows anyone to describe a schema in plain English and instantly download a CSV ready for ML training pipelines.

---

## 🎯 Problem Statement

Training machine learning models requires large, labeled datasets. Three core problems make this difficult:

1. **Data Scarcity** — Real datasets are expensive and time-consuming to collect
2. **Privacy Risks** — Real datasets contain PII (names, emails, financials) creating legal and ethical liability
3. **Manual Simulation is Hard** — Traditional tools require manually defined distributions — a slow, error-prone process

**Why Generative AI?** Claude understands context and correlations from plain English. Saying *"income should correlate with age"* is enough — no manual distribution modeling required.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web Framework | Streamlit |
| AI API | Anthropic Claude (`claude-sonnet-4-20250514`) |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Deployment | Streamlit Cloud |

---

## 🚀 How to Run Locally

### Step 1 — Get a Free Anthropic API Key
1. Go to **https://console.anthropic.com**
2. Sign up and navigate to **API Keys → Create Key**
3. Copy the key (starts with `sk-ant-...`)

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the App
```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`, paste your API key in the sidebar, and generate your first dataset.

---

## ⚙️ How It Works

### Architecture Overview

```
User defines schema → Prompt constructed → Claude API called (batched) → JSON parsed → DataFrame → CSV download
```

### Step-by-step

**1. Schema Builder**  
The user defines columns with a name, data type (String, Integer, Float, Boolean, Date, Email, Phone, Category), and a plain English description of realistic values.

**2. Prompt Construction**  
The app builds a two-part prompt:
- **System prompt** — enforces rules: return only raw JSON, think step by step (Chain-of-Thought), vary decimal places naturally
- **User prompt** — includes domain context, full schema, extra instructions, and a dynamically generated few-shot example row built from the user's column types

**3. Batch Generation**  
Large requests are split into batches of 35 rows per API call to stay within Claude's token limits. Each batch prompt includes a diversity hint (*"do not repeat values from previous batches"*). A progress bar updates in real time.

**4. Post-Processing**  
After all batches complete:
- DataFrame trimmed to exactly the requested row count
- Duplicate rows removed automatically
- Results displayed in-app with statistics and charts

**5. Export**  
One-click CSV download for use in downstream ML pipelines.

---

## 🔬 Prompt Engineering Evolution

The prompts evolved through 6 versions from naive to production quality:

| Version | Change | Result |
|---|---|---|
| v1.0 | `"Generate some fake customer data."` | ❌ Returned prose paragraph — no structure |
| v1.2 | Added JSON format + column descriptions | ⚠️ Better but JSON wrapped in markdown fences — broke parser |
| v2.0 | Separated system prompt from user prompt | ✅ Clean JSON, reliable output |
| v2.1 | Added few-shot example row | ✅ Column names and types match reliably |
| v2.2 | Added batch generation + diversity hints | ✅ Supports up to 2,000 rows without token errors |
| v2.3 | Added Chain-of-Thought + Rule 6 (decimal variety) + row trimming + deduplication | ✅ Natural decimals, exact row count, no duplicates |

**Key lesson:** Separating system rules from user task, adding few-shot examples, and explicit Chain-of-Thought reasoning each produced measurable quality improvements.

---

## 📊 Evaluation Methodology

### Tier 1 — Automated Metrics (runs on every generation)
- **Null Rate** — percentage of missing values (target: 0%)
- **Duplicate Rate** — percentage of repeated rows (target: 0%)
- **Type Accuracy** — whether column types match the schema

### Tier 2 — LLM-as-Judge (Claude evaluates the output)
A separate Claude prompt scores the dataset on 5 criteria from 1–5:

| Criterion | Score | Notes |
|---|---|---|
| Realism | 3/5 | Income decimals were mechanical (.00/.25/.50/.75) — addressed in v2.3 |
| Diversity | 4/5 | 125 unique cities, ages 18–72 |
| Type Correctness | 5/5 | All columns perfectly typed |
| Internal Consistency | 4/5 | Age-income tiers and purchase rates logically ordered |
| Hallucination Check | 4/5 | All cities real US locations, no impossible values |
| **Overall** | **4.0/5** | |

> Note: ROUGE/BLEU metrics apply to text generation tasks. For tabular data, null rate, duplicate rate, type accuracy, and LLM-as-judge scoring are the appropriate evaluation methods.

---

## 🔴 Critical Evaluation — Failure Modes Discovered

Three failure modes were identified through deliberate adversarial testing:

### Test 1 — Contradictory Instructions (⚠️ Medium)
**Input:** *"age must be exactly 25 but also highly varied across all rows"*  
**Result:** Claude silently prioritized the hard constraint — all 25 rows had `age = 25`. The variation instruction was completely ignored with no warning.  
**Lesson:** LLMs resolve contradictions silently. Production systems should validate instructions for logical conflicts before sending to the API.

### Test 2 — Missing Column Description (✅ Low)
**Input:** Age column description left blank, name kept.  
**Result:** Claude correctly inferred realistic age values (22–56) using world knowledge. Data quality maintained.  
**Lesson:** Claude is resilient for common columns, but specialized domain columns (e.g. `risk_score`, `churn_probability`) would silently produce arbitrary values without descriptions.

### Test 3 — Blank Column Name + Description (🔴 High — Fixed)
**Input:** Age column name AND description both left empty.  
**Result:** Column was silently dropped from output — dataset generated with 3 columns instead of 4. No error was raised.  
**Fix Applied (v2.3):** Input validation now checks for blank column names before sending to the API and raises a clear error message. Silent data loss is no longer possible.

---

## ⚖️ Ethical Considerations

### Bias Risk
LLMs inherit societal biases from training data. Income correlated with demographics may reflect historical inequalities rather than desired neutral distributions. Users should use the Extra Instructions field to specify explicit distribution constraints when bias is a concern.

### Data Privacy
SynthForge generates all data via API — no real user data is ever collected or stored. However, users must never paste real PII into the Extra Instructions field as this would transmit it to the Anthropic API.

### Environmental Impact
Each batch API call consumes computational energy. For this project's scale (hundreds of rows), the impact is minimal. At production scale, mathematical sampling methods (e.g. Gaussian mixtures) should be preferred over LLM-based generation.

### Responsible Use Guidelines
- Always document that a dataset is synthetic when sharing it
- Never use synthetic data as the sole validation source for high-stakes systems
- Review generated data for demographic bias before training classifiers on it

---

## 📁 Project Structure

```
synthforge/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🔗 Links

- **Live App:** https://parkeralanis-synthforge-gdsbt9nrr9ebupk8thgw7q.streamlit.app
- **API Documentation:** https://docs.anthropic.com
- **Streamlit Cloud:** https://share.streamlit.io
