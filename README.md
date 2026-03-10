# ⚗️ SynthForge — Synthetic Data Generator

A Generative AI application that produces realistic synthetic tabular datasets
using Anthropic's Claude API — built for a Gen AI course project.

---

## 🚀 Quick Setup (5 minutes)

### Step 1: Get a Free Anthropic API Key
1. Go to **https://console.anthropic.com**
2. Sign up for a free account
3. Navigate to **API Keys** → Create a new key
4. Copy the key (starts with `sk-ant-...`)

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the App
```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`

### Step 4: Use the App
1. Paste your API key in the **sidebar**
2. Set your **domain** (e.g. "Hospital patient records")
3. Define your **columns** (name, type, description)
4. Click **Generate Dataset**
5. Download your CSV!

---

## 📁 Project Structure

```
synthforge/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🎓 Project Components (All inside the app)

| Tab | Component | Description |
|-----|-----------|-------------|
| Generate | **A. Prototype** | Working data generator with CSV download |
| Prompt Log | **B. Prompt Log** | Evolution from v1.0 naive to v2.1 production |
| Evaluation | **C. Evaluation** | Auto metrics + human-in-the-loop rubric |
| Ethics | **D. Ethical Reflection** | Bias, privacy, and environmental analysis |

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to **https://share.streamlit.io**
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Deploy — you get a public link to submit!

---

## 💡 How It Works

1. User defines a schema (column names, types, descriptions)
2. App constructs a structured prompt with domain context
3. Claude API generates realistic JSON rows following the schema
4. JSON is parsed into a Pandas DataFrame and displayed
5. User can download as CSV for use in ML pipelines
