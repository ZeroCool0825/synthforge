import streamlit as st
import anthropic
import pandas as pd
import json
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SynthForge – Synthetic Data Generator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  /* Base */
  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
      background-color: #0a0e1a;
      color: #e0e6f0;
  }
  .stApp { background-color: #0a0e1a; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #0f1525;
      border-right: 1px solid #1e2d4a;
  }
  [data-testid="stSidebar"] .stTextInput input {
      background: #1a2540;
      border: 1px solid #2a3d60;
      color: #c8d8f0;
      border-radius: 8px;
      font-family: 'Space Mono', monospace;
      font-size: 12px;
  }

  /* Header */
  .hero-header {
      text-align: center;
      padding: 2.5rem 0 1.5rem;
  }
  .hero-title {
      font-family: 'Space Mono', monospace;
      font-size: 3rem;
      font-weight: 700;
      letter-spacing: -2px;
      background: linear-gradient(135deg, #4fc3f7 0%, #81d4fa 40%, #a5f3fc 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.25rem;
  }
  .hero-sub {
      color: #5a7a9e;
      font-size: 1rem;
      font-weight: 300;
      letter-spacing: 3px;
      text-transform: uppercase;
  }

  /* Cards */
  .card {
      background: #0f1525;
      border: 1px solid #1e2d4a;
      border-radius: 12px;
      padding: 1.5rem;
      margin-bottom: 1rem;
  }
  .card-title {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #4fc3f7;
      margin-bottom: 1rem;
      display: flex;
      align-items: center;
      gap: 8px;
  }

  /* Column builder rows */
  .col-row {
      background: #131d30;
      border: 1px solid #1e2d4a;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      margin-bottom: 0.5rem;
  }

  /* Buttons */
  .stButton > button {
      border-radius: 8px;
      font-family: 'DM Sans', sans-serif;
      font-weight: 500;
      transition: all 0.2s ease;
      border: none;
  }
  .stButton > button[kind="primary"] {
      background: linear-gradient(135deg, #1a6eff, #0a4dd0);
      color: white;
      padding: 0.6rem 2rem;
      font-size: 1rem;
  }
  .stButton > button[kind="primary"]:hover {
      background: linear-gradient(135deg, #2a7eff, #1a5de0);
      box-shadow: 0 0 20px rgba(26,110,255,0.4);
      transform: translateY(-1px);
  }
  .stButton > button[kind="secondary"] {
      background: #1a2540;
      color: #8ab4d8;
      border: 1px solid #2a3d60;
      font-size: 0.85rem;
  }
  .stButton > button[kind="secondary"]:hover {
      background: #1e2d4a;
      color: #c8d8f0;
  }

  /* Inputs & selects */
  .stTextInput input, .stTextArea textarea, .stSelectbox select {
      background: #131d30 !important;
      border: 1px solid #2a3d60 !important;
      border-radius: 8px !important;
      color: #c8d8f0 !important;
  }
  .stNumberInput input {
      background: #131d30 !important;
      border: 1px solid #2a3d60 !important;
      color: #c8d8f0 !important;
  }

  /* Dataframe */
  .stDataFrame {
      border: 1px solid #1e2d4a;
      border-radius: 10px;
      overflow: hidden;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
      background: #0f1525;
      border-bottom: 1px solid #1e2d4a;
      gap: 0;
  }
  .stTabs [data-baseweb="tab"] {
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      letter-spacing: 1px;
      color: #5a7a9e;
      padding: 0.75rem 1.5rem;
  }
  .stTabs [aria-selected="true"] {
      color: #4fc3f7 !important;
      border-bottom: 2px solid #4fc3f7 !important;
      background: transparent !important;
  }

  /* Status badges */
  .badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-family: 'Space Mono', monospace;
  }
  .badge-blue { background: #0d2a4a; color: #4fc3f7; border: 1px solid #1a4a7a; }
  .badge-green { background: #0d2a1a; color: #4caf88; border: 1px solid #1a4a2a; }

  /* Metrics */
  .metric-box {
      background: #131d30;
      border: 1px solid #1e2d4a;
      border-radius: 10px;
      padding: 1rem;
      text-align: center;
  }
  .metric-val {
      font-family: 'Space Mono', monospace;
      font-size: 1.8rem;
      color: #4fc3f7;
      font-weight: 700;
  }
  .metric-lbl { color: #5a7a9e; font-size: 0.8rem; margin-top: 2px; }

  /* Alert */
  .custom-alert {
      background: #1a1a0d;
      border: 1px solid #4a3d00;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      color: #d4c060;
      font-size: 0.9rem;
  }
  .custom-success {
      background: #0d1a12;
      border: 1px solid #1a4a2a;
      border-radius: 8px;
      padding: 0.75rem 1rem;
      color: #4caf88;
      font-size: 0.9rem;
  }
  label { color: #8ab4d8 !important; font-size: 0.85rem !important; }
  p { color: #8ab4d8; }
  h3 { color: #c8d8f0; font-family: 'Space Mono', monospace; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ───────────────────────────────────────────────────────
if "columns" not in st.session_state:
    st.session_state.columns = [
        {"name": "age", "type": "Integer", "description": "Age of the person (18–80)"},
        {"name": "income", "type": "Float", "description": "Annual income in USD"},
        {"name": "city", "type": "String", "description": "US city name"},
        {"name": "purchased", "type": "Boolean", "description": "Whether they made a purchase"},
    ]
if "generated_df" not in st.session_state:
    st.session_state.generated_df = None
if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚗️ SynthForge")
    st.markdown("<p style='font-size:0.8rem;'>Synthetic Data Generator</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("**🔑 Anthropic API Key**")
    api_key = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get a free key at console.anthropic.com",
        label_visibility="collapsed"
    )
    if not api_key:
        st.markdown("""<div class='custom-alert'>
        ⚠️ Enter your API key above.<br><br>
        Get one free at:<br>
        <b>console.anthropic.com</b>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("<div class='custom-success'>✓ API key set</div>", unsafe_allow_html=True)

    st.divider()

    st.markdown("**⚙️ Generation Settings**")
    num_rows = st.slider("Number of rows", min_value=25, max_value=2000, value=100, step=25)
    if num_rows > 500:
        st.markdown("<p style='font-size:0.75rem;color:#4fc3f7;'>⚡ Large dataset — will use batch generation</p>", unsafe_allow_html=True)
    temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                            help="Higher = more varied data")
    model_choice = st.selectbox("Model", ["claude-sonnet-4-20250514"], index=0)

    st.divider()
    st.markdown("<p style='font-size:0.75rem;color:#3a5a7a;'>Built for Gen AI course project<br>Powered by Anthropic Claude</p>",
                unsafe_allow_html=True)

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
  <div class='hero-title'>⚗️ SynthForge</div>
  <div class='hero-sub'>AI-Powered Synthetic Dataset Generator</div>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔬 GENERATE", "📋 PROMPT LOG", "📊 EVALUATION", "⚖️ ETHICS"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        # Dataset Context
        st.markdown("<div class='card-title'>📁 DATASET CONTEXT</div>", unsafe_allow_html=True)
        dataset_domain = st.text_input(
            "Domain / Purpose",
            value="E-commerce customer behavior",
            placeholder="e.g. Medical patient records, Financial transactions..."
        )
        extra_instructions = st.text_area(
            "Extra Instructions (optional)",
            placeholder="e.g. Make income correlated with age. Cities should be US only. No nulls.",
            height=80
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Schema Builder
        st.markdown("<div class='card-title'>🗂️ SCHEMA BUILDER — Define Your Columns</div>", unsafe_allow_html=True)

        TYPES = ["String", "Integer", "Float", "Boolean", "Date", "Email", "Phone", "Category"]

        cols_to_remove = []
        for i, col in enumerate(st.session_state.columns):
            c1, c2, c3, c4 = st.columns([2, 1.5, 3, 0.5])
            with c1:
                st.session_state.columns[i]["name"] = st.text_input(
                    "Column Name", value=col["name"],
                    key=f"name_{i}", label_visibility="collapsed",
                    placeholder="column_name"
                )
            with c2:
                st.session_state.columns[i]["type"] = st.selectbox(
                    "Type", TYPES,
                    index=TYPES.index(col["type"]) if col["type"] in TYPES else 0,
                    key=f"type_{i}", label_visibility="collapsed"
                )
            with c3:
                st.session_state.columns[i]["description"] = st.text_input(
                    "Description", value=col["description"],
                    key=f"desc_{i}", label_visibility="collapsed",
                    placeholder="Describe realistic values..."
                )
            with c4:
                if st.button("✕", key=f"del_{i}", help="Remove column"):
                    cols_to_remove.append(i)

        for i in sorted(cols_to_remove, reverse=True):
            st.session_state.columns.pop(i)

        if st.button("＋ Add Column", key="add_col"):
            st.session_state.columns.append({"name": "", "type": "String", "description": ""})
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Generate Button
        generate_clicked = st.button("⚗️ Generate Dataset", type="primary", use_container_width=True)

    with col_right:
        # Summary card
        valid_cols = [c for c in st.session_state.columns if c["name"].strip()]
        st.markdown(f"""
        <div class='card'>
          <div class='card-title'>📐 SCHEMA SUMMARY</div>
          <div class='metric-box' style='margin-bottom:1rem;'>
            <div class='metric-val'>{len(valid_cols)}</div>
            <div class='metric-lbl'>Columns Defined</div>
          </div>
          <div class='metric-box' style='margin-bottom:1rem;'>
            <div class='metric-val'>{num_rows}</div>
            <div class='metric-lbl'>Rows to Generate</div>
          </div>
        """, unsafe_allow_html=True)

        for col in valid_cols:
            type_color = {
                "String": "#4fc3f7", "Integer": "#81c784", "Float": "#ffb74d",
                "Boolean": "#f06292", "Date": "#ba68c8", "Email": "#4dd0e1",
                "Phone": "#4db6ac", "Category": "#ff8a65"
            }.get(col["type"], "#8ab4d8")
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        padding:6px 0;border-bottom:1px solid #1e2d4a;'>
              <span style='font-family:Space Mono,monospace;font-size:0.8rem;color:#c8d8f0;'>
                {col['name']}
              </span>
              <span class='badge' style='background:#0d1a2a;color:{type_color};
                    border:1px solid {type_color}33;font-size:0.7rem;'>
                {col['type']}
              </span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Results appear here
        if st.session_state.generated_df is not None:
            df = st.session_state.generated_df
            st.markdown(f"""
            <div class='custom-success'>
              ✓ Generated {len(df)} rows × {len(df.columns)} columns
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buffer.getvalue(),
                file_name="synthetic_data.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Show dataframe below
    if st.session_state.generated_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>📊 GENERATED DATA PREVIEW</div>", unsafe_allow_html=True)
        st.dataframe(st.session_state.generated_df, use_container_width=True, height=400)

# ─── Generation Logic ─────────────────────────────────────────────────────────
if generate_clicked:
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar.")
    elif not valid_cols:
        st.error("⚠️ Please define at least one column with a name.")
    elif any(not c["name"].strip() for c in st.session_state.columns):
        st.error("⚠️ One or more columns have a blank name. Please fill in all column names or remove the empty row before generating.")
    else:
        schema_text = "\n".join([
            f"- {c['name']} ({c['type']}): {c['description']}"
            for c in valid_cols
        ])

        # ── Build few-shot example row from schema ───────────────────
        # Maps each column type to a realistic placeholder value
        TYPE_EXAMPLES = {
            "String":   "Springfield",
            "Integer":  42,
            "Float":    73500.50,
            "Boolean":  True,
            "Date":     "2024-03-15",
            "Email":    "jane.doe@example.com",
            "Phone":    "555-867-5309",
            "Category": "Category_A",
        }
        example_row = {
            c["name"]: TYPE_EXAMPLES.get(c["type"], "example_value")
            for c in valid_cols
        }
        few_shot_example = json.dumps([example_row], indent=2)

        system_prompt = """You are a synthetic data generation engine. Your sole job is to produce 
realistic, statistically plausible tabular datasets as valid JSON.

RULES:
1. Return ONLY a raw JSON array of objects — no markdown, no explanation, no code fences.
2. Every row must contain ALL specified columns.
3. Values must be realistic, internally consistent, and match the stated data types.
4. Vary values naturally — avoid repetition or obvious patterns.
5. Respect any correlations or constraints mentioned by the user."""

        user_prompt = f"""Generate {num_rows} rows of synthetic data for the following dataset.

DOMAIN: {dataset_domain}

SCHEMA:
{schema_text}

ADDITIONAL INSTRUCTIONS: {extra_instructions if extra_instructions else "None"}

FEW-SHOT EXAMPLE — here is one correctly formatted row to guide your output style and structure:
{few_shot_example}

Now generate {num_rows} diverse, realistic rows following the same JSON structure. Return a JSON array with exactly {num_rows} objects."""

        # Save to prompt history
        st.session_state.prompt_history.append({
            "version": len(st.session_state.prompt_history) + 1,
            "domain": dataset_domain,
            "columns": [c["name"] for c in valid_cols],
            "rows": num_rows,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "extra": extra_instructions
        })
        st.session_state.last_prompt = user_prompt

        # ── Batch settings ──────────────────────────────────────────
        BATCH_SIZE = 100  # rows per API call — stays well within token limits
        batches = []
        num_batches = (num_rows + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division

        progress_bar = st.progress(0, text="⚗️ Starting generation...")
        status_text = st.empty()

        try:
            client = anthropic.Anthropic(api_key=api_key)

            for batch_num in range(num_batches):
                rows_so_far = batch_num * BATCH_SIZE
                rows_this_batch = min(BATCH_SIZE, num_rows - rows_so_far)

                progress = rows_so_far / num_rows
                progress_bar.progress(
                    progress,
                    text=f"⚗️ Generating rows {rows_so_far + 1}–{rows_so_far + rows_this_batch} of {num_rows}  (batch {batch_num + 1}/{num_batches})"
                )

                batch_prompt = f"""Generate {rows_this_batch} rows of synthetic data for the following dataset.

DOMAIN: {dataset_domain}

SCHEMA:
{schema_text}

ADDITIONAL INSTRUCTIONS: {extra_instructions if extra_instructions else "None"}

FEW-SHOT EXAMPLE — one correctly formatted row to guide your output:
{few_shot_example}

IMPORTANT: This is batch {batch_num + 1} of {num_batches}. Ensure diversity — do NOT repeat values from previous batches. Vary all columns naturally.

Return a JSON array with exactly {rows_this_batch} objects, one per row."""

                message = client.messages.create(
                    model=model_choice,
                    max_tokens=4096,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": batch_prompt}]
                )

                raw = message.content[0].text.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip().rstrip("```").strip()

                batch_data = json.loads(raw)
                batches.extend(batch_data)

            progress_bar.progress(1.0, text=f"✅ Complete! Generated {num_rows} rows across {num_batches} batches.")

            df = pd.DataFrame(batches)
            st.session_state.generated_df = df
            st.rerun()

        except json.JSONDecodeError as e:
            progress_bar.empty()
            st.error(f"⚠️ Claude returned malformed JSON in one of the batches. Try again.\n\nDetail: {e}")
        except anthropic.AuthenticationError:
            progress_bar.empty()
            st.error("⚠️ Invalid API key. Check your key at console.anthropic.com")
        except anthropic.RateLimitError:
            progress_bar.empty()
            st.error("⚠️ Rate limit hit. Wait a moment and try again.")
        except Exception as e:
            progress_bar.empty()
            st.error(f"⚠️ Error: {str(e)}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROMPT LOG
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📋 Prompt Engineering Log")
    st.markdown("""<p>This tab documents the evolution of prompts used to generate data — 
    a required component of this project showing how prompt design improved output quality.</p>""",
    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 Prompt Evolution (Static Documentation)")

    stages = [
        {
            "version": "v1.0 — Naive Baseline",
            "prompt": 'Generate some fake customer data.',
            "result": "❌ Claude returned a prose paragraph describing data instead of actual data. No structure, no format.",
            "lesson": "Unstructured prompts produce unstructured outputs. Need explicit format instructions."
        },
        {
            "version": "v1.1 — Format Specified",
            "prompt": 'Generate 10 rows of customer data as a CSV with columns: name, age, city, income.',
            "result": "⚠️ Got CSV but values were unrealistic (all ages were 30, income was always $50,000). No variation.",
            "lesson": "Adding format helped structure but realism requires explicit variation guidance."
        },
        {
            "version": "v1.2 — Realism Added",
            "prompt": 'Generate 10 rows of realistic, varied customer data as JSON. Columns: name (full name), age (18-80, varied), city (major US city), income (correlated with age, $20k-$200k).',
            "result": "✅ Much better variation! But JSON was wrapped in markdown code fences, breaking the parser.",
            "lesson": "Must explicitly tell Claude to return raw JSON only — no markdown wrappers."
        },
        {
            "version": "v2.0 — System Prompt Separation",
            "prompt": "[SYSTEM]: You are a data generation engine. Return ONLY raw JSON arrays.\n[USER]: Generate {n} rows for domain: {domain} with schema: {schema}",
            "result": "✅ Clean JSON output, realistic data, good variation. Parser works reliably.",
            "lesson": "Separating system (behavior rules) from user (task) dramatically improves reliability."
        },
        {
            "version": "v2.1 — Few-Shot Prompting Added",
            "prompt": "Injected a dynamically-built example row into each prompt based on the user's schema. E.g: FEW-SHOT EXAMPLE: [{\"age\": 42, \"income\": 73500.50, \"city\": \"Springfield\", \"purchased\": true}]",
            "result": "✅ Structural consistency improved significantly. Claude reliably matches column names, data types, and value ranges from the example.",
            "lesson": "Few-shot examples anchor Claude's output format — especially important for custom schemas with unusual column names or types."
        },
        {
            "version": "v2.2 — Batch Generation + Diversity Hints",
            "prompt": "Split large requests into 100-row batches. Each batch prompt includes: 'This is batch N of M. Ensure diversity — do NOT repeat values from previous batches.'",
            "result": "✅ Supports up to 2,000 rows. Data stays varied across batches. Token limit errors eliminated.",
            "lesson": "Batching is essential at scale. Adding explicit diversity instructions per-batch prevents repetition across API calls."
        },
    ]

    for stage in stages:
        with st.expander(f"🔬 {stage['version']}", expanded=False):
            st.code(stage["prompt"], language="text")
            st.markdown(f"**Result:** {stage['result']}")
            st.markdown(f"**💡 Lesson Learned:** {stage['lesson']}")

    st.markdown("---")
    st.markdown("### 🔄 Live Prompt History (This Session)")

    if not st.session_state.prompt_history:
        st.info("Generate some data in the Generate tab to see live prompts recorded here.")
    else:
        for entry in reversed(st.session_state.prompt_history):
            with st.expander(f"Generation #{entry['version']} — {entry['domain']} ({entry['rows']} rows)", expanded=False):
                st.markdown("**System Prompt:**")
                st.code(entry["system_prompt"], language="text")
                st.markdown("**User Prompt:**")
                st.code(entry["user_prompt"], language="text")
                if entry["extra"]:
                    st.markdown(f"**Extra Instructions:** {entry['extra']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 📊 Output Evaluation")
    st.markdown("<p>Rubric-based assessment of generated data quality.</p>", unsafe_allow_html=True)

    if st.session_state.generated_df is None:
        st.info("📌 Generate a dataset first, then return here for evaluation.")
    else:
        df = st.session_state.generated_df

        # Auto metrics
        st.markdown("### 🤖 Automated Metrics")
        m1, m2, m3, m4 = st.columns(4)
        null_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        dup_pct = (df.duplicated().sum() / len(df) * 100)
        col_count = len(df.columns)
        row_count = len(df)

        with m1:
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{null_pct:.1f}%</div><div class='metric-lbl'>Null Rate (lower=better)</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{dup_pct:.1f}%</div><div class='metric-lbl'>Duplicate Rate</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{row_count}</div><div class='metric-lbl'>Rows Generated</div></div>", unsafe_allow_html=True)
        with m4:
            st.markdown(f"<div class='metric-box'><div class='metric-val'>{col_count}</div><div class='metric-lbl'>Columns</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Data Distribution Charts")
        st.markdown("<p>Visual proof that generated data has realistic, varied distributions — not flat or repetitive.</p>", unsafe_allow_html=True)

        # Detect numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.select_dtypes(include=["object", "bool"]).columns.tolist()]

        chart_cols = numeric_cols[:2] + cat_cols[:2]  # up to 4 charts

        if not chart_cols:
            st.info("No columns available to chart.")
        else:
            n_charts = len(chart_cols)
            grid_cols = st.columns(min(n_charts, 2))

            for i, col in enumerate(chart_cols[:4]):
                with grid_cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor("#0f1525")
                    ax.set_facecolor("#131d30")
                    ax.tick_params(colors="#8ab4d8", labelsize=9)
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#1e2d4a")
                    ax.title.set_color("#c8d8f0")
                    ax.xaxis.label.set_color("#8ab4d8")
                    ax.yaxis.label.set_color("#8ab4d8")

                    if col in numeric_cols:
                        ax.hist(df[col].dropna(), bins=15, color="#4fc3f7", edgecolor="#0a0e1a", alpha=0.85)
                        ax.set_title(f"{col} — Distribution", fontsize=11, pad=8)
                        ax.set_xlabel(col, fontsize=9)
                        ax.set_ylabel("Count", fontsize=9)
                    else:
                        # Categorical / boolean — bar chart of value counts
                        vc = df[col].astype(str).value_counts().head(10)
                        bars = ax.bar(vc.index, vc.values, color="#1a6eff", edgecolor="#0a0e1a", alpha=0.85)
                        ax.set_title(f"{col} — Value Counts", fontsize=11, pad=8)
                        ax.set_ylabel("Count", fontsize=9)
                        plt.xticks(rotation=30, ha="right", fontsize=8)

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Column-Level Stats")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Human-in-the-Loop Rubric")
        st.markdown("<p>Rate the output quality below. Scores are logged for your project documentation.</p>", unsafe_allow_html=True)

        rubric_items = [
            ("Realism", "Do values look like real-world data?"),
            ("Diversity", "Is there sufficient variation across rows?"),
            ("Type Correctness", "Are integers/strings/booleans in correct format?"),
            ("Internal Consistency", "Are correlated columns consistent? (e.g. age vs income)"),
            ("Hallucination Check", "No impossible values like negative ages or future birthdates?"),
        ]

        scores = {}
        for label, desc in rubric_items:
            c1, c2 = st.columns([2, 3])
            with c1:
                st.markdown(f"**{label}**")
                st.caption(desc)
            with c2:
                scores[label] = st.select_slider(
                    label, options=[1, 2, 3, 4, 5],
                    value=3, key=f"score_{label}", label_visibility="collapsed"
                )

        avg_score = sum(scores.values()) / len(scores)
        grade = "A" if avg_score >= 4.5 else "B" if avg_score >= 3.5 else "C" if avg_score >= 2.5 else "D"
        color = "#4caf88" if avg_score >= 4 else "#ffb74d" if avg_score >= 2.5 else "#f06292"

        st.markdown(f"""
        <div style='text-align:center;padding:2rem;background:#0f1525;border:1px solid #1e2d4a;border-radius:12px;margin-top:1rem;'>
          <div style='font-family:Space Mono,monospace;font-size:3rem;color:{color};font-weight:700;'>{avg_score:.1f}/5</div>
          <div style='font-size:1.2rem;color:#8ab4d8;margin-top:0.5rem;'>Overall Quality Grade: <b style='color:{color};'>{grade}</b></div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ETHICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## ⚖️ Ethical Reflection")
    st.markdown("""
<div class='card'>
<div class='card-title'>🔍 BIAS ANALYSIS</div>

Synthetic data generators powered by large language models (LLMs) inherit biases present in their training corpora. When generating demographic data — such as income correlated with city, age, or name — Claude may reproduce societal inequalities rather than generate truly neutral distributions. For example, if prompted to generate "realistic" income data by city, the model may reflect historical wage gaps that disadvantage certain groups. Users of SynthForge must be cautious: if the synthetic data is used to train a downstream classifier, these embedded biases will silently propagate into model predictions.

To mitigate this, SynthForge allows users to write explicit distribution constraints in the "Extra Instructions" field — such as "ensure equal income distribution across gender categories." However, it is the user's responsibility to identify which biases are relevant to their use case and actively counteract them.
</div>

<div class='card'>
<div class='card-title'>🔐 DATA PRIVACY CONCERNS</div>

A key benefit of synthetic data is that it contains no real personally identifiable information (PII), making it safer than anonymizing real datasets. SynthForge generates all data via API — no real user data is ever collected or stored. However, two privacy risks remain:

<b>1. Prompt Leakage:</b> If a user pastes real data examples into the "Extra Instructions" field to guide generation style, that data is transmitted to Anthropic's API. Users should never include real PII in prompts.

<b>2. API Key Exposure:</b> The current implementation accepts the API key via a text field. In a production deployment, this must be moved to environment variables and server-side authentication to prevent exposure.
</div>

<div class='card'>
<div class='card-title'>🌱 ENVIRONMENTAL IMPACT</div>

Each API call to Claude consumes computational energy. A typical generation of 100 rows uses approximately one API call consuming roughly 1,000–4,000 tokens. At scale — for example, generating millions of training rows — the carbon footprint becomes non-trivial. Anthropic reports working toward carbon neutrality, but users should consider whether the synthetic data approach is more efficient than alternatives like mathematical sampling (e.g., Gaussian mixtures), which require no LLM inference at all.

For this project's scope (small datasets for ML experiments), the environmental cost is minimal. But responsible AI development requires acknowledging that convenience has an energy cost.
</div>

<div class='card'>
<div class='card-title'>✅ RESPONSIBLE USE GUIDELINES</div>

• Never use synthetic data as a substitute for real validation data in high-stakes systems (medical, legal, financial decisions).
• Always document that a dataset is synthetic when sharing it.
• Review generated data for demographic bias before using it to train classifiers.
• Do not include real PII in prompts to guide generation.
• Prefer smaller, targeted generations over bulk generation to minimize compute waste.
</div>
""", unsafe_allow_html=True)
