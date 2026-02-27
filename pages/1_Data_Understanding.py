import streamlit as st
import pandas as pd
import plotly.express as px
import os
from collections import Counter

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hour 1 - Data Understanding",
    layout="wide"
)

# ---------------- CSS (PRESERVED) ----------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

.section-card {
    padding: 20px;
    border-radius: 10px;
    background-color: #f7f9fc;
    border: 1px solid #e6eaf1;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR PROGRESS ----------------
with st.sidebar:
    st.header("Workshop Progress")
    st.markdown("**Hour 1: Data Understanding**")
    st.progress(1/6)
    st.markdown("""
    - [ ] Hour 1: Data Understanding (current)
    - [ ] Hour 2: Feature Engineering
    - [ ] Hour 3: Fine-Tuning
    - [ ] Hour 4: API Development
    - [ ] Hour 5: Containerization
    - [ ] Hour 6: MLOps
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 1 — Data Understanding")
st.subheader("Dataset Exploration and Analysis")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="section-card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand the structure of the customer support dataset</li>
<li>Analyze intent and category distributions</li>
<li>Explore text length and content patterns</li>
<li>Identify data quality issues (missing values, duplicates)</li>
<li>Extract insights to guide feature engineering</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- DATA LOADING ----------------
@st.cache_data
def load_data():
    local_path = "data/Bitext_v11.csv"
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        source = "Local CSV (data/Bitext_v11.csv)"
    else:
        from datasets import load_dataset
        dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        df = dataset["train"].to_pandas()
        source = "HuggingFace Dataset"
    return df, source

df, source = load_data()
st.info(f"**Data Source:** {source}")

st.divider()

# ---------------- DATASET OVERVIEW ----------------
st.header("1. Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", df.shape[1])
col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

st.write("**First 5 rows:**")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# ---------------- COLUMN TYPES ----------------
st.header("2. Column Types and Info")
types_df = pd.DataFrame({
    "Column": df.columns,
    "Type": df.dtypes.astype(str),
    "Non-Null Count": df.count().values,
    "Null Count": df.isna().sum().values
})
st.dataframe(types_df, use_container_width=True)

st.divider()

# ---------------- INTENT DISTRIBUTION ----------------
if "intent" in df.columns:
    st.header("3. Intent Distribution")
    intent_counts = df["intent"].value_counts().reset_index()
    intent_counts.columns = ["intent", "count"]
    
    fig = px.bar(
        intent_counts,
        x="intent",
        y="count",
        height=400,
        title="Distribution of Intents",
        labels={"intent": "Intent", "count": "Number of Samples"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(f"Total unique intents: {intent_counts.shape[0]} | Most frequent: {intent_counts.iloc[0]['intent']} ({intent_counts.iloc[0]['count']} samples)")
    st.divider()

# ---------------- CATEGORY DISTRIBUTION ----------------
if "category" in df.columns:
    st.header("4. Category Distribution")
    category_counts = df["category"].value_counts().reset_index()
    category_counts.columns = ["category", "count"]
    
    fig = px.pie(
        category_counts,
        names="category",
        values="count",
        height=400,
        title="Proportion of Categories"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# ---------------- INTENT vs CATEGORY HEATMAP ----------------
if "intent" in df.columns and "category" in df.columns:
    st.header("5. Intent vs Category Heatmap")
    heatmap = pd.crosstab(df["category"], df["intent"])
    fig = px.imshow(
        heatmap,
        text_auto=True,
        aspect="auto",
        height=500,
        title="Intent Distribution per Category",
        labels=dict(x="Intent", y="Category", color="Count")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This shows which intents belong to which categories (often one-to-many).")
    st.divider()

# ---------------- TEXT LENGTH ANALYSIS ----------------
text_col = None
for col in ["instruction", "query", "text"]:
    if col in df.columns:
        text_col = col
        break

if text_col:
    st.header(f"6. {text_col.capitalize()} Length Analysis")
    df["text_len"] = df[text_col].astype(str).str.len()
    
    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(
            df,
            x="text_len",
            nbins=50,
            height=350,
            title="Distribution of Text Length"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        fig_box = px.box(
            df,
            y="text_len",
            height=350,
            title="Box Plot of Text Length"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.metric("Average Length", f"{df['text_len'].mean():.1f} characters")
    st.metric("Max Length", f"{df['text_len'].max()} characters")
    st.divider()

# ---------------- MESSAGE EXPLORER ----------------
st.header("7. Message Explorer")
row_index = st.slider("Select row index to inspect", 0, len(df) - 1, 0)
row = df.iloc[row_index]

# Display selected fields dynamically
for col in df.columns:
    if col not in ["text_len"]:  # skip computed column
        st.markdown(f"**{col}:**")
        st.write(row[col])
st.divider()

# ---------------- DATA QUALITY ----------------
st.header("8. Data Quality Assessment")
col1, col2, col3 = st.columns(3)
with col1:
    missing_total = df.isna().sum().sum()
    st.metric("Total Missing Values", missing_total)
with col2:
    missing_rows = df.isna().any(axis=1).sum()
    st.metric("Rows with Missing Values", missing_rows)
with col3:
    duplicates = df.duplicated().sum()
    st.metric("Duplicate Rows", duplicates)

# Show columns with missing values if any
if missing_total > 0:
    missing_cols = df.columns[df.isna().any()].tolist()
    st.write("**Columns with missing values:**", missing_cols)
st.divider()

# ---------------- TOP WORDS (with optional stopwords) ----------------
if text_col:
    st.header("9. Top Words in Text")
    
    # Simple word frequency
    text_series = df[text_col].astype(str)
    all_text = " ".join(text_series).lower()
    words = all_text.split()
    
    # Option to remove common English stopwords
    remove_stopwords = st.checkbox("Remove common English stopwords", value=True)
    if remove_stopwords:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    
    word_counts = Counter(words).most_common(20)
    top_words_df = pd.DataFrame(word_counts, columns=["word", "count"])
    
    fig = px.bar(
        top_words_df,
        x="word",
        y="count",
        height=400,
        title="Top 20 Most Frequent Words"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()

# ---------------- KEY OBSERVATIONS ----------------
st.header("Key Observations")
st.markdown("""
Based on the exploration above, we can note:
- **Class balance:** Are some intents or categories underrepresented?
- **Text length variation:** Short queries might need different handling than long ones.
- **Missing values:** May need imputation or removal.
- **Common words:** Give insight into domain language and potential stopwords.
- **Intent-category relationship:** Helps understand the labeling hierarchy.
""")

st.success("Next: Hour 2 — Feature Engineering")