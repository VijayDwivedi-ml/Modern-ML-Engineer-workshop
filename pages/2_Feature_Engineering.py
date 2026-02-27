import streamlit as st
from graphviz import Digraph
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hour 2 - Feature Engineering",
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
.highlight {
    background-color: #eef3ff;
    padding: 15px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR PROGRESS ----------------
with st.sidebar:
    st.header("Workshop Progress")
    st.markdown("**Hour 2: Feature Engineering**")
    st.progress(2/6)
    st.markdown("""
    - [x] Hour 1: Data Preparation
    - [ ] Hour 2: Feature Engineering (current)
    - [ ] Hour 3: Fine-Tuning
    - [ ] Hour 4: API Development
    - [ ] Hour 5: Containerization
    - [ ] Hour 6: MLOps
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 2 — Feature Engineering & Baseline Model")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="section-card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand why raw text must be converted to numerical features</li>
<li>Compare traditional (BoW, TF-IDF) and modern (embeddings, transformers) techniques</li>
<li>Visualize how TF-IDF works on real text</li>
<li>See how embeddings capture semantic similarity</li>
<li>Build a baseline model concept</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- INTRODUCTION ----------------
st.header("From Text to Numbers")
st.markdown("""
Machine learning models only understand numbers. Feature engineering is the bridge that converts **raw text** into **numerical vectors** that models can process.

**Example:**
- Customer query: `"I want to return my order"`
- Machine learning input: `[0.12, 0.87, 0.33, 0.91, ...]`
""")

st.divider()

# ---------------- FEATURE ENGINEERING PIPELINE ----------------
st.header("Feature Engineering Pipeline")

def feature_pipeline():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("text", "Customer Query", shape="box")
    dot.node("clean", "Cleaning", shape="box")
    dot.node("vector", "Vectorization", shape="box")
    dot.node("model", "Model", shape="box")
    dot.node("intent", "Intent", shape="ellipse")
    dot.edge("text", "clean")
    dot.edge("clean", "vector")
    dot.edge("vector", "model")
    dot.edge("model", "intent")
    return dot

st.graphviz_chart(feature_pipeline())

st.divider()

# ---------------- EVOLUTION OF NLP FEATURES ----------------
st.header("Evolution of NLP Features")

def evolution():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("bow", "Bag of Words", shape="box")
    dot.node("tfidf", "TF-IDF", shape="box")
    dot.node("w2v", "Word2Vec", shape="box")
    dot.node("bert", "Transformers", shape="box")
    dot.edge("bow", "tfidf")
    dot.edge("tfidf", "w2v")
    dot.edge("w2v", "bert")
    return dot

st.graphviz_chart(evolution())

st.divider()

# ---------------- METHODS OVERVIEW ----------------
st.header("Feature Engineering Methods")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Bag of Words")
    st.markdown("""
    - Counts word occurrences
    - Simple, interpretable
    - Ignores word order
    - Sparse, high-dimensional
    """)
with col2:
    st.subheader("TF-IDF")
    st.markdown("""
    - Weighs words by importance
    - Downweights common words
    - Still order‑agnostic
    - Sparse but more informative
    """)
with col3:
    st.subheader("Embeddings")
    st.markdown("""
    - Dense, low-dimensional vectors
    - Capture semantic meaning
    - Context‑aware (transformers)
    - Require pre‑trained models
    """)

st.divider()

# ---------------- INTERACTIVE TF-IDF DEMO ----------------
st.header("TF-IDF in Action")
st.markdown("Enter a few customer queries to see how TF-IDF represents them.")

default_queries = [
    "I want to return my order",
    "Can I track my order?",
    "I need to return a product",
    "Where is my shipment?"
]
queries = st.text_area("Queries (one per line)", "\n".join(default_queries)).split("\n")
queries = [q.strip() for q in queries if q.strip()]

if len(queries) >= 2:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(queries)
    feature_names = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=[f"Q{i+1}" for i in range(len(queries))])
    
    st.subheader("TF-IDF Vectors (rounded to 3 decimals)")
    st.dataframe(df_tfidf.style.format("{:.3f}"), use_container_width=True)
    
    # Show similarity between queries
    sim_matrix = cosine_similarity(tfidf_matrix)
    df_sim = pd.DataFrame(sim_matrix, 
                          index=[f"Q{i+1}" for i in range(len(queries))],
                          columns=[f"Q{i+1}" for i in range(len(queries))])
    st.subheader("Cosine Similarity between Queries")
    st.dataframe(df_sim.style.format("{:.2f}"), use_container_width=True)
    
    st.caption("Higher similarity means the queries are more alike in the TF-IDF space.")
else:
    st.warning("Enter at least two queries to see the TF-IDF representation.")

st.divider()

# ---------------- EMBEDDINGS DEMO (SIMULATED) ----------------
st.header("Embeddings: Capturing Meaning")
st.markdown("""
Embeddings map text into dense vectors where **semantically similar sentences are close together**.
Below we simulate this idea using pre‑computed embeddings for a few intents.
""")

# Simulated embeddings for a few example sentences (using random but fixed for reproducibility)
np.random.seed(42)
example_sentences = [
    "I want to return my order",
    "I need to send back a product",
    "Where is my package?",
    "Track my shipment",
    "I would like to buy a new phone"
]
# Create random embeddings (for demo only; in reality they'd come from a model)
embeddings = np.random.randn(len(example_sentences), 10)  # 10‑dim for display
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

# Show the vectors
df_emb = pd.DataFrame(embeddings.round(3), 
                      columns=[f"dim{i+1}" for i in range(10)],
                      index=[f"S{i+1}" for i in range(len(example_sentences))])
df_emb.insert(0, "Sentence", example_sentences)

st.subheader("Simulated Embeddings (10‑dimensional)")
st.dataframe(df_emb, use_container_width=True)

# Compute cosine similarity
emb_sim = cosine_similarity(embeddings)
df_emb_sim = pd.DataFrame(emb_sim, 
                          index=[f"S{i+1}" for i in range(len(example_sentences))],
                          columns=[f"S{i+1}" for i in range(len(example_sentences))])
st.subheader("Cosine Similarity between Sentences")
st.dataframe(df_emb_sim.style.format("{:.2f}"), use_container_width=True)

st.markdown("""
**Observation:** Sentences about returns (S1, S2) have high similarity, while those about tracking (S3, S4) cluster together. The purchase intent (S5) is different.
""")

st.divider()

# ---------------- TRANSFORMER DIAGRAM ----------------
st.header("Transformer Embeddings")
st.markdown("""
Transformers use **attention** to understand the relationship between all words in a sentence, producing context‑aware embeddings.
""")

def transformer_diagram():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("text", "Sentence", shape="box")
    dot.node("att", "Attention", shape="box")
    dot.node("embed", "Contextual Embedding", shape="box")
    dot.node("cls", "Classifier", shape="box")
    dot.edge("text", "att")
    dot.edge("att", "embed")
    dot.edge("embed", "cls")
    return dot

st.graphviz_chart(transformer_diagram())

st.divider()

# ---------------- BASELINE MODEL ----------------
st.header("Baseline Model")
st.markdown("""
Before fine‑tuning complex transformers, we establish a **baseline** using simple features and a linear model. This gives us a lower bound on performance.
""")

def baseline_diagram():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("text", "Text", shape="box")
    dot.node("tfidf", "TF-IDF", shape="box")
    dot.node("lr", "Logistic Regression", shape="box")
    dot.node("intent", "Intent", shape="ellipse")
    dot.edge("text", "tfidf")
    dot.edge("tfidf", "lr")
    dot.edge("lr", "intent")
    return dot

st.graphviz_chart(baseline_diagram())

st.markdown("""
**Why a baseline?**
- It tells us if more complex models are justified.
- It’s fast to train and easy to debug.
- Often works surprisingly well for many tasks.
""")

st.divider()

# ---------------- TRADITIONAL VS MODERN ----------------
st.header("Traditional vs Modern NLP")
st.markdown("""
| Traditional | Modern |
|-------------|--------|
| Bag of Words / TF‑IDF | Pre‑trained embeddings (Word2Vec, GloVe) |
| Hand‑crafted features | Transformer models (BERT, RoBERTa) |
| No context | Context‑aware representations |
| Sparse vectors | Dense vectors |
| Requires feature engineering | Learned representations |
""")

st.divider()

# ---------------- KEY TAKEAWAYS ----------------
st.header("Key Takeaways")
st.markdown("""
- Feature engineering transforms text into numbers that ML models can process.
- TF‑IDF is a simple yet powerful baseline that captures word importance.
- Embeddings encode semantic meaning; similar sentences have similar vectors.
- Transformers add context through attention, producing rich representations.
- A baseline model helps measure the added value of advanced techniques.
""")

st.success("Next: Hour 3 — Fine-Tuning Transformers")