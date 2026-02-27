import streamlit as st
from graphviz import Digraph

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Workshop Pipeline",
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

# ---------------- TITLE ----------------
st.title("Modern Machine Learning Workshop")
st.subheader("End-to-End ML Pipeline")
st.write("Instructor: Vijay Dwivedi")

st.divider()

# ---------------- WORKSHOP OVERVIEW ----------------
st.header("Workshop Overview")

st.markdown("""
This workshop walks through the **complete lifecycle of a machine learning system**.
From understanding data to deploying models in production, each hour represents a real stage in an ML engineer’s workflow.

You will learn:
- Data understanding and preparation
- Feature engineering with embeddings
- Fine-tuning transformer models
- Building ML APIs with FastAPI
- Containerization with Docker
- MLOps and production monitoring
""")

st.divider()

# ---------------- PREREQUISITES ----------------
st.header("Prerequisites")

st.markdown("""
<div class="section-card">
Basic knowledge of:
<ul>
<li>Python programming</li>
<li>Pandas and Scikit-learn</li>
<li>Transformer models and Hugging Face</li>
<li>FastAPI and Docker fundamentals</li>
<li>Introductory machine learning concepts</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- SCHEDULE TABLE ----------------
st.header("Workshop Schedule")

schedule_data = {
    "Hour": ["1", "2", "3", "4", "5", "6"],
    "Topic": [
        "Foundation & Data Preparation",
        "Feature Engineering",
        "Fine-Tuning",
        "API Development",
        "Containerization",
        "Industry Workflow"
    ],
    "Activity": [
        "Clean and explore customer support queries; analyze intent categories.",
        "Generate embeddings and train a baseline intent classifier.",
        "Fine-tune a transformer model to improve classification accuracy.",
        "Build a FastAPI endpoint that classifies queries in real time.",
        "Package the API into a Docker container and run it locally.",
        "Understand how the system fits into a real-world ML deployment pipeline."
    ]
}

import pandas as pd
df_schedule = pd.DataFrame(schedule_data)
st.table(df_schedule)

st.divider()

# ---------------- END-TO-END PIPELINE ----------------
st.header("End-to-End ML Pipeline")

def pipeline_diagram():
    dot = Digraph()
    dot.attr(rankdir="LR", size="16")

    dot.node("H1", "Hour 1\nData Understanding\n\nEDA\nCleaning\nIntents", shape="box")
    dot.node("H2", "Hour 2\nFeature Engineering\n\nEmbeddings\nBaseline Model", shape="box")
    dot.node("H3", "Hour 3\nFine-Tuning\n\nTransformers\nTraining", shape="box")
    dot.node("H4", "Hour 4\nAPI\n\nFastAPI\nInference", shape="box")
    dot.node("H5", "Hour 5\nDocker\n\nContainers\nDeployment", shape="box")
    dot.node("H6", "Hour 6\nMLOps\n\nMonitoring\nProduction", shape="box")

    dot.edge("H1", "H2")
    dot.edge("H2", "H3")
    dot.edge("H3", "H4")
    dot.edge("H4", "H5")
    dot.edge("H5", "H6")

    return dot

st.graphviz_chart(pipeline_diagram())

st.divider()

# ---------------- HOUR DETAILS ----------------
st.header("Workshop Structure")

# Wrap each hour in a section-card for consistency
hours = [
    {
        "title": "Hour 1 — Foundation & Data Preparation",
        "focus": [
            "Dataset exploration (customer support queries)",
            "Intent category analysis",
            "Text cleaning and preprocessing",
            "Data understanding for ML"
        ]
    },
    {
        "title": "Hour 2 — Feature Engineering",
        "focus": [
            "TF-IDF and word embeddings",
            "Converting text to numerical representations",
            "Training a baseline classifier",
            "Evaluating baseline performance"
        ]
    },
    {
        "title": "Hour 3 — Fine-Tuning",
        "focus": [
            "Introduction to transformer models",
            "Attention mechanism",
            "Fine-tuning a pre-trained model",
            "Evaluation and comparison with baseline"
        ]
    },
    {
        "title": "Hour 4 — API Development",
        "focus": [
            "Building REST APIs with FastAPI",
            "Handling JSON requests/responses",
            "Integrating the trained model",
            "Testing the API locally"
        ]
    },
    {
        "title": "Hour 5 — Containerization",
        "focus": [
            "Docker basics: images and containers",
            "Writing a Dockerfile",
            "Building and running the container",
            "Exposing the API"
        ]
    },
    {
        "title": "Hour 6 — Industry Workflow",
        "focus": [
            "MLOps concepts: monitoring, retraining",
            "Deployment strategies",
            "Production system architecture",
            "Roles in an ML team"
        ]
    }
]

col1, col2 = st.columns(2)

for i, hour in enumerate(hours):
    with col1 if i < 3 else col2:
        st.markdown(f"""
        <div class="section-card">
            <h3>{hour['title']}</h3>
            <ul>
            {''.join([f'<li>{item}</li>' for item in hour['focus']])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ---------------- NLP EVOLUTION ----------------
st.header("Evolution of NLP")

def nlp_evolution():
    dot = Digraph()
    dot.attr(rankdir="LR")

    dot.node("bow", "Bag of Words\nTF-IDF", shape="box")
    dot.node("w2v", "Word2Vec\nEmbeddings", shape="box")
    dot.node("bert", "Transformers\nBERT", shape="box")
    dot.node("fine", "Fine-Tuning", shape="box")

    dot.edge("bow", "w2v")
    dot.edge("w2v", "bert")
    dot.edge("bert", "fine")

    return dot

st.graphviz_chart(nlp_evolution())

st.divider()

# ---------------- DEMO APPLICATIONS ----------------
st.header("Interactive Demos")

st.markdown("""
Two end-to-end ML demos are included in this workshop. They illustrate the concepts covered in the hours above using real datasets.

- **Regression Demo** – Full pipeline with the Diabetes dataset (linear regression, random forest, evaluation, overfitting).
- **Classification Demo** – Full pipeline with the Digits dataset (PCA, logistic regression, random forest, confusion matrix).

Use the sidebar to navigate between the main page and these demos.
""")

st.divider()

# ---------------- PRODUCTION ML SYSTEM ----------------
st.header("Production ML System")

def production_diagram():
    dot = Digraph()
    dot.attr(rankdir="LR")

    dot.node("user", "User", shape="ellipse")
    dot.node("api", "API", shape="box")
    dot.node("model", "Model", shape="box")
    dot.node("monitor", "Monitoring", shape="box")
    dot.node("data", "Data", shape="box")
    dot.node("train", "Retraining", shape="box")

    dot.edge("user", "api")
    dot.edge("api", "model")
    dot.edge("model", "api")
    dot.edge("api", "monitor")
    dot.edge("monitor", "data")
    dot.edge("data", "train")
    dot.edge("train", "model")

    return dot

st.graphviz_chart(production_diagram())

st.divider()

st.success("Begin with Hour 1 — Data Preparation. Use the sidebar to access the regression and classification demos.")