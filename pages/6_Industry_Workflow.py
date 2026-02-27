import streamlit as st
from graphviz import Digraph
import time
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Industry Workflow",
    layout="wide"
)

# ---------------- CSS (PRESERVED) ----------------
st.markdown("""
<style>
.block-container {
    padding-left: 3rem;
    padding-right: 3rem;
}

.section {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.card {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
    background-color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR PROGRESS ----------------
with st.sidebar:
    st.header("Workshop Progress")
    st.markdown("**Hour 6: Industry Workflow & MLOps**")
    st.progress(6/6)
    st.markdown("""
    - [x] Hour 1: Data Preparation
    - [x] Hour 2: Feature Engineering
    - [x] Hour 3: Fine-Tuning
    - [x] Hour 4: API Development
    - [x] Hour 5: Containerization
    - [ ] Hour 6: Industry Workflow (current)
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 6 – Industry Workflow and MLOps")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand the end-to-end ML lifecycle in production</li>
<li>Explore MLOps components: CI/CD, monitoring, retraining</li>
<li>Learn about data drift, model decay, and triggers for retraining</li>
<li>See how teams collaborate to maintain ML systems</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WORKSHOP CONTEXT ----------------
st.info("""
**Workshop Context:** You've built and containerized an ML API. Now we'll see how it fits into a production environment with monitoring, scaling, and continuous improvement.
""")

st.divider()

# ---------------- ML PIPELINE (ENHANCED) ----------------
st.header("Machine Learning Pipeline")

col1, col2 = st.columns([2, 1])

with col1:
    pipeline = Digraph()
    pipeline.attr(rankdir="LR", size="12")
    pipeline.node("Data", shape="box", style="filled", fillcolor="lightblue")
    pipeline.node("Processing", shape="box", style="filled", fillcolor="lightblue")
    pipeline.node("Training", shape="box", style="filled", fillcolor="lightblue")
    pipeline.node("Deployment", shape="box", style="filled", fillcolor="lightblue")
    pipeline.node("Monitoring", shape="box", style="filled", fillcolor="lightblue")
    pipeline.node("Retraining", shape="box", style="filled", fillcolor="lightblue")
    
    pipeline.edge("Data", "Processing")
    pipeline.edge("Processing", "Training")
    pipeline.edge("Training", "Deployment")
    pipeline.edge("Deployment", "Monitoring")
    pipeline.edge("Monitoring", "Retraining")
    pipeline.edge("Retraining", "Training", label="trigger", style="dashed")
    
    st.graphviz_chart(pipeline)

with col2:
    st.markdown("""
    **Pipeline Stages:**
    1. Data Collection
    2. Data Processing
    3. Model Training
    4. Deployment
    5. Monitoring
    6. Retraining
    """)

st.divider()

# ---------------- INTERACTIVE PIPELINE ----------------
st.header("Interactive Pipeline Exploration")

stage = st.selectbox(
    "Select a pipeline stage to learn more:",
    [
        "Data Collection",
        "Data Processing",
        "Model Training",
        "Deployment",
        "Monitoring",
        "Retraining"
    ]
)

stage_info = {
    "Data Collection": """
### Data Collection
- **Sources:** Databases, user interactions, sensors, logs, third-party APIs.
- **Challenges:** Data quality, privacy, volume, velocity.
- **Tools:** Apache Kafka, AWS Kinesis, Airflow.
- **Output:** Raw data stored in data lakes or warehouses.
""",
    "Data Processing": """
### Data Processing
- **Steps:** Cleaning, normalization, feature engineering, encoding, splitting.
- **Challenges:** Reproducibility, data leakage, schema evolution.
- **Tools:** Spark, Pandas, Dask, TFX.
- **Output:** Feature store or training datasets.
""",
    "Model Training": """
### Model Training
- **Includes:** Experiment tracking, hyperparameter tuning, model validation.
- **Challenges:** Overfitting, underfitting, concept drift.
- **Tools:** MLflow, Kubeflow, SageMaker, Vertex AI.
- **Output:** Trained model artifacts.
""",
    "Deployment": """
### Deployment
- **Strategies:** Online (API), batch, edge.
- **Tools:** FastAPI, Docker, Kubernetes, TensorFlow Serving.
- **Challenges:** Versioning, rollback, A/B testing, canary releases.
- **Output:** Live model serving endpoint.
""",
    "Monitoring": """
### Monitoring
- **Metrics:** Prediction latency, throughput, error rates, data drift, feature drift.
- **Tools:** Prometheus, Grafana, Evidently AI, WhyLabs.
- **Challenges:** Alerting, root cause analysis.
- **Output:** Dashboards, alerts.
""",
    "Retraining": """
### Retraining
- **Triggers:** Performance drop, data drift, new data availability.
- **Strategies:** Scheduled, on-demand, continuous.
- **Tools:** CI/CD pipelines (Jenkins, GitHub Actions), Kubeflow Pipelines.
- **Output:** Updated model deployed.
"""
}

st.markdown(stage_info[stage])

st.divider()

# ---------------- PRODUCTION ARCHITECTURE ----------------
st.header("Production Architecture with MLOps")

arch = Digraph()
arch.attr(rankdir="TB")

arch.node("Users", shape="ellipse")
arch.node("Load Balancer", shape="box")
arch.node("API Servers", shape="box", label="API Servers\n(container replicas)")
arch.node("Model Serving", shape="box", label="Model\n(versioned)")
arch.node("Feature Store", shape="box")
arch.node("Monitoring", shape="box", style="filled", fillcolor="lightcoral")
arch.node("Metadata Store", shape="box", label="Metadata Store\n(experiments, runs)")
arch.node("Alerting", shape="box")

arch.edge("Users", "Load Balancer")
arch.edge("Load Balancer", "API Servers")
arch.edge("API Servers", "Model Serving")
arch.edge("Model Serving", "Feature Store", style="dashed", label="features")
arch.edge("Model Serving", "Monitoring")
arch.edge("Monitoring", "Alerting", label="drift/error")
arch.edge("Metadata Store", "Model Serving", style="dashed", label="model metadata")

st.graphviz_chart(arch)

st.markdown("""
**Components explained:**
- **Load Balancer:** Distributes traffic across API instances.
- **API Servers:** Containerized FastAPI apps (from Hour 4 & 5).
- **Model Serving:** The actual ML model (can be a separate service).
- **Feature Store:** Central repository for precomputed features.
- **Monitoring:** Tracks model performance, data drift, system health.
- **Metadata Store:** Tracks experiments, model versions, training runs.
- **Alerting:** Notifies team when issues arise.
""")

st.divider()

# ---------------- DATA DRIFT SIMULATION ----------------
st.header("Data Drift Simulation")
st.markdown("""
Data drift occurs when the input data distribution changes over time, degrading model performance.
Adjust the drift intensity below and see its effect on accuracy.
""")

col1, col2 = st.columns(2)

with col1:
    drift_intensity = st.slider("Drift intensity", 0.0, 1.0, 0.3, 0.05)
    st.caption("0 = no drift, 1 = extreme drift")

# Simulate accuracy over time
np.random.seed(42)
time_steps = 30
base_accuracy = 0.92
drift_effect = drift_intensity * np.linspace(0, 0.3, time_steps)  # linear degradation
noise = np.random.normal(0, 0.01, time_steps)
accuracy = base_accuracy - drift_effect + noise
accuracy = np.clip(accuracy, 0.5, 1.0)

df_drift = pd.DataFrame({
    "Day": list(range(1, time_steps+1)),
    "Accuracy": accuracy
})

with col2:
    st.line_chart(df_drift.set_index("Day"))
    current_acc = accuracy[-1]
    st.metric("Current Accuracy", f"{current_acc:.2%}", 
              delta=f"{current_acc - base_accuracy:.2%}")

if current_acc < 0.85:
    st.warning("Accuracy dropped below threshold! Retraining recommended.")
else:
    st.success("Accuracy within acceptable range.")

st.divider()

# ---------------- RETRAINING TRIGGER SIMULATION ----------------
st.header("Retraining Trigger Simulation")
st.markdown("""
Based on monitoring metrics, retraining can be triggered automatically. Click below to simulate a retraining pipeline.
""")

if st.button("Simulate Retraining Pipeline"):
    steps = [
        "Detecting accuracy drop...",
        "Querying new data from feature store...",
        "Preprocessing data...",
        "Training new model version...",
        "Evaluating model...",
        "Registering model in metadata store...",
        "Running canary deployment...",
        "Switching traffic to new model..."
    ]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, step in enumerate(steps):
        status_text.text(step)
        progress_bar.progress((i+1)/len(steps))
        time.sleep(0.5)
    
    st.success("Retraining complete. New model deployed with A/B testing.")
    # Optional celebration (no emoji)
    # st.balloons()  # uncomment if desired

st.divider()

# ---------------- CI/CD FOR ML ----------------
st.header("CI/CD for Machine Learning")
st.markdown("""
Continuous Integration and Continuous Delivery (CI/CD) pipelines automate testing and deployment.
""")

cicd = Digraph()
cicd.attr(rankdir="LR")

cicd.node("Code Change", shape="box")
cicd.node("Tests", shape="box")
cicd.node("Build Image", shape="box")
cicd.node("Push to Registry", shape="box")
cicd.node("Deploy to Staging", shape="box")
cicd.node("Validation", shape="box")
cicd.node("Deploy to Production", shape="box")

cicd.edge("Code Change", "Tests")
cicd.edge("Tests", "Build Image")
cicd.edge("Build Image", "Push to Registry")
cicd.edge("Push to Registry", "Deploy to Staging")
cicd.edge("Deploy to Staging", "Validation")
cicd.edge("Validation", "Deploy to Production")

st.graphviz_chart(cicd)

st.markdown("""
- **Tests:** Unit tests, data validation, model evaluation.
- **Staging:** Deploy to a pre-production environment for final checks.
- **Validation:** A/B tests, shadow mode, canary releases.
""")

st.divider()

# ---------------- TEAM ROLES ----------------
st.header("ML Team Structure and Roles")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Core Roles
- **Data Scientist:** Experiments, model development, feature engineering.
- **ML Engineer:** Productionizes models, builds APIs, monitoring.
- **Data Engineer:** Builds data pipelines, maintains feature store.
- **DevOps Engineer:** Infrastructure, CI/CD, scaling.
- **Product Manager:** Defines requirements, prioritizes.
""")

with col2:
    st.markdown("""
### Collaboration Workflow
1. **PM** defines business problem.
2. **DS** experiments and delivers model.
3. **MLE** packages model into API.
4. **DE** ensures data freshness.
5. **DevOps** deploys and monitors.
6. **All** review dashboards and plan retraining.
""")

st.divider()

# ---------------- KEY TAKEAWAYS ----------------
st.header("Key Takeaways")

st.markdown("""
<div class="card">
<ul>
<li>ML in production is a continuous cycle, not a one-time effort.</li>
<li>Monitoring data drift and model performance is critical.</li>
<li>CI/CD pipelines automate safe model updates.</li>
<li>Cross-functional teams are essential for success.</li>
<li>MLOps tools (Kubeflow, MLflow, Prometheus) streamline operations.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.success("Congratulations! You've completed the workshop. Next, explore the Regression and Classification demos from the sidebar.")