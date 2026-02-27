import streamlit as st
from graphviz import Digraph
import time

st.set_page_config(layout="wide")

# ---------------- CSS (PRESERVED, with added .section for consistency) ----------------
st.markdown("""
<style>
.block-container {
    padding-left: 3rem;
    padding-right: 3rem;
}

.card {
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-bottom: 20px;
}

.diagram {
    background-color: #fafafa;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #ddd;
    font-family: monospace;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR PROGRESS ----------------
with st.sidebar:
    st.header("Workshop Progress")
    st.markdown("**Hour 5: Containerization**")
    st.progress(5/6)
    st.markdown("""
    - [x] Hour 1: Data Preparation
    - [x] Hour 2: Feature Engineering
    - [x] Hour 3: Fine-Tuning
    - [x] Hour 4: API Development
    - [ ] Hour 5: Containerization (current)
    - [ ] Hour 6: MLOps
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 5 – Containerization with Docker")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand why containerization is essential for ML deployment</li>
<li>Learn Docker fundamentals: images, containers, Dockerfile</li>
<li>Build and run a container for an ML API</li>
<li>Explore orchestration concepts (Kubernetes) and scaling</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WHY CONTAINERS ----------------
st.header("Why Containers?")
st.markdown("""
Containers package your code, model, and dependencies into a single, portable unit. They ensure consistency across environments (development, testing, production) and simplify scaling.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**Benefits:**
- Reproducibility
- Isolation
- Lightweight (vs. VMs)
- Fast startup
- Scalability
""")

with col2:
    st.markdown("""
**Use Cases:**
- Deploying ML APIs
- Microservices
- Batch processing
- Multi-cloud portability
""")

st.divider()

# ---------------- DOCKER ARCHITECTURE ----------------
st.header("Docker Architecture")
st.markdown("""
<div class="diagram">
Docker uses a client-server architecture:
- **Client** (CLI) sends commands to
- **Daemon** (server) which builds, runs, and manages containers.
- **Images** are read-only templates.
- **Containers** are runnable instances of images.
</div>
""", unsafe_allow_html=True)

docker_arch = Digraph()
docker_arch.node("Client", shape="box")
docker_arch.node("Daemon", shape="box")
docker_arch.node("Image", shape="box")
docker_arch.node("Container", shape="box")
docker_arch.node("Registry", shape="box")

docker_arch.edge("Client", "Daemon", label="commands")
docker_arch.edge("Daemon", "Image", label="build/pull")
docker_arch.edge("Daemon", "Container", label="run")
docker_arch.edge("Daemon", "Registry", label="push/pull")
docker_arch.edge("Registry", "Image", style="dashed")

st.graphviz_chart(docker_arch)

st.divider()

# ---------------- DOCKERFILE AND BUILD PROCESS ----------------
st.header("Dockerfile and Build Process")
st.markdown("""
A `Dockerfile` is a script that defines how to build an image.
""")

st.code("""
# Example Dockerfile for an ML API
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and application code
COPY model.pkl .
COPY app.py .

# Expose the port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
""", language="dockerfile")

st.markdown("**Build process:**")
build_diag = Digraph()
build_diag.node("Code")
build_diag.node("Dockerfile")
build_diag.node("Image")
build_diag.node("Container")

build_diag.edge("Code", "Dockerfile", label="written in")
build_diag.edge("Dockerfile", "Image", label="docker build")
build_diag.edge("Image", "Container", label="docker run")

st.graphviz_chart(build_diag)

st.divider()

# ---------------- WORKSHOP CONTEXT ----------------
st.info("""
**Workshop Context:** In Hour 4, you built an ML API with FastAPI. Now you'll containerize that API. 
In Hour 6, you'll learn how to deploy and monitor such containers in production.
""")

st.divider()

# ---------------- INTERACTIVE BUILD SIMULATION ----------------
st.header("Build Simulation")
st.markdown("""
Click the button to simulate the Docker build process for the ML API.
""")

if st.button("Build Container Image"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("Step 1: Reading Dockerfile", 0.2),
        ("Step 2: Pulling base image python:3.9-slim", 0.4),
        ("Step 3: Installing dependencies (scikit-learn, fastapi, uvicorn)", 0.6),
        ("Step 4: Copying model and application code", 0.8),
        ("Step 5: Exposing port 8000", 0.9),
        ("Step 6: Image built successfully", 1.0)
    ]
    
    for step, progress in steps:
        status_text.text(step)
        progress_bar.progress(progress)
        time.sleep(0.7)
    
    st.success("Image built: ml-api:latest")
    st.balloons()  # optional, but can be removed if no emojis wanted; balloons are not emojis, but they are celebratory. Remove if undesired. I'll keep as it's fun.
    # Alternative: just a success message.
else:
    st.info("Click the button to simulate.")

st.divider()

# ---------------- RUNNING A CONTAINER ----------------
st.header("Running a Container")
st.markdown("""
Once the image is built, you can run a container:
```bash
docker run -p 8000:8000 ml-api:latest

This maps port 8000 of the container to port 8000 on your host, making the API accessible at http://localhost:8000.
""")

st.divider()

#---------------- SCALING WITH CONTAINERS ----------------
st.header("Scaling with Containers")
st.markdown("""
Containers make scaling easy: you can run multiple instances behind a load balancer.
""")

replicas = st.slider("Number of container replicas", 1, 10, 3)

scale_diag = Digraph()
scale_diag.node("Load Balancer", shape="box")
for i in range(replicas):
	node_name = f"API Container {i+1}"
	scale_diag.node(node_name, shape="box")
	scale_diag.edge("Load Balancer", node_name)

st.graphviz_chart(scale_diag)

st.markdown(f"With {replicas} replicas, the system can handle more requests and provide redundancy.")

st.divider()

#---------------- ORCHESTRATION: KUBERNETES ----------------
st.header("Orchestration: Kubernetes")
st.markdown("""
When you have many containers, you need an orchestrator like Kubernetes to manage them.
""")

k8s_diag = Digraph()
k8s_diag.node("Kubernetes Cluster", shape="box")

#Pods
for i in range(3):
	pod_name = f"Pod {i+1}"
	k8s_diag.node(pod_name, shape="box")
	k8s_diag.edge("Kubernetes Cluster", pod_name)

st.graphviz_chart(k8s_diag)

st.markdown("""
Kubernetes concepts:

Pods – smallest deployable units (one or more containers)

Deployments – manage replica sets and rolling updates

Services – provide stable networking to pods

Ingress – external access
""")

st.divider()

#---------------- PRODUCTION CONSIDERATIONS ----------------
st.header("Production Considerations")
st.markdown("""

Container Registry – store and share images (Docker Hub, AWS ECR, GCR)

Resource Limits – CPU/memory constraints to avoid noisy neighbors

Health Checks – liveness and readiness probes

Logging – centralized logging (e.g., ELK stack)

Monitoring – metrics (Prometheus) and dashboards (Grafana)

Security – scan images for vulnerabilities, use minimal base images
""")

st.divider()

#---------------- KEY TAKEAWAYS ----------------
st.header("Key Takeaways")
st.markdown("""

Containers package code, model, and dependencies for consistent deployment.

Docker builds images from Dockerfiles; containers are running instances.

Scaling is achieved by running multiple containers behind a load balancer.

Orchestrators like Kubernetes automate container management in production.

Containerization is a critical step in the ML engineering lifecycle.
""")

st.success("Next: Hour 6 — Industry Workflow and MLOps")