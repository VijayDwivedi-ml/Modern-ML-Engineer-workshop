import streamlit as st
import json
import time
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

# ---------------- CSS (PRESERVED) ----------------
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
    st.markdown("**Hour 4: API Development**")
    st.progress(4/6)
    st.markdown("""
    - [x] Hour 1: Data Preparation
    - [x] Hour 2: Feature Engineering
    - [x] Hour 3: Fine-Tuning
    - [ ] Hour 4: API Development (current)
    - [ ] Hour 5: Containerization
    - [ ] Hour 6: MLOps
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 4 – API Development for Machine Learning")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand the role of APIs in ML systems</li>
<li>Learn FastAPI fundamentals for model deployment</li>
<li>Build and test a prediction API locally</li>
<li>Explore latency, scaling, and production considerations</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WHY APIs ----------------
st.header("Why APIs Matter")
st.markdown("""
APIs enable real-time predictions for web and mobile applications, automation, and integration with other services.
They provide scalability, standardization, monitoring, and logging.
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
**APIs enable:**
- Real-time predictions
- Web applications
- Mobile applications
- Automation
""")

with col2:
    st.markdown("""
**APIs provide:**
- Scalability
- Standardization
- Monitoring
- Logging
""")

st.divider()

# ---------------- ARCHITECTURE ----------------
st.header("API Architecture")
st.markdown("""
<div class="diagram">

Client
  │
  ▼
Internet
  │
  ▼
Load Balancer
  │
  ▼
API Server
  │
  ▼
ML Model
  │
  ▼
Database

</div>
""", unsafe_allow_html=True)

st.markdown("""
Each component plays a role:
- **Client** sends request
- **API** processes request and routes to model
- **Model** returns prediction
- **Database** stores logs or results
""")

st.divider()

# ---------------- REQUEST FLOW ----------------
st.header("Request Flow")
st.markdown("""
<div class="diagram">

User Input
    │
    ▼
JSON Request
    │
    ▼
API Endpoint
    │
    ▼
Model Prediction
    │
    ▼
JSON Response

</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- JSON EXPLANATION ----------------
st.header("JSON in APIs")
st.markdown("""
**JSON (JavaScript Object Notation)** is the standard data format for REST APIs.
- Requests and responses are text-based, human-readable.
- Easily parsed by any programming language.
- ML models typically accept feature values and return predictions in JSON.
""")

st.divider()

# ---------------- FASTAPI EXAMPLE ----------------
st.header("FastAPI Example")
st.markdown("""
FastAPI is a modern, fast web framework for building APIs with Python. It automatically generates interactive documentation.
""")

st.code("""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the trained model (e.g., intent classifier)
model = joblib.load("intent_classifier.pkl")

# Define request and response models
class Query(BaseModel):
    text: str

class Prediction(BaseModel):
    intent: str
    confidence: float

@app.post("/predict", response_model=Prediction)
def predict(query: Query):
    # Preprocess and predict
    text = query.text.lower()
    # In a real scenario, you would call model.predict_proba()
    if "return" in text:
        return {"intent": "return_order", "confidence": 0.94}
    elif "buy" in text or "purchase" in text:
        return {"intent": "purchase", "confidence": 0.88}
    elif "cancel" in text:
        return {"intent": "cancel_order", "confidence": 0.91}
    else:
        return {"intent": "unknown", "confidence": 0.60}
""", language="python")

st.info("FastAPI automatically generates interactive documentation at `/docs` when the server is running.")

st.divider()

# ---------------- WORKSHOP CONTEXT ----------------
st.info("""
**Workshop Context:** In Hour 3, you fine-tuned a transformer model. Now you're exposing it via API. 
In Hour 5, you'll containerize this API with Docker.
""")

st.divider()

# ---------------- SIMULATED MODEL ----------------
st.header("Model Logic (Simulated)")
st.markdown("""
This workshop uses a simulated model based on keyword matching. The logic mimics a real NLP classifier.
""")

st.code("""
def fake_model_prediction(text):
    # Simulate inference delay
    time.sleep(0.5)
    text = text.lower()
    if "return" in text:
        return {"intent": "return_order", "confidence": 0.94}
    elif "buy" in text or "purchase" in text:
        return {"intent": "purchase", "confidence": 0.88}
    elif "cancel" in text:
        return {"intent": "cancel_order", "confidence": 0.91}
    else:
        return {"intent": "unknown", "confidence": 0.60}
""", language="python")

st.divider()

# ---------------- API TESTER ----------------
st.header("API Tester")

mode = st.radio(
    "API Mode",
    ["Fake API (Workshop Mode)", "Real API"]
)

api_url = st.text_input(
    "API URL (for Real API mode)",
    "http://localhost:8000/predict"
)

user_input = st.text_input(
    "Input Text",
    "I want to return my order"
)

# Latency slider for simulation
latency_ms = st.slider("Simulated Latency (ms)", 0, 2000, 500, step=50)

# Initialize session state for request history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- FAKE MODEL FUNCTION ----------
def fake_model_prediction(text, latency):
    """Simulate an ML model's prediction logic with given latency."""
    import time
    text = text.lower()
    time.sleep(latency / 1000)  # simulate inference time

    if "return" in text:
        return {"intent": "return_order", "confidence": 0.94}
    elif "buy" in text or "purchase" in text:
        return {"intent": "purchase", "confidence": 0.88}
    elif "cancel" in text:
        return {"intent": "cancel_order", "confidence": 0.91}
    else:
        return {"intent": "unknown", "confidence": 0.60}

# ---------- BUTTON ----------
col1, col2 = st.columns(2)
with col1:
    send = st.button("Send Request", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear History", use_container_width=True)

if clear:
    st.session_state.history = []
    st.success("History cleared.")

if send:
    payload = {"text": user_input}
    start_time = time.time()

    # Display request and response side by side
    col_req, col_res = st.columns(2)

    with col_req:
        st.subheader("Request (JSON)")
        st.json(payload)

    if mode == "Fake API (Workshop Mode)":
        response = fake_model_prediction(user_input, latency_ms)
        elapsed = (time.time() - start_time) * 1000
        with col_res:
            st.subheader("Response (JSON)")
            st.json(response)
            st.caption(f"Simulated latency: {latency_ms} ms (actual: {elapsed:.0f} ms)")

        # Add to history
        st.session_state.history.append({
            "request": payload,
            "response": response,
            "latency": elapsed
        })

    else:  # Real API mode
        if not api_url:
            st.error("Please enter a valid API URL.")
        else:
            try:
                import requests
                with st.spinner("Calling real API..."):
                    resp = requests.post(api_url, json=payload, timeout=5)
                elapsed = (time.time() - start_time) * 1000
                if resp.status_code == 200:
                    response = resp.json()
                    with col_res:
                        st.subheader("Response (JSON)")
                        st.json(response)
                        st.caption(f"Actual latency: {elapsed:.0f} ms")
                    st.session_state.history.append({
                        "request": payload,
                        "response": response,
                        "latency": elapsed
                    })
                else:
                    st.error(f"API returned status code {resp.status_code}")
                    st.json(resp.text)
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.info("""
                To use the real API:
                1. Save the FastAPI code in a file `app.py`.
                2. Install dependencies: `pip install fastapi uvicorn joblib`
                3. Run: `uvicorn app:app --reload`
                4. Enter the URL above (e.g., http://localhost:8000/predict)
                """)

# Display history
if st.session_state.history:
    st.subheader("Request History")
    history_df = pd.DataFrame([
        {
            "Request": h["request"]["text"],
            "Response Intent": h["response"]["intent"],
            "Confidence": h["response"]["confidence"],
            "Latency (ms)": f"{h['latency']:.0f}"
        }
        for h in st.session_state.history[-5:]  # show last 5
    ])
    st.dataframe(history_df, use_container_width=True)

st.divider()

# ---------------- LATENCY DEMO (ENHANCED) ----------------
st.header("Latency and Load Simulation")

col1, col2 = st.columns(2)
with col1:
    if st.button("Run Load Test (10 requests)"):
        with st.spinner("Simulating load..."):
            latencies = []
            for i in range(10):
                start = time.time()
                # Simulate variable latency using exponential distribution
                sim_latency = np.random.exponential(latency_ms / 1000)
                time.sleep(sim_latency)
                latencies.append((time.time() - start) * 1000)
            st.line_chart(latencies)
            avg_lat = np.mean(latencies)
            p95_lat = np.percentile(latencies, 95)
            st.write(f"**Average:** {avg_lat:.0f} ms, **P95:** {p95_lat:.0f} ms")
            if p95_lat > 1000:
                st.warning("P95 latency exceeds 1 second – consider optimizing or scaling.")

with col2:
    st.markdown("""
    **Why latency matters:**
    - User experience degrades above 1 second.
    - Service Level Agreements (SLAs) often require p95 < 500ms.
    - Scaling and caching help reduce latency.
    """)

st.divider()

# ---------------- SCALING ----------------
st.header("Scaling APIs")
st.markdown("""
<div class="diagram">

           Load Balancer
          /      |      \\
         /       |       \\
      API1     API2     API3
        |        |        |
      Model    Model    Model

</div>
""", unsafe_allow_html=True)

st.markdown("""
Scaling improves:
- Throughput (requests per second)
- Reliability (failover)
- Availability (no single point of failure)
""")

st.divider()

# ---------------- PRODUCTION CONSIDERATIONS ----------------
st.header("Production Considerations")
st.markdown("""
- **Authentication** – API keys, OAuth
- **Rate Limiting** – Prevent abuse
- **Logging** – Track requests and errors
- **Monitoring** – Latency, error rates, traffic
- **Versioning** – Support multiple model versions
- **A/B Testing** – Compare model performance
""")

st.divider()

# ---------------- KEY TAKEAWAYS ----------------
st.header("Key Takeaways")
st.markdown("""
- APIs expose ML models to applications.
- JSON is the standard data format.
- FastAPI simplifies building robust APIs.
- Latency and scalability are critical in production.
- Monitoring and logging ensure reliability.
""")

st.success("Next: Hour 5 — Containerization")