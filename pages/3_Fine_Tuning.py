import streamlit as st
from graphviz import Digraph
import pandas as pd
import numpy as np
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hour 3 - Fine Tuning",
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
    st.markdown("**Hour 3: Fine-Tuning Transformers**")
    st.progress(3/6)
    st.markdown("""
    - [x] Hour 1: Data Preparation
    - [x] Hour 2: Feature Engineering
    - [ ] Hour 3: Fine-Tuning (current)
    - [ ] Hour 4: API Development
    - [ ] Hour 5: Containerization
    - [ ] Hour 6: MLOps
    """)
    st.markdown("---")
    st.markdown("Use the navigation above to move between hours.")

# ---------------- TITLE ----------------
st.title("Hour 3 — Fine-Tuning Transformer Models")

# ---------------- LEARNING OBJECTIVES ----------------
st.markdown("""
<div class="section-card">
<strong>Learning Objectives:</strong>
<ul>
<li>Understand why transformers outperform traditional NLP models</li>
<li>Learn how attention captures context</li>
<li>Grasp the concept of pre-training and fine-tuning</li>
<li>See a simulated fine-tuning process on intent classification</li>
<li>Compare baseline vs. fine-tuned performance</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------- WHY TRANSFORMERS ----------------
st.header("Why Transformers?")
st.markdown("""
Traditional ML pipelines require manual feature engineering (TF‑IDF, etc.). Transformers **learn representations automatically** and capture **context** through attention.

- **Baseline:** Text → TF‑IDF → Logistic Regression
- **Transformer:** Text → Embeddings → Contextual Representations → Classifier
""")

st.divider()

# ---------------- TRANSFORMER ARCHITECTURE ----------------
st.header("Transformer Architecture")
st.markdown("""
A transformer processes the entire sequence at once, using **self-attention** to weigh the importance of each word relative to others.
""")

def transformer_arch():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("input", "Input Sentence", shape="box")
    dot.node("tokens", "Tokenization", shape="box")
    dot.node("embed", "Embeddings", shape="box")
    dot.node("attn", "Multi-Head Attention", shape="box")
    dot.node("ff", "Feed-Forward", shape="box")
    dot.node("output", "Contextual Vectors", shape="box")
    dot.edge("input", "tokens")
    dot.edge("tokens", "embed")
    dot.edge("embed", "attn")
    dot.edge("attn", "ff")
    dot.edge("ff", "output")
    return dot

st.graphviz_chart(transformer_arch())

st.divider()

# ---------------- ATTENTION MECHANISM ----------------
st.header("Attention Mechanism")
st.markdown("""
Attention allows the model to focus on relevant words. For example, in *"I want to return my order"*, the word **return** attends strongly to **order**.
""")

def attention_diagram():
    dot = Digraph()
    dot.node("w1", "I")
    dot.node("w2", "return")
    dot.node("w3", "order")
    dot.edge("w2", "w3", label="strong", penwidth="2")
    dot.edge("w1", "w3", label="weak", style="dashed")
    return dot

st.graphviz_chart(attention_diagram())

st.divider()

# ---------------- PRETRAINED MODELS ----------------
st.header("Pretrained Models")
st.markdown("""
Models like **BERT**, **RoBERTa**, and **DistilBERT** are trained on massive text corpora (books, Wikipedia). They learn general language understanding, which we can adapt to our specific task.
""")

st.divider()

# ---------------- FINE-TUNING CONCEPT ----------------
st.header("What is Fine-Tuning?")
st.markdown("""
Fine-tuning takes a pretrained model and continues training on a **downstream task** (e.g., intent classification) with a small amount of task-specific data.
""")

def finetune_diagram():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("pre", "Pretrained Model (BERT)", shape="box")
    dot.node("data", "E‑commerce Queries", shape="box")
    dot.node("train", "Fine-Tuning", shape="box")
    dot.node("model", "Tuned Model", shape="box")
    dot.edge("pre", "train")
    dot.edge("data", "train")
    dot.edge("train", "model")
    return dot

st.graphviz_chart(finetune_diagram())

st.divider()

# ---------------- INTENT CLASSIFICATION MODEL ----------------
st.header("Intent Classification with Transformers")
st.markdown("""
We add a classification head on top of the transformer and train on labeled queries.
""")

def classifier():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("text", "Customer Query", shape="box")
    dot.node("bert", "Transformer", shape="box")
    dot.node("cls", "Classification Head", shape="box")
    dot.node("intent", "Intent (return, refund, etc.)", shape="ellipse")
    dot.edge("text", "bert")
    dot.edge("bert", "cls")
    dot.edge("cls", "intent")
    return dot

st.graphviz_chart(classifier())

st.divider()

# ---------------- SIMULATED FINE-TUNING DEMO ----------------
st.header("Fine-Tuning Simulation")
st.markdown("""
Adjust the number of training steps to see how the model's accuracy improves on our e‑commerce intent dataset.
""")

# Simulated data: baseline accuracy and fine-tuning curve
baseline_acc = 0.82  # from Hour 2 (TF-IDF + LR)
steps = list(range(0, 101, 10))
# Sigmoid-like improvement
accuracies = baseline_acc + (0.15 * (1 - np.exp(-np.array(steps)/30)))
accuracies = np.clip(accuracies, baseline_acc, 0.97)

step_idx = st.slider("Training steps (epochs)", 0, 100, 30, 5)
current_acc = accuracies[steps.index(step_idx) if step_idx in steps else steps[min(range(len(steps)), key=lambda i: abs(steps[i]-step_idx))]]

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Accuracy", f"{baseline_acc:.2%}")
col2.metric("Fine-Tuned Accuracy", f"{current_acc:.2%}", delta=f"{current_acc - baseline_acc:.2%}")
col3.metric("Improvement", f"{(current_acc - baseline_acc)/baseline_acc*100:.1f}%")

# Plot the improvement curve
chart_data = pd.DataFrame({"Steps": steps, "Accuracy": accuracies})
st.line_chart(chart_data.set_index("Steps"))

st.caption("Fine-tuning adapts the pretrained model to our domain, boosting accuracy significantly.")

st.divider()

# ---------------- CODE SNIPPET (HUGGING FACE) ----------------
st.header("Fine-Tuning with Hugging Face")
st.code("""
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import datasets

# Load pretrained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize dataset
def tokenize_fn(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = datasets.load_dataset("csv", data_files="intents.csv")
dataset = dataset.map(tokenize_fn, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
""", language="python")

st.info("In practice, fine-tuning takes minutes to hours on a GPU. The code above uses the Hugging Face Trainer API.")

st.divider()

# ---------------- BASELINE VS FINE-TUNED ----------------
st.header("Baseline vs Fine-Tuned: Example Predictions")
st.markdown("""
Let's compare how the baseline (TF‑IDF + LR) and a fine-tuned transformer classify the same query.
""")

example_queries = [
    "I want to return my order",
    "Where is my package?",
    "Can I get a refund?",
    "I need to buy a new phone",
    "The product is damaged"
]
selected_query = st.selectbox("Choose a query", example_queries)

# Simulated predictions (for demo)
baseline_pred = {
    "I want to return my order": ("return_order", 0.87),
    "Where is my package?": ("track_order", 0.79),
    "Can I get a refund?": ("refund", 0.82),
    "I need to buy a new phone": ("purchase", 0.91),
    "The product is damaged": ("return_order", 0.68)
}
finetuned_pred = {
    "I want to return my order": ("return_order", 0.96),
    "Where is my package?": ("track_order", 0.94),
    "Can I get a refund?": ("refund", 0.97),
    "I need to buy a new phone": ("purchase", 0.98),
    "The product is damaged": ("return_order", 0.89)
}

b_intent, b_conf = baseline_pred[selected_query]
f_intent, f_conf = finetuned_pred[selected_query]

col1, col2 = st.columns(2)
col1.markdown("**Baseline (TF‑IDF + LR)**")
col1.write(f"Intent: **{b_intent}**")
col1.write(f"Confidence: {b_conf:.2%}")

col2.markdown("**Fine-Tuned Transformer**")
col2.write(f"Intent: **{f_intent}**")
col2.write(f"Confidence: {f_conf:.2%}")

if f_conf > b_conf:
    st.success("The fine-tuned model is more confident and often more accurate.")
else:
    st.info("Sometimes the baseline may be similar, but fine-tuning usually wins on complex queries.")

st.divider()

# ---------------- TRAINING PIPELINE ----------------
st.header("Training Pipeline")
st.markdown("""
A complete fine-tuning pipeline includes data preparation, tokenization, training, evaluation, and model export.
""")

def pipeline():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("data", "Labeled Queries", shape="box")
    dot.node("token", "Tokenization", shape="box")
    dot.node("model", "Pretrained Model", shape="box")
    dot.node("train", "Training Loop", shape="box")
    dot.node("eval", "Evaluation", shape="box")
    dot.node("save", "Saved Model", shape="box")
    dot.edge("data", "token")
    dot.edge("token", "train")
    dot.edge("model", "train")
    dot.edge("train", "eval")
    dot.edge("eval", "save")
    return dot

st.graphviz_chart(pipeline())

st.divider()

# ---------------- WHY FINE-TUNING WORKS BETTER ----------------
st.header("Why Fine-Tuning Works Better")
st.markdown("""
- **Contextual understanding:** Captures word meaning based on surrounding words.
- **Domain adaptation:** Adjusts general knowledge to e‑commerce language.
- **No manual features:** End‑to‑end learning.
- **Transfer learning:** Leverages knowledge from massive pretraining.
""")

st.divider()

# ---------------- INDUSTRY WORKFLOW ----------------
st.header("Industry Workflow with Fine-Tuning")
st.markdown("""
In production, fine-tuned models are deployed via APIs (Hour 4) and containerized (Hour 5).
""")

def industry():
    dot = Digraph()
    dot.attr(rankdir="LR")
    dot.node("data", "Customer Queries", shape="box")
    dot.node("train", "Fine-Tuning", shape="box")
    dot.node("model", "Model Artifact", shape="box")
    dot.node("api", "FastAPI", shape="box")
    dot.node("prod", "Production", shape="box")
    dot.edge("data", "train")
    dot.edge("train", "model")
    dot.edge("model", "api")
    dot.edge("api", "prod")
    return dot

st.graphviz_chart(industry())

st.divider()

# ---------------- KEY TAKEAWAYS ----------------
st.header("Key Takeaways")
st.markdown("""
- Transformers use attention to understand context.
- Pretrained models provide a strong foundation.
- Fine-tuning adapts them to specific tasks with minimal data.
- Fine-tuned models significantly outperform traditional baselines.
- Hugging Face makes fine-tuning accessible.
""")

st.success("Next: Hour 4 — API Development")