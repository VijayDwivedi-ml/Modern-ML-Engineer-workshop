import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Classification Pipeline Workshop", layout="wide")

# ---------------- CSS (PRESERVED) ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}
.section {
    background-color: #f8f9fb;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- WORKSHOP HEADER ----------------
st.title("ML Pipeline Workshop — Classification (Digits Dataset)")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
<strong>Duration:</strong> 60 minutes | <strong>Level:</strong> Intermediate | <strong>Path:</strong> ML Engineering
</div>
""", unsafe_allow_html=True)

# ---------------- PROGRESS TRACKER ----------------
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

steps = [
    "Dataset Exploration",
    "Dimensionality Reduction", 
    "Model Selection",
    "Evaluation",
    "Advanced Concepts",
    "Prediction Demo"
]

# Sidebar for workshop navigation
with st.sidebar:
    st.header("Workshop Contents")
    st.markdown("---")
    for i, step in enumerate(steps):
        if i <= st.session_state.current_step:
            st.markdown(f"**Step {i+1}:** {step} (completed)")
        else:
            st.markdown(f"**Step {i+1}:** {step}")
    
    st.markdown("---")
    st.markdown("""
    **Resources**
    - [Scikit-learn Docs](https://scikit-learn.org)
    - [Digits Dataset Info](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
    """)
    
    if st.button("Reset Progress"):
        st.session_state.current_step = 0
        st.rerun()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    digits = load_digits()
    X = digits.data
    y = digits.target
    return X, y, digits

X, y, digits = load_data()

# ---------------- STEP 1: DATASET EXPLORATION ----------------
with st.container():
    st.header("1. Dataset Exploration")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Understand the structure of image data for classification
    <br>
    <strong>Key Concepts:</strong> Image dimensions, pixel values, class distribution
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 10 minutes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Shape:** {X.shape[0]} images, {X.shape[1]} features (pixels)")
        st.write(f"**Number of classes:** {len(np.unique(y))} digits (0-9)")
        st.write(f"**Image dimensions:** 8 × 8 pixels (64 features total)")
        st.write(f"**Pixel range:** {X.min():.0f} to {X.max():.0f}")
        
        # Class distribution
        class_counts = pd.Series(y).value_counts().sort_index()
        st.write("**Class distribution:**")
        st.dataframe(class_counts.to_frame("Count"), use_container_width=True)
    
    with col2:
        st.subheader("Sample Images")
        
        # Image browser
        index = st.slider("Select image index", 0, len(X)-1, 0, key="image_browser")
        
        image = X[index].reshape(8, 8)
        
        fig = px.imshow(
            image,
            color_continuous_scale="gray",
            height=300,
            title=f"Digit: {y[index]}"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show pixel values
        with st.expander("View pixel values"):
            pixel_df = pd.DataFrame(image, columns=[f"col_{i}" for i in range(8)])
            pixel_df.index = [f"row_{i}" for i in range(8)]
            st.dataframe(pixel_df, use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 0)

# ---------------- STEP 2: DIMENSIONALITY REDUCTION ----------------
with st.container():
    st.header("2. Dimensionality Reduction with PCA")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Visualize high-dimensional data in 2D space
    <br>
    <strong>Key Concepts:</strong> Principal Component Analysis, variance explained, data visualization
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 8 minutes")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("PCA Configuration")
        
        n_components = st.slider("Number of components", 2, 10, 2, 
                                 help="More components capture more variance but harder to visualize")
        
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        st.metric("Variance explained (2 components)", f"{cumulative_var[1]:.1%}")
        st.metric("Total variance explained", f"{cumulative_var[-1]:.1%}")
        
        # Show explained variance for each component
        var_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Variance %": explained_var * 100
        })
        st.dataframe(var_df, use_container_width=True)
    
    with col2:
        # 2D visualization
        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Digit": y.astype(str)
        })
        
        fig = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="Digit",
            height=400,
            title="Digits Visualization in 2D Space (PCA)",
            opacity=0.7
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Explained variance plot
    with st.expander("View Explained Variance by Component"):
        fig_var = px.bar(
            x=[f"PC{i+1}" for i in range(n_components)],
            y=explained_var,
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
            title="Explained Variance by Component"
        )
        st.plotly_chart(fig_var, use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 1)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- STEP 3: MODEL SELECTION AND TRAINING ----------------
with st.container():
    st.header("3. Model Selection and Training")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Compare different classification algorithms
    <br>
    <strong>Key Concepts:</strong> Multiclass classification, model complexity, hyperparameters
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 12 minutes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        model_type = st.selectbox(
            "Select Algorithm",
            ["Logistic Regression", "Random Forest"],
            help="Logistic Regression: Linear classifier | Random Forest: Ensemble of decision trees"
        )
        
        if model_type == "Logistic Regression":
            st.info("Logistic Regression uses a linear decision boundary for classification")
            
            max_iter = st.slider("Max Iterations", 500, 5000, 2000, step=500,
                                help="Maximum number of iterations for convergence")
            
            C_value = st.select_slider("Regularization (C)", 
                                       options=[0.01, 0.1, 1.0, 10.0, 100.0],
                                       value=1.0,
                                       help="Inverse of regularization strength (smaller = stronger regularization)")
            
            model = LogisticRegression(max_iter=max_iter, C=C_value, random_state=42)
            param_info = f"Iterations: {max_iter}, C: {C_value}"
            
        else:
            st.info("Random Forest combines multiple decision trees for robust classification")
            
            n_estimators = st.slider("Number of Trees", 10, 200, 100,
                                     help="More trees provide more stable predictions")
            max_depth = st.slider("Maximum Depth", 1, 20, 5,
                                 help="Deeper trees can capture complex patterns but may overfit")
            min_samples_split = st.slider("Min Samples Split", 2, 20, 2,
                                         help="Minimum samples required to split a node")
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            param_info = f"Trees: {n_estimators}, Depth: {max_depth}, Min Split: {min_samples_split}"
    
    # Train model
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    with col2:
        st.subheader("Training Summary")
        st.write(f"**Model:** {model_type}")
        st.write(f"**Parameters:** {param_info}")
        st.write(f"**Training samples:** {len(X_train)}")
        st.write(f"**Test samples:** {len(X_test)}")
        
        # Training accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, preds)
        
        st.metric("Training Accuracy", f"{train_acc:.3f}")
        st.metric("Test Accuracy", f"{test_acc:.3f}")
        
        # Sample predictions
        sample_indices = np.random.choice(len(X_test), 5, replace=False)
        sample_results = pd.DataFrame({
            'Actual': y_test[sample_indices],
            'Predicted': preds[sample_indices],
            'Correct': y_test[sample_indices] == preds[sample_indices]
        })
        st.write("**Sample Predictions:**")
        st.dataframe(sample_results, use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 2)

# ---------------- STEP 4: MODEL EVALUATION ----------------
with st.container():
    st.header("4. Model Evaluation")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Interpret classification performance metrics
    <br>
    <strong>Key Concepts:</strong> Accuracy, confusion matrix, per-class performance
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 8 minutes")
    
    acc = accuracy_score(y_test, preds)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Accuracy", f"{acc:.4f}",
                 help="Proportion of correct predictions")
    
    with col2:
        # Per-class accuracy
        correct_counts = pd.Series(preds == y_test).groupby(y_test).mean()
        avg_per_class = correct_counts.mean()
        st.metric("Avg Per-Class Accuracy", f"{avg_per_class:.4f}",
                 help="Average accuracy across all classes")
    
    with col3:
        # Error count
        error_count = (preds != y_test).sum()
        st.metric("Misclassifications", f"{error_count}/{len(y_test)}",
                 help=f"{(error_count/len(y_test)*100):.1f}% error rate")
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    cm = confusion_matrix(y_test, preds)
    
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        height=400,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[str(i) for i in range(10)],
        y=[str(i) for i in range(10)]
    )
    fig_cm.update_layout(title="Confusion Matrix (Predicted vs Actual)")
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Per-class metrics
    with st.expander("View Per-Class Performance"):
        report = classification_report(y_test, preds, output_dict=True)
        report_df = pd.DataFrame(report).T
        st.dataframe(report_df.round(3), use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 3)

# ---------------- STEP 5: ADVANCED CONCEPTS ----------------
with st.container():
    st.header("5. Advanced Concepts")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Overfitting Demonstration"])
    
    with tab1:
        st.markdown("""
        <div class='section'>
        <strong>Learning Objective:</strong> Understand which pixels are most important for classification
        </div>
        """, unsafe_allow_html=True)
        
        if model_type == "Random Forest":
            # Feature importance visualization
            importance = model.feature_importances_
            
            # Reshape to image for visualization
            importance_img = importance.reshape(8, 8)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_imp = px.imshow(
                    importance_img,
                    color_continuous_scale="viridis",
                    height=350,
                    title="Pixel Importance Heatmap"
                )
                fig_imp.update_layout(coloraxis_showscale=True)
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col2:
                # Top features
                top_indices = np.argsort(importance)[-10:][::-1]
                top_features = pd.DataFrame({
                    'Pixel': [f"pixel_{i}" for i in top_indices],
                    'Importance': importance[top_indices]
                })
                
                fig_bar = px.bar(
                    top_features,
                    x='Importance',
                    y='Pixel',
                    orientation='h',
                    title="Top 10 Most Important Pixels"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.info("**Insight:** Brighter areas in the heatmap show pixels that are most discriminative for digit classification")
            
        else:
            st.warning("Feature importance is only available for Random Forest models. Switch to Random Forest to see which pixels matter most.")
    
    with tab2:
        st.markdown("""
        <div class='section'>
        <strong>Learning Objective:</strong> Visualize how model complexity affects overfitting
        </div>
        """, unsafe_allow_html=True)
        
        depths = list(range(1, 15))
        train_scores = []
        test_scores = []
        
        with st.spinner("Calculating overfitting curves..."):
            for d in depths:
                m = RandomForestClassifier(max_depth=d, n_estimators=50, random_state=42)
                m.fit(X_train, y_train)
                train_scores.append(accuracy_score(y_train, m.predict(X_train)))
                test_scores.append(accuracy_score(y_test, m.predict(X_test)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=depths, y=train_scores, name="Training Accuracy", mode='lines+markers'))
        fig.add_trace(go.Scatter(x=depths, y=test_scores, name="Test Accuracy", mode='lines+markers'))
        
        fig.update_layout(
            title="Overfitting Demonstration: Effect of Tree Depth",
            xaxis_title="Maximum Depth",
            yaxis_title="Accuracy",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal depth
        optimal_depth = depths[np.argmax(test_scores)]
        st.success(f"**Optimal depth:** {optimal_depth} (where test accuracy is maximized)")
        
        st.markdown("""
        **Observations:**
        - **Underfitting (low depth):** Both training and test accuracy are low
        - **Optimal range:** Test accuracy peaks while training accuracy continues to improve
        - **Overfitting (high depth):** Training accuracy near 100%, but test accuracy drops
        """)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 4)

# ---------------- STEP 6: PREDICTION DEMO ----------------
with st.container():
    st.header("6. Prediction Demo")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Apply the trained model to classify new digit images
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 7 minutes")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.subheader("Select Test Sample")
        sample_idx = st.slider("Choose a test image", 0, len(X_test)-1, 0, key="prediction_slider")
        
        sample_data = X_test[sample_idx].reshape(8, 8)
        actual_label = y_test[sample_idx]
        
        fig = px.imshow(
            sample_data,
            color_continuous_scale="gray",
            height=250,
            title=f"Input Image (Actual: {actual_label})"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Prediction")
        
        if st.button("Classify Image", type="primary", use_container_width=True):
            prediction = model.predict([X_test[sample_idx]])[0]
            
            # Get prediction probabilities if available
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([X_test[sample_idx]])[0]
                confidence = probabilities[prediction]
                
                st.success(f"### Predicted: {prediction}")
                st.info(f"Confidence: {confidence:.1%}")
                
                if prediction == actual_label:
                    st.success("✅ Correct prediction")
                else:
                    st.error(f"❌ Incorrect (actual: {actual_label})")
            else:
                st.success(f"### Predicted: {prediction}")
                
                if prediction == actual_label:
                    st.success("✅ Correct prediction")
                else:
                    st.error(f"❌ Incorrect (actual: {actual_label})")
    
    with col3:
        st.subheader("Probability Distribution")
        
        if hasattr(model, "predict_proba"):
            if st.button("Show Probabilities", use_container_width=True):
                probabilities = model.predict_proba([X_test[sample_idx]])[0]
                
                prob_df = pd.DataFrame({
                    'Digit': [str(i) for i in range(10)],
                    'Probability': probabilities
                })
                
                fig_prob = px.bar(
                    prob_df,
                    x='Digit',
                    y='Probability',
                    title="Class Probabilities"
                )
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.info("Probability estimates not available for this model")

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 5)

# ---------------- WORKSHOP SUMMARY ----------------
with st.expander("Workshop Summary and Key Takeaways"):
    st.markdown("""
    ### What You Have Learned
    
    1. **Dataset Exploration** - Understanding image data structure and visualizing samples
    2. **Dimensionality Reduction** - Using PCA to visualize high-dimensional data in 2D
    3. **Model Selection** - Comparing Logistic Regression and Random Forest for classification
    4. **Evaluation** - Interpreting accuracy, confusion matrices, and per-class performance
    5. **Advanced Concepts** - Feature importance analysis and overfitting detection
    6. **Practical Application** - Making predictions on new images with confidence scores
    
    ### Key Insights
    
    - The digits dataset contains 8×8 pixel images (64 features) of handwritten digits
    - PCA helps visualize that digits form natural clusters in lower-dimensional space
    - Random Forests can show which pixels are most important for classification
    - Model complexity must be balanced to avoid overfitting
    
    ### Next Steps
    
    - Experiment with different hyperparameters
    - Try other classifiers (SVM, Neural Networks)
    - Apply these concepts to your own image classification problems
    - Learn about deep learning for more complex image tasks
    
    ### Resources
    
    - [Scikit-learn Documentation](https://scikit-learn.org)
    - [Digits Dataset Source](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)
    - [Workshop Repository](https://github.com/VijayDwivedi-ml/Modern-ML-Engineer-workshop/tree/main)
    """)