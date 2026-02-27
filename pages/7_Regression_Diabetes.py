import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Regression Pipeline Workshop", layout="wide")

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
st.title("ML Pipeline Workshop — Regression (Diabetes Dataset)")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
<strong>Duration:</strong> 60 minutes | <strong>Level:</strong> Intermediate | <strong>Path:</strong> ML Engineering
</div>
""", unsafe_allow_html=True)

# ---------------- PROGRESS TRACKER ----------------
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

steps = [
    "Data Exploration",
    "Feature Analysis", 
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
    - [MLOps Basics](https://ml-ops.org)
    """)
    
    if st.button("Reset Progress"):
        st.session_state.current_step = 0
        st.rerun()

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    df = pd.concat([X, y], axis=1)
    return X, y, df, data.DESCR

X, y, df, dataset_description = load_data()

# ---------------- STEP 1: DATASET OVERVIEW ----------------
with st.container():
    st.header("1. Dataset Overview")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Understand the structure and characteristics of the diabetes dataset
    <br>
    <strong>Key Concepts:</strong> Data shape, target distribution, feature types
    </div>
    """, unsafe_allow_html=True)
    
    # Time estimate
    st.caption("Estimated time: 8 minutes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Shape:** {df.shape[0]} samples, {df.shape[1]} features")
        st.write("**Features:**", ", ".join(X.columns))
        st.write("**Target:** Disease progression (quantitative measure)")
        
        # Quick dataset stats
        st.info(f"Target range: {df['target'].min():.1f} to {df['target'].max():.1f}")
        
        st.dataframe(df.head(), use_container_width=True)
    
    with col2:
        st.subheader("Target Distribution")
        fig = px.histogram(
            df,
            x="target",
            height=300,
            title="Distribution of Disease Progression",
            marginal="box"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick comprehension check
        with st.expander("Quick Check"):
            st.write("What is the approximate range of the target variable?")
            if st.checkbox("Show answer"):
                st.success(f"The target ranges from {df['target'].min():.1f} to {df['target'].max():.1f}")

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 0)

# ---------------- STEP 2: FEATURE VISUALIZATION ----------------
with st.container():
    st.header("2. Feature Analysis")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Understand relationships between features and target
    <br>
    <strong>Key Concepts:</strong> Correlation, linear relationships, feature importance intuition
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 10 minutes")
    
    # Feature selection with explanation
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Select a feature to analyze:**")
        feature = st.selectbox("", X.columns, key="feature_select", label_visibility="collapsed")
        
        # Show correlation
        correlation = df[feature].corr(df['target'])
        st.metric("Correlation with target", f"{correlation:.3f}")
        
        if abs(correlation) > 0.5:
            st.success("Strong correlation detected")
        elif abs(correlation) > 0.3:
            st.info("Moderate correlation detected")
        else:
            st.warning("Weak correlation detected")
    
    with col2:
        fig = px.scatter(
            df,
            x=feature,
            y="target",
            height=350,
            title=f"{feature} vs Target (Correlation: {correlation:.3f})",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Interactive correlation matrix
    with st.expander("View Correlation Matrix"):
        corr_matrix = df.corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 1)

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- STEP 3: MODEL TRAINING ----------------
with st.container():
    st.header("3. Model Selection and Training")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Compare different regression algorithms and understand their parameters
    <br>
    <strong>Key Concepts:</strong> Model complexity, hyperparameters, bias-variance tradeoff
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 12 minutes")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model selection with explanations
        model_type = st.selectbox(
            "Select Algorithm",
            ["Linear Regression", "Random Forest"],
            help="Linear Regression: Assumes linear relationship | Random Forest: Ensemble of decision trees"
        )
        
        if model_type == "Linear Regression":
            st.info("Linear Regression assumes a linear relationship between features and target")
            model = LinearRegression()
            param_info = "No hyperparameters"
        else:
            st.info("Random Forest combines multiple decision trees for improved predictions")
            n_estimators = st.slider("Number of Trees", 10, 200, 100, 
                                     help="More trees provide more stable predictions but increase computation time")
            max_depth = st.slider("Maximum Depth", 1, 20, 5,
                                 help="Deeper trees can capture complex patterns but may overfit")
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            param_info = f"Trees: {n_estimators}, Depth: {max_depth}"
    
    # Train model
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    with col2:
        st.subheader("Training Summary")
        st.write(f"**Model:** {model_type}")
        if model_type == "Random Forest":
            st.write(f"**Parameters:** {param_info}")
        st.write(f"**Training samples:** {len(X_train)}")
        st.write(f"**Test samples:** {len(X_test)}")
        
        # Show sample predictions
        sample_results = pd.DataFrame({
            'Actual': y_test[:5].values,
            'Predicted': preds[:5].round(2)
        })
        st.write("**Sample Predictions:**")
        st.dataframe(sample_results, use_container_width=True)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 2)

# ---------------- STEP 4: EVALUATION ----------------
with st.container():
    st.header("4. Model Evaluation")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Interpret model performance metrics and visualize predictions
    <br>
    <strong>Key Concepts:</strong> RMSE, R² score, prediction residuals
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 8 minutes")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("RMSE", f"{round(rmse, 2)}", 
                 help="Root Mean Square Error: Average prediction error in original units")
    with col2:
        st.metric("R² Score", f"{round(r2, 3)}",
                 help="Coefficient of determination: Proportion of variance explained by the model")
    with col3:
        accuracy_range = (1 - rmse/df['target'].std()) * 100
        st.metric("Accuracy Range", f"{accuracy_range:.1f}%",
                 help="Approximate accuracy within one standard deviation")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            x=y_test,
            y=preds,
            labels={"x": "Actual Values", "y": "Predicted Values"},
            height=350,
            title="Predicted vs Actual"
        )
        # Add perfect prediction line
        min_val = min(y_test.min(), preds.min())
        max_val = max(y_test.max(), preds.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = y_test - preds
        fig_res = px.scatter(
            x=preds,
            y=residuals,
            labels={"x": "Predicted Values", "y": "Residuals"},
            height=350,
            title="Residuals Plot"
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res, use_container_width=True)
    
    # Model interpretation
    with st.expander("How to interpret these metrics"):
        st.markdown(f"""
        - **RMSE = {rmse:.2f}**: On average, predictions differ from actual values by ±{rmse:.2f} units
        - **R² = {r2:.3f}**: The model explains {r2*100:.1f}% of the variance in the target variable
        - **Residuals plot**: Points should be randomly scattered around zero; patterns may indicate model misspecification
        """)

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 3)

# ---------------- STEP 5: ADVANCED CONCEPTS ----------------
with st.container():
    st.header("5. Advanced Concepts")
    
    tab1, tab2 = st.tabs(["Feature Importance", "Overfitting Demonstration"])
    
    with tab1:
        st.markdown("""
        <div class='section'>
        <strong>Learning Objective:</strong> Understand which features most influence model predictions
        </div>
        """, unsafe_allow_html=True)
        
        if model_type == "Random Forest":
            importance = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=True)
            
            fig = px.bar(
                importance,
                x="Importance",
                y="Feature",
                orientation="h",
                height=400,
                title="Feature Importance (Random Forest)"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance explanation
            top_feature = importance.iloc[-1]["Feature"]
            st.info(f"**Insight:** '{top_feature}' has the highest importance for prediction")
        else:
            st.warning("Feature importance is only available for Random Forest models. Switch to Random Forest to see this visualization.")
    
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
                m = RandomForestRegressor(max_depth=d, n_estimators=50, random_state=42)
                m.fit(X_train, y_train)
                train_scores.append(r2_score(y_train, m.predict(X_train)))
                test_scores.append(r2_score(y_test, m.predict(X_test)))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=depths, y=train_scores, name="Training Score", mode='lines+markers'))
        fig.add_trace(go.Scatter(x=depths, y=test_scores, name="Test Score", mode='lines+markers'))
        
        fig.update_layout(
            title="Overfitting Demonstration: Effect of Tree Depth",
            xaxis_title="Maximum Depth",
            yaxis_title="R² Score",
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal depth
        optimal_depth = depths[np.argmax(test_scores)]
        st.success(f"**Optimal depth:** {optimal_depth} (where test score is maximized)")

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 4)

# ---------------- STEP 6: PREDICTION DEMO ----------------
with st.container():
    st.header("6. Prediction Demo")
    st.markdown("""
    <div class='section'>
    <strong>Learning Objective:</strong> Apply the trained model to make predictions on new data
    </div>
    """, unsafe_allow_html=True)
    
    st.caption("Estimated time: 7 minutes")
    
    st.markdown("**Enter values for each feature:**")
    
    input_data = {}
    cols = st.columns(5)
    
    for i, feature in enumerate(X.columns):
        col = cols[i % 5]
        # Show feature stats as help text
        mean_val = df[feature].mean()
        std_val = df[feature].std()
        
        input_data[feature] = col.number_input(
            f"{feature}",
            value=float(mean_val),
            step=0.1,
            format="%.2f",
            help=f"Mean: {mean_val:.2f} (±{std_val:.2f})"
        )
    
    input_df = pd.DataFrame([input_data])
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("Predict Disease Progression", type="primary", use_container_width=True):
            prediction = model.predict(input_df)[0]
            
            # Show prediction with confidence interval
            if model_type == "Random Forest":
                std_pred = np.std([tree.predict(input_df)[0] for tree in model.estimators_])
            else:
                std_pred = rmse
            
            st.success(f"### Predicted progression: {prediction:.2f}")
            st.info(f"95% confidence interval: [{prediction-1.96*std_pred:.2f}, {prediction+1.96*std_pred:.2f}]")
            
            # Compare to dataset average
            avg_target = df['target'].mean()
            if prediction > avg_target:
                st.warning(f"Above average (dataset average: {avg_target:.2f})")
            else:
                st.success(f"Below average (dataset average: {avg_target:.2f})")

# Update progress
st.session_state.current_step = max(st.session_state.current_step, 5)

# ---------------- WORKSHOP SUMMARY ----------------
with st.expander("Workshop Summary and Key Takeaways"):
    st.markdown("""
    ### What You Have Learned
    
    1. **Data Exploration** - Understanding dataset structure and relationships between variables
    2. **Feature Analysis** - Identifying important features and correlations with the target
    3. **Model Selection** - Comparing different algorithms and understanding their parameters
    4. **Evaluation** - Interpreting metrics like RMSE and R² for model performance
    5. **Advanced Concepts** - Feature importance analysis and overfitting detection
    6. **Practical Application** - Making predictions on new data with confidence intervals
    
    ### Next Steps
    
    - Experiment with different hyperparameter combinations
    - Try other regression algorithms (SVR, Gradient Boosting)
    - Apply these concepts to your own datasets
    - Learn about model deployment with FastAPI
    
    ### Resources
    
    - [Scikit-learn Documentation](https://scikit-learn.org)
    - [MLOps Fundamentals](https://ml-ops.org)
    - [Workshop Repository](https://github.com/VijayDwivedi-ml/Modern-ML-Engineer-workshop/tree/main)
    """)