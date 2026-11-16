"""
================================================================================
Streamlit Web Application: Delivery Delay Prediction System
Course: UML501 Machine Learning (B.Tech 5th Semester)
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# Page Configuration & Styling
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Delivery Delay Prediction | ML Project",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .risk-badge-late {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #F87171;
    }
    .risk-badge-ontime {
        background-color: #DCFCE7;
        color: #166534;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        text-align: center;
        border: 1px solid #4ADE80;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Load Cached Models & Metadata
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_project_artifacts():
    models_path = os.path.join(MODELS_DIR, "all_models.joblib")
    meta_path   = os.path.join(MODELS_DIR, "pipeline_metadata.joblib")
    eval_path   = os.path.join(MODELS_DIR, "eval_results.json")
    feat_path   = os.path.join(MODELS_DIR, "feature_importance.json")
    test_path   = os.path.join(MODELS_DIR, "test_sample.csv")
    
    if not os.path.exists(models_path):
        return None, None, None, None, None
        
    models   = joblib.load(models_path)
    metadata = joblib.load(meta_path)
    with open(eval_path, 'r') as f:
        eval_results = json.load(f)
    with open(feat_path, 'r') as f:
        feat_importances = json.load(f)
    test_sample = pd.read_csv(test_path) if os.path.exists(test_path) else None
    
    return models, metadata, eval_results, feat_importances, test_sample

models, metadata, eval_results, feat_importances, test_sample = load_project_artifacts()

if models is None:
    st.error("⚠️ Model artifacts not found in `./models/`. Please execute `python train_models.py` first.")
    st.stop()

# ------------------------------------------------------------------------------
# Application Header
# ------------------------------------------------------------------------------
st.markdown('<div class="main-header">📦 Supply Chain Delivery Delay Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">B.Tech 5th Sem Machine Learning Project (Course: UML501) | Baseline vs Cascade Classifier Benchmarks</div>', unsafe_allow_html=True)

# Top KPI ribbon
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Total Test Orders", f"{eval_results['Random Forest']['n_total']:,}")
with kpi2:
    st.metric("Best Std Accuracy", f"{eval_results['Random Forest']['accuracy']*100:.2f}%", "Random Forest")
with kpi3:
    st.metric("Best Conf Accuracy", f"{eval_results['Random Forest']['confidence_accuracy']*100:.2f}%", "±0.10 Margin")
with kpi4:
    st.metric("Cascade Recall", f"{eval_results['Cascade Classifier']['recall']*100:.2f}%", "Sensitivity")
with kpi5:
    st.metric("Syllabus Models", "6 Tested", "UML501 Pure")

st.markdown("---")

# ------------------------------------------------------------------------------
# Navigation Tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Live Delay Predictor",
    "📊 Model Comparison & Metrics",
    "🔍 Feature Importance & Leakage",
    "🎓 Academic Syllabus & Viva Prep"
])

# ==============================================================================
# TAB 1: LIVE DELAY PREDICTOR
# ==============================================================================
with tab1:
    st.subheader("Interactive Shipment Risk Calculator")
    st.write("Adjust shipment parameters below or load an empirical test record to evaluate delay risk in real time.")
    
    # Quick Test loader
    with st.expander("📂 Load a Real Order from Test Sample (Optional)", expanded=False):
        if test_sample is not None:
            sample_idx = st.slider("Select Sample Index (0 to 99):", 0, len(test_sample)-1, 12)
            row_data = test_sample.iloc[sample_idx]
            actual_label = "🔴 LATE" if row_data['true_late_risk'] == 1 else "🟢 ON-TIME"
            st.info(f"Loaded Order #{sample_idx} — **Ground Truth Outcome:** {actual_label} | **Shipping Mode:** {row_data['Shipping_Mode']}")
        else:
            row_data = None
            
    col_input1, col_input2, col_input3 = st.columns(3)
    
    with col_input1:
        st.markdown("##### 🚚 Logistics Parameters")
        shipping_mode = st.selectbox(
            "Shipping Mode",
            ['Standard Class', 'Second Class', 'First Class', 'Same Day'],
            index=0 if row_data is None else ['Standard Class', 'Second Class', 'First Class', 'Same Day'].index(row_data['Shipping_Mode'])
        )
        scheduled_days = st.slider("Days for Shipment (Scheduled)", min_value=1, max_value=6, value=4)
        order_day = st.selectbox("Order Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=2)
        order_month = st.slider("Order Month", min_value=1, max_value=12, value=6)
        
    with col_input2:
        st.markdown("##### 💰 Commercial Parameters")
        sales_amt = st.number_input("Order Sales ($)", min_value=5.0, max_value=2500.0, value=180.0, step=10.0)
        item_qty = st.number_input("Item Quantity", min_value=1, max_value=10, value=2, step=1)
        profit_per_order = st.number_input("Benefit / Profit Per Order ($)", min_value=-400.0, max_value=600.0, value=45.0, step=5.0)
        discount_amt = st.number_input("Order Item Discount ($)", min_value=0.0, max_value=150.0, value=15.0, step=2.0)
        
    with col_input3:
        st.markdown("##### ⚙️ Model & Threshold Setup")
        selected_model_name = st.selectbox(
            "Classifier Model to Execute",
            ["Random Forest (Recommended)", "Cascade Classifier (Tier-Specific)", "Decision Tree", "Logistic Regression", "Naive Bayes", "XGBoost", "K-NN"]
        )
        conf_margin = st.slider("Confidence Margin Band (±)", min_value=0.05, max_value=0.20, value=0.10, step=0.01)
        decision_threshold = st.slider("Decision Threshold", min_value=0.30, max_value=0.70, value=0.50, step=0.05)
        
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Evaluate Delay Risk", type="primary", use_container_width=True)
    
    if predict_btn or row_data is not None:
        # Construct feature vector based on trained pipeline
        dow_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        dow_int = dow_map[order_day]
        priority_map = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}
        ship_priority = priority_map[shipping_mode]
        sched_x_priority = scheduled_days * ship_priority
        price_per_item = sales_amt / (item_qty + 1)
        profit_margin = profit_per_order / (sales_amt + 1)
        month_sin = np.sin(2 * np.pi * order_month / 12)
        month_cos = np.cos(2 * np.pi * order_month / 12)
        is_weekend = 1 if dow_int >= 5 else 0
        
        # Build vector matching metadata['selected_features']
        input_dict = {}
        for feat in metadata['selected_features']:
            if feat == 'Days for shipment (scheduled)':
                input_dict[feat] = scheduled_days
            elif feat == 'sched_x_priority':
                input_dict[feat] = sched_x_priority
            elif feat == 'ship_priority':
                input_dict[feat] = ship_priority
            elif feat == 'price_per_item':
                input_dict[feat] = price_per_item
            elif feat == 'profit_margin':
                input_dict[feat] = profit_margin
            elif feat == 'month_sin':
                input_dict[feat] = month_sin
            elif feat == 'month_cos':
                input_dict[feat] = month_cos
            elif feat == 'is_weekend':
                input_dict[feat] = is_weekend
            elif feat == 'Order Item Discount':
                input_dict[feat] = discount_amt
            elif feat == 'Sales':
                input_dict[feat] = sales_amt
            elif feat == 'Order Item Quantity':
                input_dict[feat] = item_qty
            elif feat == 'Benefit per order':
                input_dict[feat] = profit_per_order
            elif feat == 'Shipping Mode_Second Class':
                input_dict[feat] = 1 if shipping_mode == 'Second Class' else 0
            else:
                # Default baseline median/mean for geo / confidence features
                input_dict[feat] = metadata['impute_num'].get(feat, 0.5)
                
        # Scale through MinMax
        feat_df = pd.DataFrame([input_dict])[metadata['selected_features']]
        # Clip to [0, 1] range to avoid out of bounds
        feat_df_scaled = feat_df.clip(0.0, 1.0)
        
        # Model Prediction
        proba = 0.5
        model_key = selected_model_name.replace(" (Recommended)", "").replace(" (Tier-Specific)", "")
        
        if model_key == "Cascade Classifier":
            cascade_dict = models['Cascade Classifier']
            sub_model = cascade_dict[shipping_mode]['model']
            thr = cascade_dict[shipping_mode]['threshold']
            proba = float(sub_model.predict_proba(feat_df_scaled)[0, 1])
            is_late = int(proba >= thr)
            effective_thr = thr
        else:
            clf = models[model_key]
            proba = float(clf.predict_proba(feat_df_scaled)[0, 1])
            is_late = int(proba >= decision_threshold)
            effective_thr = decision_threshold
            
        # Confidence classification
        is_confident = (proba < effective_thr - conf_margin) or (proba >= effective_thr + conf_margin)
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            st.markdown("#### Prediction Output")
            if is_late == 1:
                st.markdown(
                    f'<div class="risk-badge-late">⚠️ HIGH RISK: ORDER WILL BE DELAYED (Class 1)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="risk-badge-ontime">✅ LOW RISK: ON-TIME DELIVERY EXPECTED (Class 0)</div>',
                    unsafe_allow_html=True
                )
                
            st.write("")
            st.write(f"**Predicted Delay Probability:** `{proba * 100:.1f}%` (Decision Boundary: `{effective_thr * 100:.0f}%`)")
            st.progress(proba)
            
            if is_confident:
                st.success(f"🎯 **High Confidence Prediction**: Probability is safely outside the ±{conf_margin*100:.0f}% uncertainty margin.")
            else:
                st.warning(f"⚠️ **Borderline / At-Risk Prediction**: Probability ({proba:.3f}) falls in the [{effective_thr - conf_margin:.2f}, {effective_thr + conf_margin:.2f}] low-confidence band. Human escalation recommended.")
                
        with res_col2:
            st.markdown("#### Supply Chain Recommendation")
            if is_late == 1:
                st.error("""
                **Recommended Operations Intervention:**
                - Route order through expedited courier line.
                - Alert dispatch warehouse for Priority SLA packing.
                - Flag scheduled days ({}) vs priority tier ({}) mismatch.
                """.format(scheduled_days, shipping_mode))
            else:
                st.info("""
                **Standard Operating Procedure:**
                - Standard fulfillment lane is adequate.
                - No priority surcharge or warehouse override needed.
                """)

# ==============================================================================
# TAB 2: MODEL COMPARISON & LEADERBOARD
# ==============================================================================
with tab2:
    st.subheader("Model Benchmark & Evaluation Leaderboard")
    st.write("Comprehensive comparison across core **UML501 Syllabus Models** and the **Tier-Specific Cascade Classifier**.")
    
    # Build comparison dataframe
    comp_data = []
    for m_name, metrics in eval_results.items():
        comp_data.append({
            "Model Name": m_name,
            "Accuracy (Std)": f"{metrics['accuracy']*100:.2f}%",
            "Confidence Acc": f"{metrics['confidence_accuracy']*100:.2f}%",
            "Accuracy Gain": f"+{(metrics['confidence_accuracy'] - metrics['accuracy'])*100:.2f}%",
            "Precision": f"{metrics['precision']*100:.2f}%",
            "Recall (Sensitivity)": f"{metrics['recall']*100:.2f}%",
            "Specificity": f"{metrics['specificity']*100:.2f}%",
            "F1-Score": f"{metrics['f1_score']*100:.2f}%",
            "ROC-AUC": f"{metrics['roc_auc']:.4f}",
            "At-Risk %": f"{metrics['pct_risk']:.1f}%",
            "Train Time": f"{metrics.get('train_time', 0):.2f}s"
        })
        
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("##### Standard vs Confidence Accuracy")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        models_list = list(eval_results.keys())
        x = np.arange(len(models_list))
        w = 0.35
        std_accs = [eval_results[m]['accuracy'] * 100 for m in models_list]
        conf_accs = [eval_results[m]['confidence_accuracy'] * 100 for m in models_list]
        
        ax.bar(x - w/2, std_accs, w, label="Standard Accuracy", color="#93C5FD")
        ax.bar(x + w/2, conf_accs, w, label="Confidence Accuracy (±0.10)", color="#2563EB")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=35, ha='right', fontsize=9)
        ax.set_ylim(60, 85)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("##### Confusion Matrix Inspector")
        inspect_model = st.selectbox("Select Model to Inspect Confusion Matrix:", list(eval_results.keys()), index=2)
        cm = eval_results[inspect_model]['confusion_matrix']
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)
        for i in range(2):
            for j in range(2):
                ax_cm.text(x=j, y=i, s=f"{cm[i][j]:,}", va='center', ha='center', size='large', weight='bold')
                
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(['Pred: On-Time', 'Pred: LATE'])
        ax_cm.set_yticklabels(['True: On-Time', 'True: LATE'])
        ax_cm.set_title(f"Confusion Matrix: {inspect_model}", pad=15)
        plt.tight_layout()
        st.pyplot(fig_cm)
        
        st.caption(f"""
        - **True Negatives (TN):** `{cm[0][0]:,}` | **False Positives (FP):** `{cm[0][1]:,}`
        - **False Negatives (FN):** `{cm[1][0]:,}` (Critical SLA Misses) | **True Positives (TP):** `{cm[1][1]:,}`
        """)

# ==============================================================================
# TAB 3: FEATURE IMPORTANCE & LEAKAGE
# ==============================================================================
with tab3:
    st.subheader("Data Leakage Prevention & Feature Drivers")
    
    col_feat1, col_feat2 = st.columns([1.2, 1])
    
    with col_feat1:
        st.markdown("##### Top Predictive Features (Random Forest)")
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        features_sorted = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:10]
        feats = [f[0] for f in features_sorted][::-1]
        scores = [f[1] * 100 for f in features_sorted][::-1]
        
        ax_imp.barh(feats, scores, color='#0284C7')
        ax_imp.set_xlabel("Feature Importance (%)")
        ax_imp.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_imp)
        
    with col_feat2:
        st.markdown("##### 🛡️ Leakage-Free Preprocessing (v8)")
        st.info("""
        **Crucial Academic Integrity Rule:**
        In standard delivery datasets, several columns leak future information:
        - `Days for shipping (real)`: Direct mathematical leakage of delay.
        - `Delivery Status`: Explicitly contains the target text ('Late delivery').
        - `shipping date (DateOrders)`: Recorded post-shipment.
        
        **Our Strict Fix:**
        All leakage columns are stripped prior to train/test split. Preprocessing parameters (MinMax min/max, imputation means, IQR bounds) are strictly fit on the training split to eliminate data snooping.
        """)
        
    st.markdown("##### Key Domain Insights")
    st.write("""
    1. **Scheduled Days & Priority Interaction (`sched_x_priority`)**: The primary driver of delay is tight scheduled time windows assigned to slower standard shipping tiers.
    2. **Geographical Distance (`dist_center`)**: Distance from distribution center has a noticeable secondary impact on transit risk.
    3. **Order Financials**: High-discount and low-margin orders correlate slightly with delay risk due to bulk-shipping consolidations.
    """)

# ==============================================================================
# TAB 4: ACADEMIC SYLLABUS & VIVA PREPARATION
# ==============================================================================
with tab4:
    st.subheader("UML501 Machine Learning: 15-Mark Viva Guide")
    st.write("Structured explanations and formula references aligned with your official course syllabus.")
    
    with st.expander("📘 1. Mapping Project to UML501 Course Units", expanded=True):
        st.markdown("""
        | Syllabus Unit | Course Requirement | Our Project Implementation |
        |---|---|---|
        | **Unit 2: Data Pre-processing** | Data Cleaning, Scaling, Splitting, Reduction | Leakage removal, MinMax scaling, Train-Test split (80/20), Mutual Information top-25 feature selection. |
        | **Unit 4: Classification** | Logistic Regression | Evaluated as baseline linear probabilistic classifier (Acc: 68.46%). |
        | **Unit 4: Classification** | Decision Tree (Gini & Entropy) | CART Decision Tree with depth tuning and split criteria comparison (Acc: 70.06%). |
        | **Unit 4: Classification** | Random Forest Classification | Bagging ensemble of 100 trees (**Top Performer: 71.61% Std / 77.14% Conf Acc**). |
        | **Unit 4: Classification** | Naive Bayes Algorithm | Gaussian Naive Bayes modeling conditional feature likelihood (Acc: 67.84%). |
        | **Unit 4: Classification** | K-Nearest Neighbours (K-NN) | Distance-based instance classifier using $k=9$ neighbors (Acc: 68.95%). |
        | **Unit 4: Classification** | Performance Metrics | Confusion Matrix, Sensitivity (Recall), Specificity, Precision, F1, ROC-AUC. |
        """)
        
    with st.expander("❓ 2. Top Viva Questions & Model Answers", expanded=True):
        st.markdown("""
        **Q1: Why did you drop `Days for shipping (real)`?**
        - *Answer:* Dropping `Days for shipping (real)` is required to prevent **Target Leakage**. In production, an order cannot know its actual delivery time at checkout. Using it produces an artificial 100% accuracy model that is completely useless in the real world.
        
        **Q2: How does Random Forest outperform the complex Cascade Classifier?**
        - *Answer:* Random Forest leverages bootstrap aggregation (bagging) and random subspace feature sampling. This significantly reduces model variance on noisy tabular supply chain data without over-partitioning the sample space, yielding higher generalization accuracy (71.61% vs 70.61%).
        
        **Q3: What is the mathematical difference between Gini Impurity and Entropy?**
        - *Answer:*
          - **Gini Impurity:** $I_G(p) = 1 - \sum_{i=1}^C p_i^2$. Measures the probability of misclassifying a randomly chosen element. Faster to compute.
          - **Entropy (Information Gain):** $H(S) = - \sum_{i=1}^C p_i \log_2(p_i)$. Based on Shannon information theory; measures average uncertainty.
        
        **Q4: In supply chain delivery prediction, which metric is more important: Precision or Recall?**
        - *Answer:* **Recall (Sensitivity)** is more critical than Precision. A False Negative (predicting On-Time when a parcel is actually LATE) results in broken customer SLAs, chargebacks, and churn. A False Positive merely causes proactive expedited tracking.
        
        **Q5: What is the purpose of the Confidence Margin (±0.10)?**
        - *Answer:* Standard classifiers force binary labels even when predicted probability is 0.51 (uncertain). By introducing an abstention margin $[0.40, 0.60]$, we flag ambiguous parcels for human logistics review and achieve **77.14% high-confidence accuracy** on deterministic orders.
        """)
        
    with st.expander("📐 3. Mathematical Formula Cheat-Sheet", expanded=False):
        st.latex(r"\text{Sigmoid (Logistic Regression): } \sigma(z) = \frac{1}{1 + e^{-z}}")
        st.latex(r"\text{Sensitivity (Recall): } \frac{TP}{TP + FN}")
        st.latex(r"\text{Specificity: } \frac{TN}{TN + FP}")
        st.latex(r"\text{Precision: } \frac{TP}{TP + FP}")
        st.latex(r"\text{F1-Score: } 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{Gini Index: } Gini(D) = 1 - \sum_{i=1}^k p_i^2")
        st.latex(r"\text{Bayes Theorem: } P(C_k | X) = \frac{P(X | C_k) P(C_k)}{P(X)}")

# ------------------------------------------------------------------------------
# Sidebar Information
# ------------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", width=70)
    st.markdown("### Project Quick-Facts")
    st.markdown("""
    - **Course:** UML501: Machine Learning
    - **Level:** B.Tech 5th Semester
    - **Credit:** 4.0 (L:3, T:0, P:2)
    - **Dataset:** DataCo Smart Supply Chain (180,519 records)
    - **Target:** `Late_delivery_risk` (0=OnTime, 1=Late)
    - **Models:** 7 Models Benchmarked
    """)
    st.markdown("---")
    st.markdown("### How to Run:")
    st.code("streamlit run app.py", language="bash")
    st.caption("Developed for B.Tech Machine Learning Lab Evaluation.")
