"""
================================================================================
UML501: MACHINE LEARNING (B.Tech 5th Semester) - MINI PROJECT (15 MARKS)
Project Title: Supply Chain Delivery Delay Prediction
Syllabus Modules Covered:
  - Unit 2: Data Pre-processing (Cleaning, Scaling, Feature Selection, Train-Test Split)
  - Unit 4: Classification (Logistic Regression, Decision Trees [Gini/Entropy],
            Random Forest, Gaussian Naive Bayes, K-Nearest Neighbors)
  - Performance Metrics: Accuracy, Sensitivity (Recall), Specificity, Precision, F1, Confusion Matrix
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DataCo_Enhanced_v1.csv")
FALLBACK_DATA_PATH = r"D:\Documents\study\AI\Project-I\DataCo_Enhanced_v1.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH if os.path.exists(DEFAULT_DATA_PATH) else FALLBACK_DATA_PATH)
TARGET_COL = "Late_delivery_risk"

def run_syllabus_pipeline(sample_size=30000):
    print("=" * 75)
    print("  UML501 MACHINE LEARNING: DELIVERY DELAY PREDICTION EXPERIMENT")
    print("=" * 75)
    
    # --------------------------------------------------------------------------
    # 1. DATA PREPROCESSING (Syllabus Unit 2)
    # --------------------------------------------------------------------------
    print("\n[Step 1] Loading dataset and performing Data Cleaning...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    if sample_size and len(df) > sample_size:
        df, _ = train_test_split(df, train_size=sample_size, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
        df = df.reset_index(drop=True)
        
    print(f"  Loaded sample shape: {df.shape}")
    
    # Target variable extraction
    y = df[TARGET_COL].astype(int)
    
    # TARGET LEAKAGE PREVENTION: Drop columns that contain future information
    # or direct proxies of the target (e.g. Days for shipping (real), Delivery Status)
    leakage_cols = [
        'Days for shipping (real)', 'shipping date (DateOrders)', 'shipping_date',
        'Target_Date', 'Delivery Status', 'Order Status',
        'Customer Email', 'Customer Password', 'Product Description',
        'Product Status', 'Product Image', 'Order Item Total', 'Product Price',
        'Product Category Id', 'lon_round', 'lat_round', 'Late_delivery_risk'
    ]
    df_clean = df.drop(columns=[c for c in leakage_cols if c in df.columns])
    
    # Feature Engineering (Syllabus: Data Transformation)
    print("[Step 2] Feature Engineering & Transformation...")
    if 'Shipping Mode' in df_clean.columns:
        mode_map = {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}
        df_clean['ship_priority'] = df_clean['Shipping Mode'].map(mode_map).fillna(1)
        
    if {'Days for shipment (scheduled)', 'ship_priority'}.issubset(df_clean.columns):
        df_clean['sched_x_priority'] = df_clean['Days for shipment (scheduled)'] * df_clean['ship_priority']
        
    if {'Sales', 'Order Item Quantity'}.issubset(df_clean.columns):
        df_clean['price_per_item'] = df_clean['Sales'] / (df_clean['Order Item Quantity'] + 1)
    if {'Benefit per order', 'Sales'}.issubset(df_clean.columns):
        df_clean['profit_margin'] = df_clean['Benefit per order'] / (df_clean['Sales'] + 1)
        
    # Select clean numerical and key categorical predictors
    key_features = [
        'Days for shipment (scheduled)', 'ship_priority', 'sched_x_priority',
        'Sales', 'Order Item Quantity', 'Benefit per order', 'profit_margin',
        'Order Item Discount', 'Sales per customer', 'Latitude', 'Longitude'
    ]
    available_features = [f for f in key_features if f in df_clean.columns]
    X = df_clean[available_features].copy()
    
    # Handling Missing Values (Mean Imputation)
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())
            
    # Train-Test Split (Syllabus: Splitting dataset into Training and Testing set)
    print("[Step 3] Splitting dataset into Train (80%) and Test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    # Feature Scaling (Syllabus: Feature Scaling via Normalization [0, 1])
    print("[Step 4] Feature Scaling using MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # --------------------------------------------------------------------------
    # 2. MODEL TRAINING (Syllabus Unit 4: Classification)
    # --------------------------------------------------------------------------
    models = {
        "1. Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_STATE
        ),
        "2. Decision Tree (Gini)": DecisionTreeClassifier(
            criterion='gini', max_depth=6, min_samples_split=20, random_state=RANDOM_STATE
        ),
        "3. Decision Tree (Entropy)": DecisionTreeClassifier(
            criterion='entropy', max_depth=6, min_samples_split=20, random_state=RANDOM_STATE
        ),
        "4. Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "5. Gaussian Naive Bayes": GaussianNB(),
        "6. K-Nearest Neighbors (K-NN)": KNeighborsClassifier(
            n_neighbors=9, n_jobs=-1
        )
    }
    
    results = []
    print("\n" + "=" * 75)
    print("  TRAINING & EVALUATION (UML501 SYLLABUS BENCHMARK)")
    print("=" * 75)
    
    for name, model in models.items():
        t0 = time.time()
        
        # Fit model
        if "K-NN" in name:
            # KNN on 10k subset for fast execution
            model.fit(X_train_scaled.iloc[:10000], y_train.iloc[:10000])
            y_pred = model.predict(X_test_scaled.iloc[:2000])
            y_proba = model.predict_proba(X_test_scaled.iloc[:2000])[:, 1]
            y_eval = y_test.iloc[:2000]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
            y_eval = y_test
            
        elapsed = time.time() - t0
        
        # Performance Metrics (Syllabus: Accuracy, Sensitivity, Specificity, Precision)
        acc  = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec  = recall_score(y_eval, y_pred, zero_division=0)  # Sensitivity
        f1   = f1_score(y_eval, y_pred, zero_division=0)
        auc  = roc_auc_score(y_eval, y_proba) if y_proba is not None else 0.0
        
        # Specificity = TN / (TN + FP)
        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results.append({
            "Model": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec * 100:.2f}%",
            "Recall (Sensitivity)": f"{rec * 100:.2f}%",
            "Specificity": f"{spec * 100:.2f}%",
            "F1-Score": f"{f1 * 100:.2f}%",
            "ROC-AUC": f"{auc:.4f}",
            "Train Time": f"{elapsed:.2f}s"
        })
        
    # Display Results Table
    res_df = pd.DataFrame(results)
    print("\n" + res_df.to_string(index=False))
    
    print("\n" + "=" * 75)
    print("  CONFUSION MATRIX BREAKDOWN (Random Forest)")
    print("=" * 75)
    rf_model = models["4. Random Forest"]
    rf_pred = rf_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, rf_pred)
    print(f"  True Negatives (On-Time correctly predicted) : {cm[0][0]}")
    print(f"  False Positives (Predicted Late, was On-Time): {cm[0][1]}")
    print(f"  False Negatives (Predicted On-Time, was LATE): {cm[1][0]}  <-- Critical Supply Chain Risk")
    print(f"  True Positives (Late correctly predicted)    : {cm[1][1]}")
    
    print("\n" + "=" * 75)
    print("  FEATURE IMPORTANCE (Random Forest)")
    print("=" * 75)
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    for feat, imp in importances.items():
        print(f"  {feat:<32s} : {imp * 100:.2f}%")
        
    return res_df

if __name__ == "__main__":
    run_syllabus_pipeline(sample_size=30000)
