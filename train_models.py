"""
train_models.py
End-to-end model training, evaluation, and serialization for Delivery Delay Prediction.
Includes:
  1. Syllabus-Pure Models: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN
  2. Advanced Model: Tier-Specific Cascade Classifier with Confidence Margins
Saves all artifacts to ./models/ for instant loading in the Streamlit web app.
"""

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.feature_selection import mutual_info_classif
from category_encoders import TargetEncoder
import xgboost as xgb
import lightgbm as lgb

RANDOM_STATE = 42
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "DataCo_Enhanced_v1.csv")
FALLBACK_DATA_PATH = r"D:\Documents\study\AI\Project-I\DataCo_Enhanced_v1.csv"
DATA_PATH = os.environ.get("DATA_PATH", DEFAULT_DATA_PATH if os.path.exists(DEFAULT_DATA_PATH) else FALLBACK_DATA_PATH)
TARGET_COL = "Late_delivery_risk"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


def load_and_preprocess(sample_size=40000):
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if sample_size and len(df) > sample_size:
        print(f"Sampling {sample_size} stratified rows for fast, reproducible training...")
        df, _ = train_test_split(
            df, train_size=sample_size, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
        )
        df = df.reset_index(drop=True)
    
    print(f"Dataset shape: {df.shape}")
    y = df[TARGET_COL].copy().astype(int)
    df.drop(columns=[TARGET_COL], inplace=True)
    
    drop_cols = [
        'Days for shipping (real)', 'shipping date (DateOrders)', 'shipping_date',
        'Target_Date', 'Delivery Status', 'Order Status',
        'Customer Email', 'Customer Password', 'Product Description',
        'Product Status', 'Product Image',
        'Order Item Total', 'Product Price', 'Product Category Id',
        'lon_round', 'lat_round', 'Customer Fname', 'Customer Lname',
        'Customer Street', 'Order Zipcode', 'Product Card Id'
    ]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    
    shipping_mode_series = df['Shipping Mode'].copy()
    
    # ── Feature Engineering ──
    date_col = 'order date (DateOrders)'
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors='coerce')
        df['order_month']     = dt.dt.month.fillna(6).astype(int)
        df['order_dayofweek'] = dt.dt.dayofweek.fillna(2).astype(int)
        df['month_sin'] = np.sin(2 * np.pi * df['order_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['order_month'] / 12)
        df['dow_sin']   = np.sin(2 * np.pi * df['order_dayofweek'] / 7)
        df['dow_cos']   = np.cos(2 * np.pi * df['order_dayofweek'] / 7)
        df['is_weekend'] = (df['order_dayofweek'] >= 5).astype(int)
        df.drop(columns=[date_col], inplace=True)
    
    if 'Shipping Mode' in df.columns:
        df['ship_priority'] = df['Shipping Mode'].map(
            {'Standard Class': 1, 'Second Class': 2, 'First Class': 3, 'Same Day': 4}
        ).fillna(1)
        
    if {'Days for shipment (scheduled)', 'ship_priority'}.issubset(df.columns):
        df['sched_x_priority'] = df['Days for shipment (scheduled)'] * df['ship_priority']
        
    if {'Sales', 'Order Item Quantity'}.issubset(df.columns):
        df['price_per_item'] = df['Sales'] / (df['Order Item Quantity'] + 1)
    if {'Benefit per order', 'Sales'}.issubset(df.columns):
        df['profit_margin'] = df['Benefit per order'] / (df['Sales'] + 1)
    if {'Order Item Discount', 'Order Item Product Price'}.issubset(df.columns):
        df['discount_ratio'] = df['Order Item Discount'] / (df['Order Item Product Price'] + 1)
        
    X_train, X_test, y_train, y_test, sm_train, sm_test = train_test_split(
        df, y, shipping_mode_series, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    sm_train = sm_train.reset_index(drop=True)
    sm_test  = sm_test.reset_index(drop=True)
    
    num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    impute_num = {}
    for col in num_cols:
        val = float(X_train[col].mean())
        impute_num[col] = val
        X_train[col] = X_train[col].fillna(val)
        X_test[col]  = X_test[col].fillna(val)
        
    impute_cat = {}
    for col in cat_cols:
        mode_val = X_train[col].mode()
        val = str(mode_val[0]) if len(mode_val) > 0 else 'Unknown'
        impute_cat[col] = val
        X_train[col] = X_train[col].fillna(val)
        X_test[col]  = X_test[col].fillna(val)
        
    outlier_bounds = {}
    for col in num_cols:
        Q1, Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lo, hi = float(Q1 - 1.5 * IQR), float(Q3 + 1.5 * IQR)
            outlier_bounds[col] = (lo, hi)
            X_train[col] = X_train[col].clip(lo, hi)
            X_test[col]  = X_test[col].clip(lo, hi)
            
    if {'Latitude', 'Longitude'}.issubset(X_train.columns):
        lat_med = float(X_train['Latitude'].median())
        lon_med = float(X_train['Longitude'].median())
        X_train['dist_center'] = np.sqrt((X_train['Latitude'] - lat_med)**2 + (X_train['Longitude'] - lon_med)**2)
        X_test['dist_center']  = np.sqrt((X_test['Latitude'] - lat_med)**2 + (X_test['Longitude'] - lon_med)**2)
    else:
        lat_med, lon_med = 0.0, 0.0
        
    log_candidates = ['Sales', 'Order Item Discount', 'Order Item Product Price',
                      'Benefit per order', 'Order Item Profit Ratio',
                      'dist_center', 'price_per_item', 'Sales per customer',
                      'Order Profit Per Order']
    log_transformed = []
    for col in log_candidates:
        if col in X_train.columns and abs(X_train[col].skew()) > 0.5:
            X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
            X_test[f'{col}_log']  = np.log1p(X_test[col].clip(lower=0))
            log_transformed.append(col)
            
    cat_cols_current = X_train.select_dtypes(include=['object']).columns.tolist()
    low_card = [c for c in cat_cols_current if X_train[c].nunique() < 5]
    high_card = [c for c in cat_cols_current if X_train[c].nunique() >= 5]
    
    if low_card:
        X_train = pd.get_dummies(X_train, columns=low_card, drop_first=True, dtype=int)
        X_test  = pd.get_dummies(X_test,  columns=low_card, drop_first=True, dtype=int)
        X_test  = X_test.reindex(columns=X_train.columns, fill_value=0)
        
    te = None
    if high_card:
        te = TargetEncoder(cols=high_card, smoothing=10)
        X_train[high_card] = te.fit_transform(X_train[high_card], y_train)
        X_test[high_card]  = te.transform(X_test[high_card])
        
    scaler = MinMaxScaler()
    feature_cols = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)
    
    var = X_train_scaled.var()
    low_var = var[var < 0.001].index.tolist()
    X_train_scaled.drop(columns=low_var, inplace=True)
    X_test_scaled.drop(columns=low_var, inplace=True)
    
    corr = X_train_scaled.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [c for c in upper.columns if any(upper[c] > 0.90)]
    X_train_scaled.drop(columns=high_corr, inplace=True)
    X_test_scaled.drop(columns=high_corr, inplace=True)
    
    X_train_scaled = X_train_scaled.fillna(0)
    X_test_scaled  = X_test_scaled.fillna(0)
    
    print("Selecting top 25 features via Mutual Information...")
    mi = pd.Series(
        mutual_info_classif(X_train_scaled, y_train, random_state=RANDOM_STATE),
        index=X_train_scaled.columns
    ).sort_values(ascending=False)
    
    selected_features = mi.head(25).index.tolist()
    X_tr_final = X_train_scaled[selected_features].copy()
    X_te_final = X_test_scaled[selected_features].copy()
    
    print(f"Final selected features ({len(selected_features)}): {selected_features}")
    
    metadata = {
        'drop_cols': drop_cols,
        'impute_num': impute_num,
        'impute_cat': impute_cat,
        'outlier_bounds': outlier_bounds,
        'lat_med': lat_med,
        'lon_med': lon_med,
        'log_transformed': log_transformed,
        'low_card': low_card,
        'high_card': high_card,
        'target_encoder': te,
        'scaler': scaler,
        'train_cols_after_ohe': feature_cols,
        'selected_features': selected_features,
        'mi_scores': {k: float(v) for k, v in mi.head(25).to_dict().items()}
    }
    
    return X_tr_final, X_te_final, y_train, y_test, sm_train, sm_test, metadata


def compute_metrics(y_true, y_pred, y_proba=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred).tolist()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    auc = float(roc_auc_score(y_true, y_proba)) if y_proba is not None else 0.0
    
    if y_proba is not None:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        conf_mask = (y_proba < 0.5 - MARGIN) | (y_proba >= 0.5 + MARGIN)
        n_conf = int(conf_mask.sum())
        n_risk = int((~conf_mask).sum())
        acc_conf = float(accuracy_score(y_true_np[conf_mask], y_pred_np[conf_mask])) if n_conf > 0 else acc
        pct_risk = float(n_risk / len(y_true) * 100)
    else:
        acc_conf, pct_risk, n_conf, n_risk = acc, 0.0, len(y_true), 0
        
    return {
        'accuracy': round(float(acc), 4),
        'confidence_accuracy': round(float(acc_conf), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'specificity': round(float(spec), 4),
        'f1_score': round(float(f1), 4),
        'roc_auc': round(float(auc), 4),
        'confusion_matrix': cm,
        'n_total': int(len(y_true)),
        'n_conf': n_conf,
        'n_risk': n_risk,
        'pct_risk': round(pct_risk, 2)
    }


def train_all_models():
    X_train, X_test, y_train, y_test, sm_train, sm_test, metadata = load_and_preprocess(sample_size=35000)
    
    models = {}
    eval_results = {}
    roc_curves = {}
    
    # 1. Logistic Regression (Syllabus)
    print("\n--- Training Model 1: Logistic Regression ---")
    t0 = time.time()
    lr = LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    lr_time = time.time() - t0
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    models['Logistic Regression'] = lr
    eval_results['Logistic Regression'] = {**compute_metrics(y_test, lr_pred, lr_proba), 'train_time': round(lr_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, lr_proba)
    roc_curves['Logistic Regression'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"LR Acc: {eval_results['Logistic Regression']['accuracy']}")
    
    # 2. Decision Tree (Syllabus)
    print("\n--- Training Model 2: Decision Tree (Gini & Entropy) ---")
    t0 = time.time()
    dt = DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_split=20, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    dt_time = time.time() - t0
    dt_proba = dt.predict_proba(X_test)[:, 1]
    dt_pred = (dt_proba >= 0.5).astype(int)
    models['Decision Tree'] = dt
    eval_results['Decision Tree'] = {**compute_metrics(y_test, dt_pred, dt_proba), 'train_time': round(dt_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, dt_proba)
    roc_curves['Decision Tree'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"DT Acc: {eval_results['Decision Tree']['accuracy']}")
    
    # 3. Random Forest (Syllabus)
    print("\n--- Training Model 3: Random Forest Classifier ---")
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    models['Random Forest'] = rf
    eval_results['Random Forest'] = {**compute_metrics(y_test, rf_pred, rf_proba), 'train_time': round(rf_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    roc_curves['Random Forest'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"RF Acc: {eval_results['Random Forest']['accuracy']}")
    
    # 4. Naive Bayes (Syllabus)
    print("\n--- Training Model 4: Gaussian Naive Bayes ---")
    t0 = time.time()
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_time = time.time() - t0
    gnb_proba = gnb.predict_proba(X_test)[:, 1]
    gnb_pred = (gnb_proba >= 0.5).astype(int)
    models['Naive Bayes'] = gnb
    eval_results['Naive Bayes'] = {**compute_metrics(y_test, gnb_pred, gnb_proba), 'train_time': round(gnb_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, gnb_proba)
    roc_curves['Naive Bayes'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"NB Acc: {eval_results['Naive Bayes']['accuracy']}")
    
    # 5. K-NN (Syllabus)
    print("\n--- Training Model 5: K-Nearest Neighbors (K-NN) ---")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=9, n_jobs=-1)
    knn.fit(X_train.iloc[:10000], y_train.iloc[:10000])
    knn_time = time.time() - t0
    knn_proba = knn.predict_proba(X_test.iloc[:2000])[:, 1]
    knn_pred = (knn_proba >= 0.5).astype(int)
    models['K-NN'] = knn
    eval_results['K-NN'] = {**compute_metrics(y_test.iloc[:2000], knn_pred, knn_proba), 'train_time': round(knn_time, 2)}
    fpr, tpr, _ = roc_curve(y_test.iloc[:2000], knn_proba)
    roc_curves['K-NN'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"KNN Acc: {eval_results['K-NN']['accuracy']}")

    # 6. XGBoost
    print("\n--- Training Model 6: XGBoost ---")
    t0 = time.time()
    xgb_mod = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1
    )
    xgb_mod.fit(X_train, y_train)
    xgb_time = time.time() - t0
    xgb_proba = xgb_mod.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_proba >= 0.5).astype(int)
    models['XGBoost'] = xgb_mod
    eval_results['XGBoost'] = {**compute_metrics(y_test, xgb_pred, xgb_proba), 'train_time': round(xgb_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, xgb_proba)
    roc_curves['XGBoost'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"XGB Acc: {eval_results['XGBoost']['accuracy']}")
    
    # 7. Tier-Specific Cascade Classifier
    print("\n--- Training Model 7: Tier-Specific Cascade Classifier ---")
    t0 = time.time()
    cascade_models = {}
    
    # First Class -> Logistic Regression
    fc_tr = (sm_train == 'First Class').values
    fc_m = LogisticRegression(C=0.1, max_iter=2000, random_state=RANDOM_STATE)
    fc_m.fit(X_train[fc_tr], y_train[fc_tr])
    cascade_models['First Class'] = {'model': fc_m, 'threshold': 0.5}
    
    # Second Class -> XGBoost
    sc_tr = (sm_train == 'Second Class').values
    sc_m = xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1)
    sc_m.fit(X_train[sc_tr], y_train[sc_tr])
    cascade_models['Second Class'] = {'model': sc_m, 'threshold': 0.5}
    
    # Same Day -> Gradient Boosting
    sd_tr = (sm_train == 'Same Day').values
    sd_m = GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)
    sd_m.fit(X_train[sd_tr], y_train[sd_tr])
    cascade_models['Same Day'] = {'model': sd_m, 'threshold': 0.5}
    
    # Standard Class -> LightGBM with threshold sweep
    st_tr = (sm_train == 'Standard Class').values
    st_m = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.08, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    st_m.fit(X_train[st_tr], y_train[st_tr])
    st_proba_val = st_m.predict_proba(X_train[st_tr])[:, 1]
    
    best_thr, best_acc = 0.5, 0.0
    for thr in np.arange(0.35, 0.55, 0.02):
        acc = accuracy_score(y_train[st_tr], (st_proba_val >= thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    cascade_models['Standard Class'] = {'model': st_m, 'threshold': best_thr}
    
    cascade_preds = np.zeros(len(y_test), dtype=int)
    cascade_probas = np.zeros(len(y_test), dtype=float)
    
    for mode in ['First Class', 'Second Class', 'Same Day', 'Standard Class']:
        mask = (sm_test == mode).values
        if mask.any():
            m_info = cascade_models[mode]
            p = m_info['model'].predict_proba(X_test[mask])[:, 1]
            cascade_probas[mask] = p
            cascade_preds[mask] = (p >= m_info['threshold']).astype(int)
            
    cascade_time = time.time() - t0
    models['Cascade Classifier'] = cascade_models
    eval_results['Cascade Classifier'] = {**compute_metrics(y_test, cascade_preds, cascade_probas), 'train_time': round(cascade_time, 2)}
    fpr, tpr, _ = roc_curve(y_test, cascade_probas)
    roc_curves['Cascade Classifier'] = {'fpr': [round(x, 4) for x in fpr.tolist()], 'tpr': [round(x, 4) for x in tpr.tolist()]}
    print(f"Cascade Acc: {eval_results['Cascade Classifier']['accuracy']}, Conf Acc: {eval_results['Cascade Classifier']['confidence_accuracy']}")
    
    # Feature Importance from Random Forest
    rf_feat_imp = {k: round(float(v), 4) for k, v in pd.Series(rf.feature_importances_, index=metadata['selected_features']).sort_values(ascending=False).head(15).to_dict().items()}
    
    # Save artifacts
    print("\nSaving trained models and evaluation artifacts...")
    joblib.dump(models, os.path.join(MODELS_DIR, "all_models.joblib"))
    joblib.dump(metadata, os.path.join(MODELS_DIR, "pipeline_metadata.joblib"))
    
    test_sample_df = X_test.head(100).copy()
    test_sample_df['true_late_risk'] = y_test.head(100).values
    test_sample_df['Shipping_Mode'] = sm_test.head(100).values
    test_sample_df.to_csv(os.path.join(MODELS_DIR, "test_sample.csv"), index=False)
    
    with open(os.path.join(MODELS_DIR, "eval_results.json"), 'w') as f:
        json.dump(eval_results, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, "roc_curves.json"), 'w') as f:
        json.dump(roc_curves, f, indent=2)
        
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), 'w') as f:
        json.dump(rf_feat_imp, f, indent=2)
        
    print(f"All artifacts successfully saved to {MODELS_DIR} [OK]")
    return eval_results


if __name__ == "__main__":
    t_start = time.time()
    results = train_all_models()
    print(f"\nAll models completed in {time.time() - t_start:.2f}s")
