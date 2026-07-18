# 📦 Delivery Delay Prediction — Cascade ML Classifier

A machine learning system that predicts whether a supply chain delivery will be **On-Time**, **At-Risk**, or **Delayed** — built on the DataCo Smart Supply Chain dataset (180,519 rows) augmented with weather data.

---

## 🔍 The Core Problem

A naive global model trained on this dataset achieves ~80% accuracy but **fails badly on certain shipping tiers** — Standard Class produces an F1-score of just 0.04 for the delay class. The reason: class imbalance in `Late_delivery_risk` is **conditional on Shipping Mode, not global**.

- **First Class** → ~95% of orders are late (majority is late)
- **Second Class** → ~76% late (moderate imbalance)
- **Same Day** → ~46% late (near balanced)
- **Standard Class** → ~38% late (delay is the minority — 59% of all data)

A single global model lumps these together, creating a Late-biased decision boundary that works well for First Class but completely misses Standard Class delays. The **Cascade Classifier** solves this by training a specialised model per tier.

---

## 🏗️ Architecture Overview

```
Raw Data (180,519 rows)
        │
        ▼
┌─────────────────────────────────────┐
│     v8 Leakage-Free Pipeline        │
│  Load → Clean → Feature Engineering│
│  → Split → Impute → IQR Cap         │
│  → Log Transform → Encode           │
│  → Normalize → Feature Reduction    │
│  → Final: 30 features               │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│       Cascade Classifier            │
│  Split by Shipping Mode             │
│                                     │
│  First Class   → Logistic Regression│
│  Second Class  → XGBoost            │
│  Same Day      → Gradient Boosting  │
│  Standard Class→ LightGBM + OOF thr │
└─────────────────────────────────────┘
        │
        ▼
  Confidence-Based Output
  Confident On-Time | At-Risk | Confident Late
```

---

## ⚙️ Preprocessing Pipeline (11 Steps, Leakage-Free)

All statistics are computed **on the training set only** and applied identically to the test set. The test set is never seen during any preprocessing step.

| Step | What It Does | Why |
|---|---|---|
| Load & Clean | Drop leakage columns (`Days for shipping (real)`, delivery status, order status) | These are post-hoc labels that directly encode the target |
| Feature Engineering | Create 12 new features from existing data (row-level only) | Safe to create before the split — no global stats used |
| Train/Test Split | 80/20 stratified by target | Everything after this is fit on train only |
| Imputation | Mean (numeric) / mode (categorical) — train stats only | Fills NaN without leaking test information |
| Outlier Capping | IQR bounds computed on train | Winsorises extremes, preserves row count |
| Geo Feature | Distance from geographic centroid (train median) | Geographic remoteness signal without leakage |
| Log Transform | log1p for features with skewness > 0.5 | Compresses right-skewed distributions (Sales, Profit, Discount) |
| Encoding | OHE for < 5 unique values; Target Encoding (smoothing=10) for ≥ 5 | Avoids dimension explosion on high-cardinality features |
| Normalize | MinMaxScaler fit on train → [0, 1] | Required for Logistic Regression; no single feature dominates |
| Feature Reduction | 3-stage: variance filter → correlation filter → MI top-30 | Removes noise, reduces overfitting |
| Baseline LR | Logistic Regression as interpretable first model | Establishes a performance floor |

### **Feature Engineering — What Was Created**

| Feature | Logic | Rationale |
|---|---|---|
| `order_month`, `order_dayofweek` | Extracted from order date | Seasonal delivery patterns |
| `month_sin/cos`, `dow_sin/cos` | sin/cos cyclical encoding | December → January is continuous, not a jump |
| `is_weekend` | 1 if dayofweek ≥ 5 | Weekend operations differ |
| `ship_priority` | Ordinal: Standard=1, Second=2, First=3, Same Day=4 | Urgency signal |
| `sched_x_priority` | scheduled_days × ship_priority | Interaction: urgent + long schedule = high risk |
| `price_per_item` | Sales / (Quantity + 1) | Item value signal |
| `profit_margin` | Benefit / (Sales + 1) | Profit pressure on delivery |
| `discount_ratio` | Discount / (Price + 1) | Heavy discounts may correlate with delays |
| `dist_center` | Distance from lat/lon centroid (train median only) | Geographic remoteness |

### **3-Stage Feature Reduction**

| Stage | Threshold | Typical Drop | Remaining |
|---|---|---|---|
| Near-zero variance | var < 0.001 | ~3–5 features | ~65 |
| High correlation | \|r\| > 0.85 | ~19 features (12 pairs, 6 exact duplicates) | ~46 |
| Mutual Information top-N | Top 30 by MI score | ~16 features | **30** |

---

## 🤖 Models

### Baseline Models (4 single-model benchmarks)

| Model | Std Accuracy | Conf Accuracy | At-Risk % | CV Accuracy |
|---|---|---|---|---|
| Logistic Regression | ~75.5% | ~79.0% | ~15.2% | ~80.6% |
| XGBoost | ~81.2% | ~86.0% | ~14.7% | ~86.3% |
| Random Forest | ~80.3% | ~85.5% | ~16.2% | — |
| GBM (sklearn) | ~76.6% | ~80.5% | ~14.2% | ~82.4% |

### **Cascade Classifier — The Innovation**

| Tier | Model | Why This Model | CV Acc | Std Acc | Conf Acc |
|---|---|---|---|---|---|
| First Class | Logistic Regression (C=10) | ~95% delay rate — linear boundary sufficient | ~97.9% | ~97.8% | ~98.5% |
| Second Class | XGBoost + scale_pos_weight=3.3 | 75% imbalance — penalise missed On-Time 3.3× | ~93.7% | ~92.5% | ~95.0% |
| Same Day | Gradient Boosting (max_depth ≤ 4) | Small dataset (~1,975 rows) — shallow trees prevent overfitting | ~97.1% | ~96.5% | ~97.5% |
| Standard Class | LightGBM + OOF threshold tuning (≈0.37) | Largest tier (59% of data), delay is minority — leaf-wise growth + is_unbalance=True | ~90.6% | ~87.0% | ~91.0% |

**Cascade overall: 90.5% standard accuracy · 93.5% confident accuracy · only 8.1% At-Risk**

---

## 🎯 Confidence-Based Evaluation

Rather than forcing every prediction to a hard 0 or 1, each model outputs a probability. Predictions within **±0.10 of the decision boundary** are flagged as **At-Risk** for human review.

```
P(Late) < 0.40           → Confident On-Time  ✅  (included in accuracy)
0.40 ≤ P(Late) < 0.60   → At-Risk            ⚠️  (flagged for review)
P(Late) ≥ 0.60           → Confident Late     🔴  (included in accuracy)
```

This is operationally more useful than a uniform accuracy number — it concentrates human attention on cases where the model is genuinely uncertain, which is where it matters most in supply chain operations.

---

## 📁 Repository Structure

```
delivery-delay-prediction/
├── final.ipynb          ← Full pipeline: preprocessing + all 5 models + comparison
├── report.pdf           ← Project presentation with problem framing and results
└── README.md
```

---

## 📊 Key Results

- ✅ Cascade Classifier outperforms all 4 single-model baselines on every metric
- ✅ Lowest At-Risk % (8.1%) — most confident of all models tested
- ✅ Resolves the Standard Class F1 = 0.04 failure of the global model
- ✅ 11-step leakage-free pipeline — test set seen exactly once, at final evaluation
- ✅ Confidence-based output makes the system actionable for operations teams, not just accurate on paper

---

## 🛠️ Libraries Used

```python
pandas · numpy · scikit-learn · xgboost · lightgbm · matplotlib
```
