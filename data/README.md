# Dataset Information

This project predicts delivery delay risks using the **DataCo Smart Supply Chain for Big Data Analysis** dataset.

## Dataset Details
- **Source:** [Kaggle - DataCo Smart Supply Chain for Big Data Analysis](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)
- **Target Variable:** `Late_delivery_risk` (Binary Classification: `0` = On-Time, `1` = Late)
- **Raw Observations:** 180,519 supply chain transactions across global markets.

## Setup Instructions for Re-training
1. Download `DataCo_Enhanced_v1.csv` (or the raw `DataCoSupplyChainDataset.csv`).
2. Place the file inside this directory:
   ```
   data/DataCo_Enhanced_v1.csv
   ```
   Alternatively, specify a custom path using the `DATA_PATH` environment variable:
   ```bash
   export DATA_PATH="/path/to/your/DataCo_Enhanced_v1.csv"  # Linux / macOS
   $env:DATA_PATH="D:\path\to\DataCo_Enhanced_v1.csv"       # Windows PowerShell
   ```

> **Note:** The Streamlit web application (`app.py`) runs out-of-the-box without requiring this raw 130MB dataset, as it utilizes the pre-trained model artifacts and evaluation sample stored in the `models/` directory.
