# 6001CMD CW1 - Credit Card Default: Data Quality and Preprocessing

Dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

All analysis is done in Python (pandas, scikit-learn, imbalanced-learn). Excel was not used.
The original UCI file is read programmatically with pandas.

## Setup
    python -m venv venv
    venv\Scripts\activate        (Windows)   |   source venv/bin/activate   (macOS/Linux)
    pip install -r requirements.txt

Place `default of credit card clients.xls` (downloaded from the UCI link above) in this folder.
If `default of credit card clients.csv` is not present, the scripts read the .xls directly.

## Run order
1. `python eda_credit_risk.py`        Tasks 1 and 3: dataset inspection, data quality, figures in `figures/`
2. `python preprocessing_pipeline.py` Task 4: preprocessing pipeline (no predictive model is trained)
