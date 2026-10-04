# ============================================================
# 6001CMD MACHINE LEARNING CW1
# TASK 4: DATA PREPROCESSING STRATEGY
# Dataset: UCI Default of Credit Card Clients
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from imblearn.over_sampling import SMOTENC


# ============================================================
# 0. SETTINGS
# ============================================================

DATASET_PATH = "default of credit card clients.csv"
EXCEL_SOURCE = "default of credit card clients.xls"   # original UCI download
TARGET = "default payment next month"

RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs("figures", exist_ok=True)


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

# Load the CSV if present; otherwise read the original UCI .xls directly
# with pandas (programmatic loading only, Excel software is not used).
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, header=1)
else:
    df = pd.read_excel(EXCEL_SOURCE, header=1)

print("\n" + "=" * 65)
print("TASK 4 - PREPROCESSING PIPELINE")
print("=" * 65)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# 2. DATA QUALITY VERIFICATION
# ============================================================

print("\n" + "=" * 65)
print("STEP 1 - DATA QUALITY VERIFICATION")
print("=" * 65)

print(
    f"Missing Values: "
    f"{df.isnull().sum().sum()}"
)

exact_duplicates = df.duplicated().sum()

repeated_profiles = (
    df.drop(columns=["ID"])
    .duplicated()
    .sum()
)

print(
    f"Exact Duplicate Rows (including ID): "
    f"{exact_duplicates}"
)

print(
    f"Repeated Profiles (excluding ID): "
    f"{repeated_profiles}"
)

# IMPORTANT:
# We do NOT automatically delete the 35 repeated profiles.
# Different IDs indicate that they may represent different clients
# who happen to have identical recorded characteristics.


# ============================================================
# 3. CLEAN DOCUMENTED CATEGORICAL INCONSISTENCIES
# ============================================================

print("\n" + "=" * 65)
print("STEP 2 - CATEGORICAL DATA CLEANING")
print("=" * 65)

print(
    "EDUCATION before cleaning:",
    sorted(df["EDUCATION"].unique())
)

print(
    "MARRIAGE before cleaning:",
    sorted(df["MARRIAGE"].unique())
)


# EDUCATION:
# 1 = graduate school
# 2 = university
# 3 = high school
# 4 = others
#
# The raw dataset also contains undocumented values 0, 5 and 6.
# For this proposed preprocessing strategy they are grouped as
# category 4 = "Other".

df["EDUCATION"] = df["EDUCATION"].replace(
    {
        0: 4,
        5: 4,
        6: 4
    }
)


# MARRIAGE:
# 1 = married
# 2 = single
# 3 = others
#
# Undocumented 0 is grouped into category 3 = "Other".

df["MARRIAGE"] = df["MARRIAGE"].replace(
    {
        0: 3
    }
)


print(
    "EDUCATION after cleaning:",
    sorted(df["EDUCATION"].unique())
)

print(
    "MARRIAGE after cleaning:",
    sorted(df["MARRIAGE"].unique())
)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 65)
print("STEP 3 - REMOVE IDENTIFIER AND SEPARATE X / y")
print("=" * 65)

# ID is retained during data-quality checking but is removed
# from predictive features because it is only an identifier.

X = df.drop(
    columns=[
        "ID",
        TARGET
    ]
)

y = df[TARGET].astype(int)

print(f"Predictive feature matrix: {X.shape}")
print(f"Target vector: {y.shape}")


# ============================================================
# 5. STRATIFIED TRAIN-TEST SPLIT
#
# This happens BEFORE:
# - scaling
# - PCA
# - resampling
#
# to avoid data leakage.
# ============================================================

print("\n" + "=" * 65)
print("STEP 4 - STRATIFIED TRAIN-TEST SPLIT")
print("=" * 65)

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(
    "\nTraining class distribution BEFORE balancing:"
)

print(
    Counter(y_train)
)

print(
    "\nTest class distribution:"
)

print(
    Counter(y_test)
)


# ============================================================
# 6. DEFINE FEATURE TYPES
# ============================================================

# These coded variables represent categorical/discrete states.
categorical_cols = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]


# Six highly correlated monthly billing variables
bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]


# Other continuous numerical predictors
numeric_cols = [
    "LIMIT_BAL",
    "AGE",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]


print("\nCategorical / discrete features:")
print(categorical_cols)

print("\nBilling features for optional PCA:")
print(bill_cols)

print("\nOther numerical features:")
print(numeric_cols)


# ============================================================
# 7. CONTINUOUS-FEATURE TRANSFORMATION (FITTED ON REAL TRAINING DATA)
#
# Order matters:
#   - Transformations are fitted on the ORIGINAL training data,
#     before SMOTENC, so PCA and scaling parameters are not
#     influenced by synthetic observations.
#   - Scaling happens BEFORE SMOTENC so that SMOTENC's
#     nearest-neighbour distances are not dominated by the
#     large NT$ magnitudes.
#
# BILL_AMT : Yeo-Johnson (handles negative values) -> PCA(2)
# PAY_AMT  : log(1 + x) (non-negative)             -> StandardScaler
# LIMIT_BAL, AGE : StandardScaler
# ============================================================

print("\n" + "=" * 65)
print("STEP 5 - TRANSFORM CONTINUOUS FEATURES (TRAINING-FITTED)")
print("=" * 65)

pay_amt_cols = [c for c in numeric_cols if c.startswith("PAY_AMT")]
other_numeric_cols = ["LIMIT_BAL", "AGE"]

bill_pipeline = Pipeline(
    steps=[
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("pca", PCA(n_components=2, random_state=RANDOM_STATE))
    ]
)

pay_amt_pipeline = Pipeline(
    steps=[
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler())
    ]
)

continuous_transformer = ColumnTransformer(
    transformers=[
        ("bill_pca", bill_pipeline, bill_cols),
        ("pay_amt_log", pay_amt_pipeline, pay_amt_cols),
        ("numerical", StandardScaler(), other_numeric_cols)
    ],
    remainder="drop"
)

pca_feature_names = ["BILL_PCA_1", "BILL_PCA_2"]
continuous_feature_names = (
    pca_feature_names
    + [f"LOG_{c}" for c in pay_amt_cols]
    + other_numeric_cols
)


def build_frame(X_part, fitted=True):
    """Transformed continuous features + untouched categorical codes."""
    values = (
        continuous_transformer.transform(X_part)
        if fitted
        else continuous_transformer.fit_transform(X_part)
    )
    cont = pd.DataFrame(values, columns=continuous_feature_names, index=X_part.index)
    return pd.concat([cont, X_part[categorical_cols]], axis=1)


# Fit on real training data only; test set is only transformed.
X_train_cont = build_frame(X_train, fitted=False)
X_test_cont = build_frame(X_test, fitted=True)

pca_model = continuous_transformer.named_transformers_["bill_pca"].named_steps["pca"]
explained_variance = pca_model.explained_variance_ratio_.sum() * 100

print(f"PCA explained variance (2 components, real training data): {explained_variance:.2f}%")

print("\nSkewness BEFORE vs AFTER transformation (training data):")
skew_compare = pd.DataFrame({
    "Before": X_train[pay_amt_cols].skew().values,
    "After log(1+x)": X_train_cont[[f"LOG_{c}" for c in pay_amt_cols]].skew().values
}, index=pay_amt_cols).round(2)
print(skew_compare)
skew_compare.to_csv("figures/skewness_before_after_log.csv")


# ============================================================
# 8. BALANCING - SMOTENC (TRAINING SET ONLY)
#
# SMOTENC is selected instead of plain SMOTE because the
# dataset contains mixed categorical and numerical features.
# It now operates on scaled continuous features.
# ============================================================

print("\n" + "=" * 65)
print("STEP 6 - TRAINING-SET BALANCING USING SMOTENC")
print("=" * 65)

categorical_indices = [X_train_cont.columns.get_loc(c) for c in categorical_cols]

smote_nc = SMOTENC(
    categorical_features=categorical_indices,
    random_state=RANDOM_STATE
)

X_train_balanced, y_train_balanced = smote_nc.fit_resample(X_train_cont, y_train)

# Convert back to DataFrame because some versions of
# imbalanced-learn may return NumPy data.
X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train_cont.columns)
y_train_balanced = pd.Series(y_train_balanced, name=TARGET)

print("Class distribution BEFORE SMOTENC:", Counter(y_train))
print("Class distribution AFTER SMOTENC:", Counter(y_train_balanced))
print("\nIMPORTANT: The test set has NOT been balanced.")


# ============================================================
# 9. CLASS DISTRIBUTION AFTER BALANCING
# ============================================================

balanced_counts = y_train_balanced.value_counts().sort_index()

plt.figure(figsize=(7, 5))
plt.bar(["0 = Non-Default", "1 = Default"], balanced_counts.values)
plt.title("Training Class Distribution After SMOTENC")
plt.xlabel("Target Class")
plt.ylabel("Number of Training Samples")
for i, value in enumerate(balanced_counts.values):
    plt.text(i, value + 200, f"{value:,}", ha="center")
plt.tight_layout()
plt.savefig("figures/class_distribution_after_smotenc.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 10. ONE-HOT ENCODING (AFTER SMOTENC)
#
# Encoding after SMOTENC keeps the categorical codes intact
# during resampling. Handles different scikit-learn versions.
# ============================================================

print("\n" + "=" * 65)
print("STEP 7 - ONE-HOT ENCODE CATEGORICAL FEATURES")
print("=" * 65)

try:
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

train_cat = categorical_encoder.fit_transform(X_train_balanced[categorical_cols].astype(int))
test_cat = categorical_encoder.transform(X_test_cont[categorical_cols].astype(int))

encoded_feature_names = categorical_encoder.get_feature_names_out(categorical_cols).tolist()
final_feature_names = continuous_feature_names + encoded_feature_names

X_train_processed_df = pd.DataFrame(
    np.hstack([X_train_balanced[continuous_feature_names].values.astype(float), train_cat]),
    columns=final_feature_names
)
X_test_processed_df = pd.DataFrame(
    np.hstack([X_test_cont[continuous_feature_names].values.astype(float), test_cat]),
    columns=final_feature_names
)

print(f"Processed training shape: {X_train_processed_df.shape}")
print(f"Processed test shape: {X_test_processed_df.shape}")

print(f"\nFinal number of processed features: {len(final_feature_names)}")
print("\nFeature breakdown:")
print(f"PCA components   : {len(pca_feature_names)}")
print(f"Numeric features : {len(continuous_feature_names) - len(pca_feature_names)}")
print(f"One-hot columns  : {len(encoded_feature_names)}")
for c in categorical_cols:
    n = sum(name.startswith(c + "_") for name in encoded_feature_names)
    print(f"  {c}: {n}")

print("\nFirst five processed training rows:")
print(X_train_processed_df.head())


# ============================================================
# 11. SAVE PREPROCESSED OUTPUT
#
# Optional evidence / future use.
# No predictive model is trained in this CW1 script.
# ============================================================

X_train_processed_df.to_csv("processed_training_features.csv", index=False)
X_test_processed_df.to_csv("processed_test_features.csv", index=False)
y_train_balanced.to_csv("processed_training_target.csv", index=False)
y_test.reset_index(drop=True).to_csv("processed_test_target.csv", index=False)


# ============================================================
# 12. FINAL PIPELINE SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("PREPROCESSING PIPELINE COMPLETE")
print("=" * 65)

print(
    """
Final workflow:

1. Load raw UCI dataset
2. Validate missing values and duplicate profiles
3. Resolve selected categorical inconsistencies
4. Remove ID from predictive features
5. Stratified train-test split
6. Fit on REAL training data only:
     BILL_AMT  -> Yeo-Johnson + standardise -> PCA (2 components)
     PAY_AMT   -> log(1+x) -> StandardScaler
     LIMIT_BAL, AGE -> StandardScaler
7. Apply the same training-fitted continuous transformations to the test set
8. Apply SMOTENC ONLY to the transformed training data
9. One-hot encode categorical/discrete variables
   - Fit encoder on balanced training data
   - Transform test data using the same encoder
10. Produce ML-ready training and test datasets

No predictive model was trained.
"""
)
