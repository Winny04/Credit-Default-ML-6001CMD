# ============================================================
# 6001CMD MACHINE LEARNING CW1
# TASK 1 + TASK 3: DATASET INSPECTION AND DATA QUALITY ANALYSIS
# Dataset: UCI Default of Credit Card Clients
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 0. SETTINGS
# ============================================================

DATASET_PATH = "default of credit card clients.csv"
EXCEL_SOURCE = "default of credit card clients.xls"   # original UCI download
TARGET = "default payment next month"

# Create a folder for clean figures used in the report
os.makedirs("figures", exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# Task 1: Dataset Characteristics
# ============================================================

# The UCI CSV contains an extra first header row (X1...X23, Y),
# therefore header=1 is required.
# Load the CSV if present; otherwise read the original UCI .xls directly
# with pandas (programmatic loading only, Excel software is not used).
if os.path.exists(DATASET_PATH):
    df = pd.read_csv(DATASET_PATH, header=1)
else:
    df = pd.read_excel(EXCEL_SOURCE, header=1)

print("\n" + "=" * 60)
print("TASK 1 - BASIC DATASET INFORMATION")
print("=" * 60)

# Do NOT use print(df.info()) because it prints an unnecessary "None"
df.info()

print(f"\nTotal Instances: {df.shape[0]}")
print(f"Total Raw Columns: {df.shape[1]}")

# Raw dataset:
# 1 ID + 23 predictive features + 1 target
predictor_count = df.shape[1] - 2

print(f"Identifier Columns: 1")
print(f"Predictive Features: {predictor_count}")
print(f"Target Variables: 1")

print("\nColumn Names:")
for col in df.columns:
    print(f"- {col}")


# ============================================================
# 2. TARGET VARIABLE
# Task 1: Identify Target
# ============================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE")
print("=" * 60)

print(f"Target Variable: {TARGET}")

target_counts = df[TARGET].value_counts().sort_index()
target_percentages = (
    df[TARGET].value_counts(normalize=True).sort_index() * 100
)

print("\nTarget Counts:")
print(target_counts)

print("\nTarget Percentages:")
print(target_percentages.round(2))


# ============================================================
# 3. MISSING VALUES
# Task 3: Data Quality
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 - MISSING VALUES")
print("=" * 60)

missing_by_column = df.isnull().sum()
total_missing = missing_by_column.sum()

print(f"Total Missing Values: {total_missing}")

if total_missing > 0:
    print("\nMissing Values by Column:")
    print(missing_by_column[missing_by_column > 0])
else:
    print("No missing values were detected.")


# ============================================================
# 4. DUPLICATE / REPEATED PROFILE ANALYSIS
# Important:
# Exact duplicate rows WITH ID and repeated profiles WITHOUT ID
# are NOT the same thing.
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE AND REPEATED PROFILE CHECK")
print("=" * 60)

exact_duplicates = df.duplicated().sum()

# Remove ID only for diagnostic comparison.
# We DO NOT delete these records automatically.
df_without_id = df.drop(columns=["ID"])
repeated_profiles = df_without_id.duplicated().sum()

print(f"Exact Duplicate Rows (including ID): {exact_duplicates}")
print(f"Repeated Profiles (excluding ID): {repeated_profiles}")

print(
    "\nInterpretation: repeated profiles after removing ID are not "
    "automatically treated as erroneous duplicates because different "
    "clients may legitimately have identical recorded characteristics."
)


# ============================================================
# 5. DATA INCONSISTENCY / NOISE
# EDUCATION, MARRIAGE and Repayment Status
# ============================================================

print("\n" + "=" * 60)
print("DATA INCONSISTENCY CHECKS")
print("=" * 60)

print(
    "Unique values in EDUCATION:",
    sorted(df["EDUCATION"].unique())
)

print(
    "Unique values in MARRIAGE:",
    sorted(df["MARRIAGE"].unique())
)

payment_status_cols = [
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6"
]

print("\nRepayment Status Codes:")

for col in payment_status_cols:
    print(f"{col}: {sorted(df[col].unique())}")


# ============================================================
# 6. CLASS IMBALANCE
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df[TARGET].value_counts().sort_index()
class_percentages = (
    df[TARGET].value_counts(normalize=True).sort_index() * 100
)

class_summary = pd.DataFrame({
    "Count": class_counts,
    "Percentage (%)": class_percentages.round(2)
})

print(class_summary)

# Plot original class distribution
plt.figure(figsize=(7, 5))

plt.bar(
    ["0 = Non-Default", "1 = Default"],
    class_counts.values
)

plt.title("Class Distribution: Default vs Non-Default")
plt.xlabel("Target Class")
plt.ylabel("Number of Clients")

# Add count values above bars
for i, value in enumerate(class_counts.values):
    plt.text(
        i,
        value + 300,
        f"{value:,}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "figures/class_distribution_original.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 7. DESCRIPTIVE STATISTICS
# Required statistical-summary evidence
# ============================================================

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

financial_cols = [
    "LIMIT_BAL",
    "AGE",
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

descriptive_stats = df[financial_cols].describe().T

print(descriptive_stats)

# Save for Appendix / report table if required
descriptive_stats.to_csv(
    "figures/descriptive_statistics.csv"
)


# ============================================================
# 8. FEATURE DISTRIBUTIONS AND SKEWNESS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS AND SKEWNESS")
print("=" * 60)

limit_skew = df["LIMIT_BAL"].skew()
age_skew = df["AGE"].skew()

print(f"LIMIT_BAL Skewness: {limit_skew:.2f}")
print(f"AGE Skewness: {age_skew:.2f}")

plt.figure(figsize=(12, 5))

# LIMIT_BAL
plt.subplot(1, 2, 1)

plt.hist(
    df["LIMIT_BAL"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Credit Limit (LIMIT_BAL)")
plt.xlabel("Credit Limit (NT$)")
plt.ylabel("Frequency")


# AGE
plt.subplot(1, 2, 2)

plt.hist(
    df["AGE"],
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")


plt.tight_layout()

plt.savefig(
    "figures/feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. OUTLIER ANALYSIS
# IQR-based statistical evidence + boxplot
# ============================================================

print("\n" + "=" * 60)
print("OUTLIER ANALYSIS USING IQR")
print("=" * 60)

bill_cols = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

outlier_results = []

for col in bill_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    outlier_results.append({
        "Feature": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "IQR-Defined Extreme Observations": outlier_count,
        "Minimum": df[col].min(),
        "Maximum": df[col].max()
    })


outlier_table = pd.DataFrame(outlier_results)

print(outlier_table.to_string(index=False))

outlier_table.to_csv(
    "figures/outlier_summary.csv",
    index=False
)


# Boxplot
plt.figure(figsize=(11, 6))

df.boxplot(
    column=bill_cols
)

plt.title(
    "Boxplot of Monthly Bill Amounts "
    "(IQR-Defined Extreme Observations)"
)

plt.xlabel("Billing Feature")
plt.ylabel("Amount (NT$)")

plt.tight_layout()

plt.savefig(
    "figures/bill_amount_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9b. SKEWNESS AND OUTLIERS OF PAYMENT AND CREDIT-LIMIT VARIABLES
# Evidence for the skewness table and the second IQR table
# ============================================================

print("\n" + "=" * 60)
print("SKEWNESS OF MONETARY VARIABLES")
print("=" * 60)

pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]

skew_table = df[bill_cols + pay_amt_cols].skew().round(2)
print("Skewness of monetary variables:")
print(skew_table)

print("\nShare of zero values in PAY_AMT (%):")
print((df[pay_amt_cols] == 0).mean().mul(100).round(1))

rows = []
for col in ["LIMIT_BAL"] + pay_amt_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    n = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    rows.append({
        "Feature": col,
        "IQR Extremes": n,
        "% of clients": round(n / len(df) * 100, 2),
        "Max": df[col].max()
    })

other_outliers = pd.DataFrame(rows)
print("\nIQR extremes (LIMIT_BAL and PAY_AMT):")
print(other_outliers.to_string(index=False))

skew_table.to_csv("figures/skewness_monetary.csv")
other_outliers.to_csv("figures/outlier_summary_payment.csv", index=False)


# ============================================================
# 10. CORRELATION AND MULTICOLLINEARITY
# ============================================================

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# ID is excluded because it is only an identifier
corr_matrix = df.drop(columns=["ID"]).corr()

# Focus on correlations among BILL_AMT variables
bill_corr = df[bill_cols].corr()

# Extract unique feature pairs
correlation_pairs = []

for i in range(len(bill_cols)):
    for j in range(i + 1, len(bill_cols)):

        feature_1 = bill_cols[i]
        feature_2 = bill_cols[j]

        correlation = bill_corr.loc[
            feature_1,
            feature_2
        ]

        correlation_pairs.append({
            "Feature 1": feature_1,
            "Feature 2": feature_2,
            "Pearson r": correlation
        })


correlation_table = pd.DataFrame(
    correlation_pairs
).sort_values(
    by="Pearson r",
    ascending=False
)

print("\nStrongest BILL_AMT Correlations:")
print(
    correlation_table.head(10).to_string(
        index=False,
        formatters={
            "Pearson r": "{:.3f}".format
        }
    )
)

correlation_table.to_csv(
    "figures/bill_amount_correlations.csv",
    index=False
)


# Full correlation heatmap
plt.figure(figsize=(14, 11))

sns.heatmap(
    corr_matrix,
    annot=False,
    cmap="coolwarm",
    linewidths=0.4,
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "figures/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 11. CORRELATION WITH THE TARGET
# Task 3: Correlation among variables (feature relevance)
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 - CORRELATION WITH TARGET")
print("=" * 60)

target_corr = (
    df.drop(columns=["ID"])
    .corr()[TARGET]
    .drop(TARGET)
    .sort_values(key=abs, ascending=False)
    .round(3)
)

print(target_corr.head(10).to_string())
target_corr.to_csv("figures/target_correlation.csv", header=["Pearson r"])

plt.figure(figsize=(8, 7))
target_corr.sort_values().plot(kind="barh")
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Pearson Correlation of Each Feature with Default")
plt.xlabel("Pearson r")
plt.tight_layout()
plt.savefig("figures/target_correlation.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 12. NOISE CHECKS
# Task 3: Noise / logically suspicious records
#
# These rules flag records whose values contradict each other.
# They are evidence of possible noise, NOT proof of error.
# ============================================================

print("\n" + "=" * 60)
print("TASK 3 - NOISE / SUSPICIOUS RECORD CHECKS")
print("=" * 60)

bill_cols_all = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols_all = [f"PAY_AMT{i}" for i in range(1, 7)]

noise_checks = {
    # PAY_0 = -2 is commonly read as "no consumption",
    # yet the matching bill (BILL_AMT1) is non-zero.
    "PAY_0 = -2 but BILL_AMT1 != 0":
        ((df["PAY_0"] == -2) & (df["BILL_AMT1"] != 0)).sum(),
    # Bill exceeds the granted credit limit.
    "BILL_AMT1 > LIMIT_BAL":
        (df["BILL_AMT1"] > df["LIMIT_BAL"]).sum(),
    # Any negative bill amount (credit balance).
    "Any negative BILL_AMT":
        (df[bill_cols_all] < 0).any(axis=1).sum(),
    # All six bills and payments are zero (inactive account).
    "All BILL_AMT and PAY_AMT = 0":
        ((df[bill_cols_all + pay_amt_cols_all] == 0).all(axis=1)).sum(),
    # Inactive account but labelled as default (possible label noise).
    "Inactive account labelled default":
        (((df[bill_cols_all + pay_amt_cols_all] == 0).all(axis=1))
         & (df[TARGET] == 1)).sum(),
    # Repeated profiles (excluding ID) with conflicting labels.
    "Identical features, conflicting label":
        (df.drop(columns=["ID", TARGET])
         .assign(_y=df[TARGET])
         .groupby(list(df.drop(columns=["ID", TARGET]).columns))["_y"]
         .nunique()
         .gt(1)
         .sum()),
}

noise_table = pd.DataFrame(
    {
        "Records": pd.Series(noise_checks),
        "% of clients": (pd.Series(noise_checks) / len(df) * 100).round(2),
    }
)

print(noise_table.to_string())
noise_table.to_csv("figures/noise_checks.csv")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)

print("Dataset successfully analysed using Python.")
print("No Excel-based EDA was used.")
print("Clean figures have been saved in the 'figures' folder.")

print("\nImportant findings:")
print(f"- Instances: {df.shape[0]}")
print(f"- Raw columns: {df.shape[1]}")
print(f"- Predictive features: {predictor_count}")
print(f"- Missing values: {total_missing}")
print(f"- Exact duplicate rows: {exact_duplicates}")
print(f"- Repeated profiles excluding ID: {repeated_profiles}")
print(
    f"- Non-default class: "
    f"{class_percentages.loc[0]:.2f}%"
)
print(
    f"- Default class: "
    f"{class_percentages.loc[1]:.2f}%"
)
print(f"- LIMIT_BAL skewness: {limit_skew:.2f}")
print(f"- AGE skewness: {age_skew:.2f}")

