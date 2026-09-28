import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

# 1. Fetch dataset
print("Fetching dataset...")
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
df = pd.concat([X, y], axis=1)

print("\n--- DATASET OVERVIEW ---")
print(f"Dataset Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)

# 2. Missing Values & Duplicates
print("\n--- DATA QUALITY CHECK ---")
missing_values = df.isnull().sum()
print("\nMissing Values per column:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "No missing values found.")

duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# 3. Statistical Summary (Numerical Features)
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 4. Data Visualisations (For your screenshots!)

# Set style for plots
sns.set_theme(style="whitegrid")

# Plot 1: Target Variable Distribution (Class Imbalance)
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='y', palette='Set2')
plt.title('Target Variable Distribution (Class Imbalance)')
plt.ylabel('Count')
plt.xlabel('Subscribed to Term Deposit (y)')
plt.tight_layout()
plt.show() # A window will pop up. Take a screenshot, then close it to continue.

# Plot 2: Correlation Heatmap (Numerical features only)
plt.figure(figsize=(10, 8))
# Select only numerical columns for correlation
numerical_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()

# Plot 3: Distribution of 'age' (Checking Skewness/Outliers)
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Age')
plt.tight_layout()
plt.show()

# Plot 4: Boxplot of 'balance' (Checking for Outliers)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['balance'], color='lightgreen')
plt.title('Boxplot of Balance (Checking Outliers)')
plt.tight_layout()
plt.show()

print("\nEDA completed. You can use these printed outputs and graphs for your report screenshots.")
