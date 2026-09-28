import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
bank_marketing = fetch_ucirepo(id=222)

# Extract features and targets as pandas dataframes
X = bank_marketing.data.features
y = bank_marketing.data.targets

# Combine them to check the full shape
df = pd.concat([X, y], axis=1)

print("Dataset Shape:", df.shape)
print("\nTarget Variable Distribution:")
print(df["y"].value_counts(normalize=True))