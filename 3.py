import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
df = pd.read_csv("iris.csv")

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

print("\nOriginal Dataset:")
print(df.head())

# 1. Handling Missing Values
df = df.dropna(subset=["species"])
numeric_cols = df.select_dtypes(include="number").columns
df[numeric_cols] = df[numeric_cols].fillna(
    df[numeric_cols].mean()
)

print("\nMissing Values After Handling:")
print(df.isnull().sum())

# 2. Normalization
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\nNormalized Data:")
print(df.head())

# 3. Label Encoding
encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

print("\nLabel Encoded Data:")
print(df.head())

print("\nPreprocessing Completed Successfully!")