import pandas as pd
import numpy as np

NAME = "Prince"
REG_NO = "2502030029"
print("Name:", NAME, "| Register Number:", REG_NO)

# Load CSV using Pandas
df = pd.read_csv("iris.csv")
print("\nFirst 5 rows:")
print(df.head())
print("\nShape:", df.shape)

# Load CSV using NumPy (numeric columns only)
arr = np.loadtxt("iris.csv", delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))
print("\nNumPy array shape:", arr.shape)
print(arr[:5])