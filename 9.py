import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

data = {
    "Height": [150, 155, 160, 162, 170, 175, 180, 185],
    "Weight": [45, 48, 50, 55, 65, 70, 75, 80]
}

df = pd.DataFrame(data)

print("\nOriginal Dataset:")
print(df)
scaler = StandardScaler()
X = scaler.fit_transform(df)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

print("\nClustered Dataset:")
print(df)
plt.scatter(
    df["Height"], df["Weight"],
    c=df["Cluster"], cmap="viridis", s=100
)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("K-Means Clustering")
plt.show()

print("\nK-Means clustering completed successfully!")