import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load CSV file
df = pd.read_csv("iris_dataset.csv")

# Select features
X = df.drop("species", axis=1)

# Elbow Method
inertia = []

for k in range(1, 8):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    inertia.append(model.inertia_)

# Plot Elbow Graph
plt.plot(range(1, 8), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Apply K-Means (K=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# Display results
print("Cluster Labels:")
print(y_pred)

print("Cluster Centers:")
print(kmeans.cluster_centers_)

# Plot clusters
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_pred)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()