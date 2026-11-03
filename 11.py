from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

iris = load_iris()
X = iris.data

print("\nOriginal Data Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nReduced Data Shape:", X_pca.shape)
print("\nFirst Five Rows:")
print(X_pca[:5])

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nPCA completed successfully!")