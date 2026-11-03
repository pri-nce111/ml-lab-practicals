from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a pipeline with scaling and KNN
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# Define hyperparameters to test
params = {
    "knn__n_neighbors": [3, 5, 7, 9],
    "knn__weights": ["uniform", "distance"]
}

# Perform hyperparameter tuning
grid = GridSearchCV(pipeline, params, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

# Predict using the best model
y_pred = grid.predict(X_test)

print("\nBest Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", round(grid.best_score_, 2))
print("Test Accuracy:", round(accuracy_score(y_test, y_pred), 2))

print("\nKNN classification and hyperparameter tuning completed!")