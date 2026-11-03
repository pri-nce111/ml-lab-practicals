import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
print("Student Name: Prince Prajapati")
print("Registration Number: 2502030029")
print("=" * 50)
df = pd.read_csv("iris.csv")
X = df.drop("species", axis=1)
y = df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation="relu",
    solver="adam",
    max_iter=2000,
    random_state=42
)
#neural network
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nNeural Network Results")
print("-" * 50)
print("Model: MLPClassifier")
print("Hidden Layer: 10 neurons")
print("Test Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nSample Predictions:")
for actual, predicted in zip(y_test.iloc[:10], y_pred[:10]):
    print(f"Actual: {actual:12} Predicted: {predicted}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
