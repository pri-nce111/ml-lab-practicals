from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear")
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Labels:", y_test)
print("Predicted Labels:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("SVM classification completed successfully!")