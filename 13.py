
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")

iris = load_iris()
X = iris.data
y = iris.target

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": make_pipeline(
        StandardScaler(), SVC()
    )
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-Validation Results:")

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("\nModel:", name)
    print("Fold Scores:", scores.round(3))
    print("Mean Accuracy:", round(scores.mean(), 3))

print("\nPerformance comparison completed successfully!")