import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load CSV file
df = pd.read_csv("naive_bayes_dataset.csv")

# Separate input and output
X = df["text"]
y = df["sentiment"]

# Convert text into numerical values
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test new sentence
text = ["This movie is amazing"]
text = cv.transform(text)

print("Prediction:", model.predict(text)[0])