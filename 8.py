from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
print("Name: Prince Prajapati")
print("Registration Number: 2502030029")
texts = [
    "Win a free prize now",
    "Claim your free reward",
    "Congratulations you won a prize",
    "Free offer limited time",
    "Meeting scheduled for tomorrow",
    "Please submit your assignment",
    "Let us attend the class",
    "The project meeting is today"
]
labels = [
    "Spam", "Spam", "Spam", "Spam",
    "Not Spam", "Not Spam", "Not Spam", "Not Spam"
]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.25, random_state=42, stratify=labels
)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nActual Labels:", list(y_test))
print("Predicted Labels:", list(y_pred))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))

new_message = ["Congratulations, you won a free prize"]
new_message_vector = vectorizer.transform(new_message)
prediction = model.predict(new_message_vector)

print("\nNew Message:", new_message[0])
print("Classification:", prediction[0])

print("\nNaive Bayes classification completed successfully!")