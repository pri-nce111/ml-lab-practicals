import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create a simple dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [12, 18, 25, 35, 42, 50, 58, 65, 75, 85]
}

df = pd.DataFrame(data)

print("Name: Prince Prajapati")
print("Registration Number: 2502030029")
print("\nDataset:")
print(df)

# Define input and output
X = df[["Hours"]]
y = df["Marks"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict marks
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print("\nActual Marks:", list(y_test))
print("Predicted Marks:", y_pred.round(2))
print("Mean Squared Error:", round(mse, 2))

print("\nLinear Regression implemented successfully!")