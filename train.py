# Step 1: Bring in our tools
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Step 2: Read the CSV file (open the lunchbox)
data = pd.read_csv("House_Price_Prediction_Dataset.csv")

# Step 3: Turn words into numbers (computers only understand numbers!)
# Example: "Downtown" -> 0, "Suburban" -> 1, etc.
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"])

# Step 4: Separate the "question" from the "answer"
# X = all the house info (bedrooms, area, etc.) = the question
# y = the price = the answer we want to predict
X = data.drop(columns=["Id", "Price"])
y = data["Price"]

# Step 5: Split data into "practice" and "quiz" sets
# 80% to practice on, 20% to quiz the model on afterward
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Create the model (an empty brain) and teach it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Quiz the model with data it hasn't seen
predictions = model.predict(X_test)

error = mean_absolute_error(y_test, predictions)
print(f"On average, our model is off by: ${error:,.2f}")

# Step 8: Save the trained brain into a file
joblib.dump(model, "house_price_model.pkl")
print("Model saved as house_price_model.pkl ✅")