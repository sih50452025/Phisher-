# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv("phishing.csv")

# Drop unnecessary columns if present
if 'Index' in data.columns:
    data = data.drop(['Index'], axis=1)

# Features and target
X = data.drop("class", axis=1)
y = data["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model (dictionary format so we can expand later if needed)
with open("phishing.pkl", "wb") as f:
    pickle.dump({"model": model}, f)

print("✅ Model trained and saved as phishing.pkl")
