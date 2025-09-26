# model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

DEFAULT_MODEL_PATH = "phishing.pkl"

def train_and_save(csv_path="phishing.csv", model_path=DEFAULT_MODEL_PATH):
    """Train RandomForest model on phishing.csv and save as pickle"""
    # Load dataset
    data = pd.read_csv(csv_path)

    # Drop unnecessary columns
    if 'Index' in data.columns:
        data = data.drop(['Index'], axis=1)

    # Features and target
    X = data.drop("class", axis=1)
    y = data["class"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model trained and saved as {model_path}")
    return model

# Allow running standalone
if __name__ == "__main__":
    train_and_save() 
