import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tldextract
from urllib.parse import urlparse
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('phishing.csv')
df = df.dropna()

# Define feature extraction function (for URLs specifically) if needed
# If dataset already contains engineered features, you may skip this step

# Assuming the dataset columns except 'class' are features
feature_cols = [col for col in df.columns if col != 'class']

X = df[feature_cols]
y = df['class']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Phish'], yticklabels=['Legit', 'Phish'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Save model and feature columns for later use
with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
    
