# model.py
import pandas as pd
import re
import pickle
import tldextract
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DEFAULT_MODEL_PATH = "phishing.pkl"
CSV_PATH = "phishing.csv"

# ---------- Feature Engineering ----------
def extract_features(url: str):
    parsed = urlparse(url)
    domain_info = tldextract.extract(url)

    return [
        len(url),                                   # URL length
        url.count("."),                             # Dots in URL
        url.count("@"),                             # @ symbol
        url.count("-"),                             # Hyphens
        url.count("="),                             # Equal signs
        1 if url.startswith("https") else 0,        # HTTPS flag
        len(parsed.netloc),                         # Domain length
        len(parsed.path),                           # Path length
        1 if re.search(r"\d", url) else 0,          # Digit present
        1 if re.search(r"//", parsed.path) else 0,  # Redirect
        1 if domain_info.suffix in ["com", "org", "net"] else 0, # Popular TLD
    ]

FEATURE_NAMES = [
    "url_length", "dot_count", "at_count", "hyphen_count", "equal_count",
    "https_flag", "domain_length", "path_length", "digit_in_url",
    "redirect_path", "popular_tld"
]

# ---------- Train and Save ----------
def train_and_save(model_path=DEFAULT_MODEL_PATH):
    print("🔄 Training phishing detection model...")

    # Load dataset
    df = pd.read_csv(CSV_PATH)

    # Expect dataset with ["url", "label"] (1 = phishing, 0 = legitimate)
    df["features"] = df["url"].apply(extract_features)
    X = pd.DataFrame(df["features"].to_list(), columns=FEATURE_NAMES)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print("✅ Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved at {model_path}")

# ---------- Predict ----------
def load_model(model_path=DEFAULT_MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def predict_url(url: str, model):
    feats = extract_features(url)
    return model.predict([feats])[0]
