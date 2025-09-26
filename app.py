# app.py
from flask import Flask, render_template, request, jsonify
import os
import pickle
import numpy as np
import re
from urllib.parse import urlparse

# local helper to train/save model if missing
from model import train_and_save, DEFAULT_MODEL_PATH

MODEL_PATH = DEFAULT_MODEL_PATH

# ---------------- Trusted Domains ----------------
TRUSTED_DOMAINS = [
    "google.com", "github.com", "instagram.com", "microsoft.com", "apple.com",
    "facebook.com", "amazon.com", "wikipedia.org", "yahoo.com"
]

# ---------------- Suspicious Keywords ----------------
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "update", "account", "banking",
    "confirm", "paypal", "signin", "password", "credential"
]

# ---------------- Feature columns (same order as model.feature_columns expected) ----------------
FEATURE_COLS = [
    'length_url','qty_dot_url','qty_hyphen_url','qty_slash_url','qty_questionmark_url','qty_equal_url',
    'qty_at_url','qty_and_url','qty_exclamation_url','qty_hashtag_url','qty_dollar_url','qty_percent_url',
    'qty_tld_url','qty_dot_domain','domain_length','subdomain_count','subdomain_length','query_count',
    'has_ip','https_presence'
]

def ensure_model():
    """Load model; if missing, train via model.train_and_save(). Returns (model, feature_columns)."""
    if not os.path.exists(MODEL_PATH):
        print(f"{MODEL_PATH} not found — training model now.")
        train_and_save(out_path=MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)
    model = obj.get("model")
    feature_columns = obj.get("feature_columns", FEATURE_COLS)
    return model, feature_columns

def extract_features_from_url(url):
    u = url.strip()
    if not (u.startswith('http://') or u.startswith('https://')):
        u = 'http://' + u
    parsed = urlparse(u)
    path = parsed.path or ""
    query = parsed.query or ""
    host = parsed.netloc or ""
    host = host.split(":")[0]

    length_url = len(u)
    qty_dot_url = u.count(".")
    qty_hyphen_url = u.count("-")
    qty_slash_url = u.count("/")
    qty_questionmark_url = u.count("?")
    qty_equal_url = u.count("=")
    qty_at_url = u.count("@")
    qty_and_url = u.count("&")
    qty_exclamation_url = u.count("!")
    qty_hashtag_url = u.count("#")
    qty_dollar_url = u.count("$")
    qty_percent_url = u.count("%")

    common_tlds = ['.com', '.net', '.org', '.info', '.io', '.gov', '.edu', '.co', '.uk', '.ru', '.cn']
    qty_tld_url = sum(u.count(tld) for tld in common_tlds)
    qty_dot_domain = host.count(".")
    domain_parts = host.split(".")
    domain_length = len(domain_parts[0]) if domain_parts and domain_parts[0] else 0
    subdomain_count = max(0, len(domain_parts) - 2) if len(domain_parts) >= 2 else 0
    subdomain_length = sum(len(p) for p in domain_parts[:-2]) if subdomain_count > 0 else 0
    query_count = len([part for part in query.split("&") if part]) if query else 0
    has_ip = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host) else 0
    https_presence = 1 if parsed.scheme == "https" else 0

    feats = {
        'length_url': length_url,
        'qty_dot_url': qty_dot_url,
        'qty_hyphen_url': qty_hyphen_url,
        'qty_slash_url': qty_slash_url,
        'qty_questionmark_url': qty_questionmark_url,
        'qty_equal_url': qty_equal_url,
        'qty_at_url': qty_at_url,
        'qty_and_url': qty_and_url,
        'qty_exclamation_url': qty_exclamation_url,
        'qty_hashtag_url': qty_hashtag_url,
        'qty_dollar_url': qty_dollar_url,
        'qty_percent_url': qty_percent_url,
        'qty_tld_url': qty_tld_url,
        'qty_dot_domain': qty_dot_domain,
        'domain_length': domain_length,
        'subdomain_count': subdomain_count,
        'subdomain_length': subdomain_length,
        'query_count': query_count,
        'has_ip': has_ip,
        'https_presence': https_presence
    }
    # return feature vector in the specified order
    return [feats[c] for c in FEATURE_COLS], host, path, query

# Load or train model
model, model_feature_columns = ensure_model()

# ---------------- Flask App ----------------
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def index():
    result = None
    score = None
    url = ""
    is_legit = None
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        if url:
            feats, host, path, query = extract_features_from_url(url)

            # ---- Rule 1: Trusted domain override ----
            for trusted in TRUSTED_DOMAINS:
                if host.endswith(trusted):
                    result = f"Legitimate ✅ (trusted: {trusted})"
                    score = 0.0
                    is_legit = True
                    return render_template("index.html", result=result, score=score, url=url, is_legit=is_legit)

            # ---- Rule 2: Suspicious keyword boost ----
            for word in SUSPICIOUS_KEYWORDS:
                if word in url.lower():
                    result = f"Phishing 🚨 (keyword: {word})"
                    score = 1.0
                    is_legit = False
                    return render_template("index.html", result=result, score=score, url=url, is_legit=is_legit)

            # ---- Otherwise, use ML model ----
            # apply same transform as used in your previous logic: log1p on length_url
            feats_copy = feats.copy()
            feats_copy[0] = np.log1p(feats_copy[0])
            X = np.array(feats_copy).reshape(1, -1)

            try:
                pred = int(model.predict(X)[0])
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = float(round(model.predict_proba(X)[0][1], 4))
                    score = prob
                result = "Phishing 🚨" if pred == 1 else "Legitimate ✅"
                is_legit = (pred == 0)
            except Exception as e:
                result = f"Error during prediction: {e}"
                is_legit = None

    return render_template("index.html", result=result, score=score, url=url, is_legit=is_legit)

if __name__ == "__main__":
    # development server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
