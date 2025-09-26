import pandas as pd
import numpy as np
import pickle
import tldextract
from urllib.parse import urlparse, parse_qs
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Step 1: Load phishing URL dataset
dataset_url = "https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv"
df = pd.read_csv(dataset_url)
df = df.dropna()

# Step 2: Define advanced URL features
feature_cols = [
    'length_url',
    'qty_dot_url',
    'qty_hyphen_url',
    'qty_slash_url',
    'qty_questionmark_url',
    'qty_equal_url',
    'qty_at_url',
    'qty_and_url',
    'qty_exclamation_url',
    'qty_hashtag_url',
    'qty_dollar_url',
    'qty_percent_url',
    'qty_tld_url',
    'qty_dot_domain',
    'domain_length',
    'subdomain_count',
    'subdomain_length',
    'query_count',
    'has_ip',
    'https_presence'
]

# Add placeholder columns for new features in dataset
df['subdomain_count'] = 0
df['subdomain_length'] = 0
df['query_count'] = 0
df['has_ip'] = 0
df['https_presence'] = 1  # Approximated for training

# Log-normalize length_url
df['length_url'] = np.log1p(df['length_url'])

# Advanced feature extractor function
def extract_advanced_url_features(url):
    url = url if url.startswith('http') else 'http://' + url
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    query_params = parse_qs(parsed.query)
    query_count = len(query_params)

    subdomains = ext.subdomain.split('.') if ext.subdomain else []
    subdomain_count = len(subdomains)
    subdomain_length = sum(len(s) for s in subdomains) if subdomains else 0

    has_ip = 1 if re.match(r'^\d+\.\d+\.\d+\.\d+$', ext.domain) else 0

    features = {
        'length_url': np.log1p(len(url)),
        'qty_dot_url': url.count('.'),
        'qty_hyphen_url': url.count('-'),
        'qty_slash_url': url.count('/'),
        'qty_questionmark_url': url.count('?'),
        'qty_equal_url': url.count('='),
        'qty_at_url': url.count('@'),
        'qty_and_url': url.count('&'),
        'qty_exclamation_url': url.count('!'),
        'qty_hashtag_url': url.count('#'),
        'qty_dollar_url': url.count('$'),
        'qty_percent_url': url.count('%'),
        'qty_tld_url': 1 if ext.suffix else 0,
        'qty_dot_domain': ext.domain.count('.'),
        'domain_length': len(ext.domain),
        'subdomain_count': subdomain_count,
        'subdomain_length': subdomain_length,
        'query_count': query_count,
        'has_ip': has_ip,
        'https_presence': 1 if url.startswith('https://') else 0
    }
    return [features[col] for col in feature_cols]

# Step 4: Train/test split and train model
X = df[feature_cols]
y = df['phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)




print("✅ Test accuracy:", clf.score(X_test, y_test))

# Save trained model with feature columns
with open("phishing.pkl", "wb") as f:
    pickle.dump({'model': clf, 'feature_cols': feature_cols}, f)

print("💾 Model saved as phishing.pkl")
