"""
Phishing URL Detector (beginner project)
----------------------------------------
Tools: Python, pandas, scikit-learn, Streamlit

How to run:
    streamlit run phishing_detector.py

Dataset:
    Kaggle "Phishing Site URLs"
    https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
"""

import os
import re
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# STEP 1: Settings (you can change these)
# ---------------------------------------------------------------
CSV_FILE = "phishing_site_urls.csv"       # name of the dataset file
SAMPLE_SIZE = 50000                       # use only 50k rows so training is fast
SUSPICIOUS_WORDS = ["login", "verify", "secure"]
SUSPICIOUS_TLDS = [".tk", ".xyz"]

# The order of these names = the order of the numbers we feed to the model
FEATURE_NAMES = [
    "url_length",
    "has_https",
    "has_ip",
    "has_at_symbol",
    "num_dots",
    "suspicious_words",
    "suspicious_tld",
]


# ---------------------------------------------------------------
# STEP 2: Feature extraction
# Turn one URL (text) into a list of numbers the model can understand.
# ---------------------------------------------------------------
def extract_features(url):
    url = str(url).strip().lower()

    # Get the domain part (like "example.com") from the URL.
    # If the URL has no http:// or https://, we add one so parsing works.
    to_parse = url if "://" in url else "http://" + url
    try:
        host = urlparse(to_parse).hostname or ""
    except ValueError:
        host = ""

    # 1. Length of the URL (phishing URLs are often very long)
    url_length = len(url)

    # 2. HTTPS = 1, anything else = 0
    has_https = 1 if url.startswith("https") else 0

    # 3. Does the domain look like an IP address? (e.g. 192.168.1.1)
    has_ip = 1 if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host) else 0

    # 4. Is there an "@" symbol? (it can hide the real website)
    has_at_symbol = 1 if "@" in url else 0

    # 5. How many dots? (many dots = many subdomains = suspicious)
    num_dots = url.count(".")

    # 6. How many suspicious words (login, verify, secure) appear?
    suspicious_words = sum(url.count(word) for word in SUSPICIOUS_WORDS)

    # 7. Does the domain end with a suspicious TLD like .tk or .xyz?
    suspicious_tld = 1 if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0

    return [
        url_length,
        has_https,
        has_ip,
        has_at_symbol,
        num_dots,
        suspicious_words,
        suspicious_tld,
    ]


# ---------------------------------------------------------------
# STEP 3: Load the dataset and train the model
# ---------------------------------------------------------------
def load_data():
    df = pd.read_csv(CSV_FILE)

    # Different Kaggle files name their columns differently.
    # We find the URL column and the label column automatically.
    cols = {c.lower(): c for c in df.columns}
    url_col = cols.get("url") or cols.get("urls")
    label_col = cols.get("label") or cols.get("type") or cols.get("status")
    if url_col is None or label_col is None:
        raise ValueError(f"Could not find URL/label columns. Found: {list(df.columns)}")

    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]

    # Turn labels into numbers: 1 = phishing, 0 = safe
    bad_words = ["bad", "phishing", "malicious", "defacement", "malware", "1"]
    df["target"] = df["label"].astype(str).str.lower().isin(bad_words).astype(int)

    # Take a smaller random sample so it runs fast on a normal laptop
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
    return df


# st.cache_resource means: train only once, then reuse the model
@st.cache_resource
def train_model():
    df = load_data()

    # Build the feature table (one row per URL)
    X = pd.DataFrame([extract_features(u) for u in df["url"]], columns=FEATURE_NAMES)
    y = df["target"]

    # 80% of data to learn from, 20% to test on
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest = many decision trees that vote together
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Check how good the model is on data it has never seen
    predictions = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
    }

    # Print results in the terminal
    print("\n=== Model Results ===")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return model, metrics


# ---------------------------------------------------------------
# STEP 4: Streamlit web page
# ---------------------------------------------------------------
st.set_page_config(page_title="Phishing URL Detector", page_icon="🎣")
st.title("🎣 Phishing URL Detector")
st.write("Paste a URL below and the model will guess if it is Safe or Phishing.")

if not os.path.exists(CSV_FILE):
    st.error(
        f"Dataset not found. Put '{CSV_FILE}' in the same folder as this file. "
        "Download it from the Kaggle link at the top of the code."
    )
    st.stop()

with st.spinner("Training the model (only happens the first time)..."):
    model, metrics = train_model()

# Show the model scores
st.subheader("Model scores")
cols = st.columns(4)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value * 100:.1f}%")

# URL input box
st.subheader("Check a URL")
user_url = st.text_input("Enter URL", placeholder="http://secure-login-verify.example.tk/account")

if st.button("Check"):
    if not user_url.strip():
        st.warning("Please enter a URL first.")
    else:
        features = pd.DataFrame([extract_features(user_url)], columns=FEATURE_NAMES)

        # predict_proba gives [chance of safe, chance of phishing]
        safe_chance, phishing_chance = model.predict_proba(features)[0]

        if phishing_chance >= 0.5:
            st.error(f"🚨 Phishing  |  Confidence: {phishing_chance * 100:.1f}%")
        else:
            st.success(f"✅ Safe  |  Confidence: {safe_chance * 100:.1f}%")

        # Show the numbers the model looked at (nice for learning)
        with st.expander("See the features used"):
            st.table(features.T.rename(columns={0: "value"}))

st.caption("Beginner project. Do not rely on this alone to judge real links.")