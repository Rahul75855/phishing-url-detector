# Phishing URL Detector

A beginner-friendly machine learning project that checks if a URL is **Safe** or **Phishing**, with a confidence score, using Python, scikit-learn and Streamlit.

## What is this project?

Phishing is when attackers create fake websites that look real (like a fake bank login page) to steal passwords and money. This project looks at the **URL itself** and predicts whether it is likely a phishing link.

You paste a URL in a simple web app, click **Check**, and get:

* Safe or Phishing
* A confidence percentage
* The exact features the model looked at (good for learning)

## Features

* Trains a **Random Forest** model on a real phishing URL dataset from Kaggle
* Extracts 7 simple features from every URL
* Shows **Accuracy, Precision, Recall and F1 Score**
* Clean web UI built with **Streamlit**
* Whole project is in **one file** with simple comments, easy for beginners

## Features extracted from each URL

|Feature|Why it matters|
|-|-|
|URL length|Phishing URLs are often very long|
|HTTPS or HTTP|Missing HTTPS can be a warning sign|
|IP address in URL|Real sites rarely use raw IPs|
|`@` symbol|Can hide the real destination|
|Number of dots|Many dots usually means many fake subdomains|
|Suspicious words (`login`, `verify`, `secure`)|Common in fake login pages|
|Suspicious TLD (`.tk`, `.xyz`)|Often used for cheap throwaway domains|

## Tech stack

* **Python 3**
* **pandas** for data handling
* **scikit-learn** for the RandomForestClassifier and metrics
* **Streamlit** for the web interface

## Dataset

Kaggle: [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)

Download the CSV, rename it to `phishing_site_urls.csv` if needed, and place it in the same folder as `phishing_detector.py`. The dataset is not included in this repo because of its size and Kaggle's terms.

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/Rahul75855/phishing-url-detector.git
cd phishing-url-detector

# 2. Install requirements
pip install -r requirements.txt

# 3. Put phishing_site_urls.csv in this folder, then run
streamlit run phishing_detector.py
```

The app opens at `http://localhost:XXXX`. The first run trains the model (about 30 to 60 seconds), after that it is cached.

## Example

|URL|Expected result|
|-|-|
|`http://secure-login-verify.example.tk/account`|Phishing|
|`https://www.google.com`|Safe|

## Model results

Trained on a 50,000 URL sample with an 80/20 train-test split.

|Metric|Score|
|-|-|
|Accuracy|70.6%|
|Precision|48.7%|
|Recall|57.2%|
|F1 Score|52.6%|

## Limitations

* Uses only 7 hand-made features, so it can miss tricky phishing URLs
* Looks only at the URL text, not the actual page content
* Can give false alarms on safe links with long or unusual URLs
* This is a learning project, **do not rely on it alone for real security decisions**

## Future improvements

* Add more features (subdomain count, special character count, domain age)
* Use TF-IDF on URL text for better pattern detection
* Compare with other models (Logistic Regression, XGBoost)
* Save the trained model with `joblib` so it loads instantly
* Deploy on Streamlit Community Cloud

## What I learned

* Turning raw text (URLs) into numeric features
* Training and evaluating a classification model
* Understanding precision vs recall in a security problem
* Building a quick ML web app with Streamlit

## License

MIT License. Free to use and learn from.

