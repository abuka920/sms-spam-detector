# 🛡️ SMS Spam & Smishing Detector

**Live demo:** https://kalif-spam-detector.streamlit.app

Paste a text message and find out if it's spam or a phishing attempt (smishing). It gives a risk score and points out the red flags in plain English, the way a fraud analyst would.

## Results

Tested on 1,035 messages the model never saw during training:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Naive Bayes (picked)** | **98.7%** | **94.7%** | **95.4%** | **0.951** |
| Linear SVM | 98.6% | 96.0% | 92.4% | 0.942 |
| Logistic Regression | 98.5% | 95.3% | 92.4% | 0.938 |

Training compares all three and keeps the one with the best F1 score.

## How it works

**1. Machine learning model**
- Word TF-IDF (1 and 2 word phrases) catches wording like "claim your prize"
- Character TF-IDF (3 to 5 characters) catches tricks like "FR33" and "w1n"

**2. Fraud rules layer**

The training data is from 2012. It has never seen a fake Canada Post fee or a fake e-Transfer deposit. So I added rules for modern smishing: **a link plus a money or account hook** gets flagged as high risk.

Final risk = the higher of the model score and the rule score. Real fraud systems layer rules over models the same way.

| Message | ML model | Rules | Result |
|---|---|---|---|
| "Canada Post: parcel on hold due to unpaid fees. Pay $1.99 here bit.ly/..." | 0.5% | 85% | 🚨 Spam |
| "Hey, running 10 min late. Save me a seat?" | ~0% | 0% | ✅ Legit |

**3. Red flags explained**

Every check lists what looks suspicious: urgency, prize or money, account scares, phone numbers, links.

## Features

- Check a single message with built-in scam examples
- Upload a CSV and check hundreds of messages at once, then download the results
- Model performance table right in the app
- Command line tool

## Stack

`Python` `scikit-learn` `pandas` `Streamlit`

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model trains itself on first run (about 4 seconds).

Command line:

```bash
python spam_model.py                        # retrain and print the scores
python predict.py "You won a free cruise!"  # check one message
python -m pytest                            # run the tests
```

## Project layout

```
app.py              Streamlit web app
spam_model.py       Training, model comparison, rules and red flags
predict.py          Command line checker
data/sms_spam.tsv   Training data (5,574 labelled SMS messages)
tests/              Unit tests
```

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) by Tiago A. Almeida and José María Gómez Hidalgo, from the UCI Machine Learning Repository (CC BY 4.0).

---

Built by **Kalif Abu** · [LinkedIn](https://linkedin.com/in/kalif-abu-262430264)
