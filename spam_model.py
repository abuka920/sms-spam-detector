"""Train, save, load and explain the SMS spam / smishing classifier."""

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "sms_spam.tsv"
MODEL_PATH = ROOT / "models" / "spam_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANDOM_STATE = 42


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], quoting=3)
    df = df.dropna().drop_duplicates(subset="text")
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df


def make_features():
    """Word n-grams catch phrases like 'claim your prize'.
    Character n-grams catch tricks like 'FR33', 'w1n' and odd spacing."""
    return FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)),
    ])


CANDIDATES = {
    "Naive Bayes": lambda: MultinomialNB(alpha=0.1),
    "Logistic Regression": lambda: LogisticRegression(C=20, max_iter=2000, class_weight="balanced"),
    "Linear SVM": lambda: CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=5),
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "report": classification_report(y_test, pred, target_names=["ham", "spam"]),
    }


def train(verbose=True):
    """Compare three models, keep the one with the best F1 on the held out test set."""
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["is_spam"], test_size=0.2, stratify=df["is_spam"], random_state=RANDOM_STATE
    )

    results, best_name, best_model = {}, None, None
    for name, make_clf in CANDIDATES.items():
        model = Pipeline([("features", make_features()), ("clf", make_clf())])
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = {k: v for k, v in metrics.items() if k != "report"}
        if verbose:
            print(f"{name:<20} acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
                  f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}")
        if best_model is None or metrics["f1"] > results[best_name]["f1"]:
            best_name, best_model = name, model

    if verbose:
        print(f"\nBest model: {best_name}\n")
        print(evaluate(best_model, X_test, y_test)["report"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    summary = {
        "best_model": best_name,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "results": results,
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2))
    return best_model, summary


def load_model():
    """Load the saved model, or train one if it doesn't exist yet (takes a few seconds)."""
    if not MODEL_PATH.exists():
        train(verbose=False)
    return joblib.load(MODEL_PATH)


# ---------- Explaining a prediction ----------

RED_FLAGS = {
    "Urgency": ["urgent", "immediately", "now", "expires", "expiring", "final notice", "act fast",
                "today only", "asap", "within 24 hours", "last chance"],
    "Prize or money": ["win", "won", "winner", "prize", "cash", "free", "reward", "claim", "gift card",
                       "e-transfer", "deposit", "refund"],
    "Account or payment scare": ["account", "suspended", "locked", "verify", "unusual activity", "bank",
                                 "payment", "unpaid", "fees", "on hold", "fraud", "password", "pin"],
    "Call or text a number": ["call", "reply", "txt", "text back"],
}
LINK_RE = re.compile(r"(https?://|www\.|\b[a-z0-9-]+\.(?:com|co|ca|net|info|xyz|ly|link|top)\b)", re.I)
LONG_NUMBER_RE = re.compile(r"\d[\d\s-]{8,}\d")


def _has_phrase(text, phrase):
    return re.search(r"(?<![a-z0-9'])" + re.escape(phrase) + r"(?![a-z0-9'])", text) is not None


def red_flags(message):
    """Plain-English warning signs, the kind a fraud analyst would point out."""
    lower = message.lower()
    found = {cat: sorted({p for p in phrases if _has_phrase(lower, p)}) for cat, phrases in RED_FLAGS.items()}
    if re.search(r"[$£€]\s?\d", message):
        found["Prize or money"].append("dollar amount")
    if LONG_NUMBER_RE.search(message):
        found["Call or text a number"].append("long number")
    links = LINK_RE.findall(message)
    if links:
        found["Link"] = sorted({l.lower() for l in links})
    return {cat: hits for cat, hits in found.items() if hits}


def rule_score(flags):
    """Smishing rules. The classic pattern is a link plus a money or account hook."""
    if "Link" in flags and ({"Prize or money", "Account or payment scare"} & flags.keys()):
        return 0.85
    return min(0.2 * len(flags), 0.6)


def predict(model, message):
    """Final risk is the higher of the ML score and the rule score.

    The model learned from a 2012 SMS dataset, so it misses newer scams like fake
    delivery fees and e-Transfer deposits. The rules cover that gap, the same way
    real fraud systems layer rules on top of models.
    """
    model_proba = float(model.predict_proba([message])[0][1])
    flags = red_flags(message)
    risk = max(model_proba, rule_score(flags))
    return {
        "is_spam": risk >= 0.5,
        "spam_probability": round(risk, 4),
        "model_probability": round(model_proba, 4),
        "rule_score": rule_score(flags),
        "red_flags": flags,
    }


if __name__ == "__main__":
    train()
