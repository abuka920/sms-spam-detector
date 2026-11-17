import json

from spam_model import METRICS_PATH, load_model, predict, red_flags


def test_model_is_accurate():
    load_model()
    summary = json.loads(METRICS_PATH.read_text())
    best = summary["results"][summary["best_model"]]
    assert best["accuracy"] > 0.97
    assert best["f1"] > 0.92


def test_obvious_spam_and_ham():
    model = load_model()
    assert predict(model, "WINNER!! You have won a £1000 cash prize. Call 09061701461 to claim now")["is_spam"]
    assert not predict(model, "Hey, running 10 min late. Save me a seat?")["is_spam"]
    assert not predict(model, "I know you're busy, call me when you can")["is_spam"]


def test_modern_smishing_caught_by_rules():
    model = load_model()
    msg = "Canada Post: your parcel is on hold due to unpaid fees. Pay $1.99 here bit.ly/cp-redeliver"
    assert predict(model, msg)["is_spam"]


def test_no_false_flags_inside_words():
    assert red_flags("I know, I won't be there") == {}


def test_red_flags():
    flags = red_flags("URGENT: your bank account is suspended, verify at http://x.co")
    assert "Urgency" in flags
    assert "Account or payment scare" in flags
    assert "Link" in flags
