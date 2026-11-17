"""Streamlit app for the SMS spam and smishing detector.

Run with:  streamlit run app.py
"""

import json

import pandas as pd
import streamlit as st

from spam_model import METRICS_PATH, load_model, predict

st.set_page_config(page_title="SMS Spam Detector", page_icon="🛡️", layout="centered")


@st.cache_resource(show_spinner="Training the model (first run only)...")
def get_model():
    return load_model()


model = get_model()

st.title("🛡️ SMS Spam & Smishing Detector")
st.caption("Paste a text message. The model scores how likely it is to be spam or a phishing attempt.")

examples = {
    "Pick an example...": "",
    "Fake bank alert": "ALERT: Your account has been suspended due to unusual activity. Verify now at http://secure-bank-verify.co to avoid closure.",
    "Prize scam": "Congratulations! You've WON a $1000 gift card. Claim your prize now, reply YES or call 08712345678.",
    "Normal text": "Hey are we still on for dinner tonight? I can pick you up around 7.",
    "Delivery scam": "Canada Post: your parcel is on hold due to unpaid fees. Pay $1.99 here to release it: bit.ly/cp-redeliver",
}
choice = st.selectbox("Try an example", list(examples))
message = st.text_area("Message", value=examples[choice], height=130)

if st.button("Check message", type="primary") and message.strip():
    st.session_state["result"] = predict(model, message)

# Kept in session state so the result stays on screen while you use the rest of the page
if result := st.session_state.get("result"):
    pct = result["spam_probability"] * 100

    if result["is_spam"]:
        st.error(f"🚨 Likely SPAM ({pct:.1f}% spam probability)")
    else:
        st.success(f"✅ Looks legit ({pct:.1f}% spam probability)")
    st.progress(min(int(pct), 100))
    st.caption(f"ML model: {result['model_probability'] * 100:.1f}% · Fraud rules: {result['rule_score'] * 100:.0f}% · Final risk is the higher of the two")

    if result["red_flags"]:
        st.markdown("**Red flags found**")
        for category, hits in result["red_flags"].items():
            st.markdown(f"- **{category}:** {', '.join(hits)}")
    else:
        st.markdown("No common red flags found.")

st.divider()

with st.expander("📂 Check a whole file of messages"):
    upload = st.file_uploader("CSV with a column named 'text'", type="csv")
    if upload:
        try:
            df = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Couldn't read that CSV. Put quotes around messages that contain commas. ({exc})")
            st.stop()
        if "text" not in df.columns:
            st.error("The CSV needs a column named 'text'.")
        else:
            probs = [predict(model, t)["spam_probability"] for t in df["text"].fillna("").astype(str)]
            df["spam_probability"] = probs
            df["prediction"] = ["spam" if p >= 0.5 else "ham" for p in probs]
            st.write(f"Flagged **{(df['prediction'] == 'spam').sum()}** of {len(df)} messages as spam.")
            st.dataframe(df, use_container_width=True)
            st.download_button("📥 Download results", df.to_csv(index=False), "spam_results.csv", "text/csv")

with st.expander("📊 Model performance"):
    if METRICS_PATH.exists():
        summary = json.loads(METRICS_PATH.read_text())
        st.write(f"Best model: **{summary['best_model']}**, tested on {summary['test_size']} held out messages.")
        table = pd.DataFrame(summary["results"]).T[["accuracy", "precision", "recall", "f1"]]
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)
