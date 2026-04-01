import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
import time
from sklearn.metrics import accuracy_score

# ---------------------------
# Fix torch bug
# ---------------------------
try:
    del torch._classes
except:
    pass

warnings.filterwarnings("ignore")

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("Customer Review Sentiment Analyzer & Auto-Responder")

# ---------------------------
# Email Config
# ---------------------------
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "spkincident@gmail.com"
SENDER_PASSWORD = st.secrets["email_password"]

def send_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# ---------------------------
# Session
# ---------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

# ---------------------------
# Upload
# ---------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.stop()

if "Review_text" not in df.columns:
    st.error("Missing Review_text column")
    st.stop()

df = df.head(100)

# ---------------------------
# Models
# ---------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return sentiment, tokenizer, model

sentiment_pipeline, tokenizer, model = load_models()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ---------------------------
# Threshold
# ---------------------------
NEGATIVE_THRESHOLD = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.7)

# ---------------------------
# Functions
# ---------------------------
def analyze(texts):
    results = sentiment_pipeline(texts, return_all_scores=True)
    labels, scores = [], []
    for r in results:
        top = max(r, key=lambda x: x['score'])
        labels.append(label_map[top['label']])
        scores.append(round(top['score'], 2))
    return labels, scores

def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."

    prompt = f"""
You are a professional customer support agent.

Write a polite, empathetic, short response:
- Acknowledge issue
- Apologize
- Offer help

Review: {review}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    return f"Thank you for your review. {reply}"

# ---------------------------
# Process
# ---------------------------
if not st.session_state.processed:

    labels, scores = analyze(df["Review_text"].tolist())

    responses = []
    times = []

    for i, row in df.iterrows():
        start = time.time()
        responses.append(generate_response(labels[i], row["Review_text"]))
        times.append(time.time() - start)

    df["Sentiment"] = labels
    df["Confidence"] = scores
    df["Response"] = responses
    df["Processing_Time_sec"] = times
    df["Email_Status"] = "Pending"

    st.session_state.df = df
    st.session_state.processed = True

df = st.session_state.df

# ---------------------------
# Email Trigger
# ---------------------------
df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if r["Sentiment"] == "Negative" and r["Confidence"] >= NEGATIVE_THRESHOLD else "No",
    axis=1
)

# ---------------------------
# Metrics
# ---------------------------
total = len(df)
triggered = len(df[df["Email_Trigger"] == "Yes"])

st.metric("Emails to Send", triggered)
st.metric("Trigger Rate", f"{(triggered/total)*100:.1f}%")

# ---------------------------
# Toggle
# ---------------------------
show_only = st.checkbox("Show only triggered rows")

display_df = df[df["Email_Trigger"] == "Yes"] if show_only else df

# ---------------------------
# Highlight
# ---------------------------
def highlight(row):
    if row["Email_Trigger"] == "Yes":
        return ['background-color: #ffcccc'] * len(row)
    elif row["Sentiment"] == "Negative":
        return ['background-color: #ffe6e6'] * len(row)
    else:
        return [''] * len(row)

st.dataframe(display_df.style.apply(highlight, axis=1))

st.caption("🔴 Email Triggered | 🌸 Negative")

# ---------------------------
# Bulk Send
# ---------------------------
if st.button("🚀 Send All Emails"):
    for idx, row in df[df["Email_Trigger"] == "Yes"].iterrows():
        if row["Email_Status"] == "Sent":
            continue

        body = f"""
<h3>Customer Review</h3>
<p>{row['Review_text']}</p>

<h3>Response</h3>
<p>{row['Response']}</p>
"""

        if send_email(row.get("Email",""), "Customer Support", body):
            df.at[idx, "Email_Status"] = "Sent"

# ---------------------------
# Individual Emails
# ---------------------------
st.subheader("Trigger Email Actions")

negative_df = df[df["Email_Trigger"] == "Yes"]

for idx, row in negative_df.iterrows():

    with st.expander(f"Review {idx+1}"):

        st.write(row["Review_text"])
        st.progress(row["Confidence"])

        st.write("LLM Response:", row["Response"])

        use_manual = st.checkbox("Use Manual Response", key=f"chk_{idx}")

        manual = ""
        if use_manual:
            manual = st.text_area("Enter response", key=f"txt_{idx}")

        final_response = manual.strip() if use_manual and manual else row["Response"]

        disabled = row["Email_Status"] == "Sent"

        if st.button("📧 Send Email", key=f"btn_{idx}", disabled=disabled):

            body = f"""
<h3>Customer Review</h3>
<p>{row['Review_text']}</p>

<h3>Response</h3>
<p>{final_response}</p>
"""

            if send_email(row.get("Email",""), "Customer Support", body):
                df.at[idx, "Email_Status"] = "Sent"
                st.success("Email Sent")

# ---------------------------
# Chart
# ---------------------------
fig = px.bar(df["Sentiment"].value_counts().reset_index(),
             x="Sentiment", y="count", color="Sentiment")
st.plotly_chart(fig)

# ---------------------------
# Download
# ---------------------------
st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
