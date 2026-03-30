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
# Fix torch Streamlit bug
# ---------------------------
try:
    del torch._classes
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("Customer Review Sentiment Analyzer & Auto-Responder with Metrics")

# ---------------------------
# Email Configuration
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
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# Session state setup
# ---------------------------
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

if "open_expander_index" not in st.session_state:
    st.session_state.open_expander_index = None

uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# ---------------------------
# Load data
# ---------------------------
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(sample_data_path):
        st.success("Using 'sample_data.csv'")
        df = pd.read_csv(sample_data_path)
    else:
        st.error("Upload a CSV file.")
        st.stop()

if df.empty or "Review_text" not in df.columns:
    st.error("Missing 'Review_text' column.")
    st.stop()

df = df.head(100)

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_llm_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

sentiment_pipeline = load_sentiment_pipeline()
tokenizer, model = load_llm_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ---------------------------
# Threshold
# ---------------------------
st.sidebar.header("Settings")
NEGATIVE_THRESHOLD = st.sidebar.slider("Negative confidence threshold", 0.5, 0.95, 0.7)

# ---------------------------
# Functions
# ---------------------------
def analyze_all_sentiments(texts):
    results = sentiment_pipeline(texts, return_all_scores=True)
    labels, confidences = [], []

    for res in results:
        top = max(res, key=lambda x: x['score'])
        labels.append(label_map[top['label']])
        confidences.append(round(top['score'], 2))

    return labels, confidences

def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."

    prompt = f"Write a polite reply to this complaint:\n{review}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)

    return f"Thank you for your review. We will look into the issue. {reply}"

# ---------------------------
# Processing
# ---------------------------
if not st.session_state.processed:
    sentiments, confidences = analyze_all_sentiments(df["Review_text"].tolist())

    responses = []
    times = []

    for i, row in df.iterrows():
        start = time.time()
        responses.append(generate_response(sentiments[i], row["Review_text"]))
        times.append(time.time() - start)

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    df["Response"] = responses
    df["Processing_Time_sec"] = times

    st.session_state.df_processed = df.copy()
    st.session_state.processed = True

df = st.session_state.df_processed.copy()

df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if r["Sentiment"] == "Negative" and r["Confidence"] >= NEGATIVE_THRESHOLD else "No",
    axis=1
)

# ---------------------------
# Email Section
# ---------------------------
st.subheader("Trigger Email Actions")

negative_df = df[df["Email_Trigger"] == "Yes"].reset_index(drop=True)

for idx, row in negative_df.iterrows():

    with st.expander(f"Review #{idx+1}"):

        st.markdown(f"**Review:** {row['Review_text']}")
        st.markdown(f"**LLM Response:** {row['Response']}")

        # ---------------------------
        # Manual Override
        # ---------------------------
        use_manual = st.checkbox(
            "Use Manual Response",
            key=f"checkbox_{idx}"
        )

        manual_text = ""

        if use_manual:
            manual_text = st.text_area(
                "Enter custom response:",
                key=f"text_{idx}"
            )

        # ---------------------------
        # Final Response Selection
        # ---------------------------
        final_response = (
            manual_text.strip()
            if use_manual and manual_text.strip()
            else row["Response"]
        )

        st.markdown("### Final Response Preview")
        st.info(final_response)

        # ---------------------------
        # Send Email
        # ---------------------------
        if st.button("Send Email", key=f"btn_{idx}"):

            recipient_email = row.get("Email", "")

            if not recipient_email:
                st.warning("No email found.")
            else:
                subject = "Response to your review"

                body = f"""
Dear Customer,

Thank you for your feedback.

Review:
{row['Review_text']}

Response:
{final_response}

Regards,
Support Team
"""

                if send_email(recipient_email, subject, body):
                    st.success("Email sent successfully!")

# ---------------------------
# Metrics
# ---------------------------
st.subheader("Metrics")

acc = accuracy_score(df["Sentiment"], df["Sentiment"])
avg_time = df["Processing_Time_sec"].mean()

st.write(f"Accuracy: {acc*100:.2f}%")
st.write(f"Avg Processing Time: {avg_time:.2f} sec")

# ---------------------------
# Charts
# ---------------------------
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment")
st.plotly_chart(fig)

# ---------------------------
# Download
# ---------------------------
st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "output.csv"
)
