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
from datetime import datetime

# =============================
# Fix Streamlit & Transformers warnings
# =============================
try:
    del torch._classes
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# =============================
# App Config
# =============================
st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# =============================
# Email Configuration (Use secrets for security)
# =============================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "spkincident@gmail.com"          
SENDER_PASSWORD = st.secrets["email_password"]  # Gmail App Password

# =============================
# Path for the negative reviews log file
# =============================
log_file_path = "negative_reviews.log"

# Ensure the log file exists at startup (prevents FileNotFoundError)
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as f:
        pass  # Creates an empty file
        

# =============================
# Utility Functions
# =============================
def send_email(recipient_email: str, subject: str, body: str) -> bool:
    """
    Sends an email using Gmail's SMTP server.
    Args:
        recipient_email (str): Recipient's email address
        subject (str): Email subject
        body (str): Email body text
    Returns:
        bool: True if sent successfully, False otherwise
    """
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
        st.error(f"❌ Failed to send email: {e}")
        return False


def log_negative_review(review_id, review_text, sentiment, confidence, response):
    """
    Logs negative reviews into a CSV file for tracking.
    Creates 'negative_reviews_log.csv' if it does not exist.
    Args:
        review_id (str/int): Unique review identifier
        review_text (str): The review text
        sentiment (str): Sentiment classification (Negative)
        confidence (float): Model confidence score
        response (str): Auto-generated customer support response
    """
    log_file = "negative_reviews_log.csv"
    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Review_ID": review_id,
        "Review_Text": review_text,
        "Sentiment": sentiment,
        "Confidence": confidence,
        "Response": response
    }

    log_df = pd.DataFrame([log_entry])

    if os.path.exists(log_file):
        log_df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        log_df.to_csv(log_file, index=False)


@st.cache_resource
def load_sentiment_pipeline():
    """
    Loads the CardiffNLP Twitter RoBERTa sentiment analysis pipeline.
    Cached for performance.
    """
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


@st.cache_resource
def load_llm_model():
    """
    Loads FLAN-T5-small model and tokenizer for response generation.
    Cached for performance.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model


def analyze_all_sentiments(texts):
    """
    Runs sentiment analysis on a list of text strings.
    Args:
        texts (list): List of review texts
    Returns:
        tuple: (list of labels, list of confidence scores)
    """
    results = sentiment_pipeline([t[:512] for t in texts], return_all_scores=True)
    labels, confidences = [], []
    for res in results:
        top = max(res, key=lambda x: x['score'])
        label = label_map.get(top['label'], "Unknown")
        confidence = round(top['score'], 2)
        labels.append(label)
        confidences.append(confidence)
    return labels, confidences


def generate_response(sentiment, review):
    """
    Generates an auto-response for negative reviews.
    Args:
        sentiment (str): Sentiment label
        review (str): Review text
    Returns:
        str: Auto-generated or default response
    """
    if sentiment != "Negative":
        return "No response needed."
    prompt = (
        "You are a polite and helpful customer support agent. "
        "Write a short, professional reply to this negative customer review:\n"
        f"Review: {review}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=150)
    llm_reply = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return f"Thank you for your review. We will look into the issue. {llm_reply.rstrip('.!?')}."


def highlight_negative(row):
    """
    Highlights negative sentiment rows in the DataFrame preview.
    """
    return ['background-color: #ffe6e6'] * len(row) if row["Sentiment"] == "Negative" else [''] * len(row)


# =============================
# Session State
# =============================
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

if "open_expander_index" not in st.session_state:
    st.session_state.open_expander_index = None

# =============================
# File Upload Handling
# =============================
uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(sample_data_path):
        st.success("✅ Using 'sample_data.csv' from directory.")
        df = pd.read_csv(sample_data_path)
    else:
        st.error("❌ 'sample_data.csv' not found. Please upload a CSV.")
        st.stop()

# Validate CSV
if df.empty or "Review_text" not in df.columns:
    st.error("❌ Missing or empty 'Review_text' column.")
    st.stop()

# Limit rows for faster demo
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting to first {MAX_ROWS} rows for demo.")
    df = df.head(MAX_ROWS)

# =============================
# Load Models
# =============================
sentiment_pipeline = load_sentiment_pipeline()
tokenizer, model = load_llm_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# =============================
# Process Data
# =============================
if not st.session_state.processed:
    progress_bar = st.progress(0)
    sentiments, confidences = analyze_all_sentiments(df["Review_text"].tolist())
    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    responses = []

    for i, row in df.iterrows():
        resp = generate_response(row["Sentiment"], row["Review_text"])
        responses.append(resp)

        # Log only negative reviews
        if row["Sentiment"] == "Negative":
            log_negative_review(
                row.get("Unique_ID", f"Row{i+1}"),
                row["Review_text"],
                row["Sentiment"],
                row["Confidence"],
                resp
            )

        progress_bar.progress((i + 1) / len(df))

    df["Response"] = responses
    df["Email_Trigger"] = df["Sentiment"].apply(lambda s: "Yes" if s == "Negative" else "No")
    st.session_state.df_processed = df.copy()
    st.session_state.processed = True

df = st.session_state.df_processed

# =============================
# Display Processed Data
# =============================
st.success("✅ Processing complete!")

cols_to_show = [col for col in ["Unique_ID", "Date", "Category", "Review_text", "Sentiment", "Confidence", "Response", "Email_Trigger"] if col in df.columns]
styled_df = df[cols_to_show].style.apply(highlight_negative, axis=1)
st.dataframe(styled_df, use_container_width=True)

# =============================
# Email Sending Section
# =============================
st.subheader("📬 Trigger Email Actions (Only for Negative Reviews)")
negative_df = df[df["Email_Trigger"] == "Yes"].reset_index(drop=True)

for idx, row in negative_df.iterrows():
    uid = row.get('Unique_ID', f'Row {idx+1}')
    expanded = st.session_state.open_expander_index == idx

    with st.expander(f"✉️ Email for Review #{idx+1} - {uid}", expanded=expanded):
        st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
        st.markdown(f"**Date:** {row.get('Date', 'N/A')}")
        st.markdown(f"**Review:** {row['Review_text']}")
        st.markdown(f"**Response to be sent:** {row['Response']}")

        if st.button(f"📧 Send Email (Row {idx})", key=f"send_button_{idx}"):
            recipient_email = row.get("Email", "")
            st.session_state.open_expander_index = idx

            if recipient_email:
                subject = f"Response to your review (ID: {uid})"
                body = (
                    f"Dear Customer,\n\n"
                    f"Thank you for your feedback. Please find our response below.\n\n"
                    f"---\n"
                    f"Review Details:\n"
                    f"ID: {uid}\n"
                    f"Category: {row.get('Category', 'N/A')}\n"
                    f"Date: {row.get('Date', 'N/A')}\n"
                    f"Review:\n{row['Review_text']}\n\n"
                    f"Our Response:\n{row['Response']}\n"
                    f"---\n\n"
                    f"Best regards,\nCustomer Support Team"
                )
                if send_email(recipient_email, subject, body):
                    st.success(f"✅ Email sent to {recipient_email}")
            else:
                st.warning("⚠️ No Email address found in this row.")

# =============================
 # Log the negative review for future reference
# =============================
   
    with open("negative_reviews.log", "a", encoding="utf-8") as log_file:
        log_file.write(f"Review: {review_text}\n")
        log_file.write(f"Sentiment: {sentiment_label}\n")
        log_file.write(f"Generated Response: {response_text}\n")
        log_file.write("-" * 50 + "\n")


# =============================
# Charts
# =============================
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

if "Category" in df.columns:
    st.subheader("📈 Sentiment by Category")
    grouped = df.groupby(["Category", "Sentiment"]).size().reset_index(name="Count")
    fig2 = px.bar(grouped, x="Category", y="Count", color="Sentiment", barmode="group",
                  color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig2, use_container_width=True)

# =============================
# Download Processed CSV
# =============================
st.download_button(
    label="⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)

# =============================
# Display log download button (only if file has content)
# =============================

if os.path.getsize(log_file_path) > 0:
    with open(log_file_path, "r") as f:
        st.download_button(
            label="Download Negative Reviews Log",
            data=f,
            file_name="negative_reviews.log",
            mime="text/plain"
        )
else:
    st.info("No negative reviews have been logged yet.")

