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

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("AI-Driven Sentiment Analysis & Automated Customer Response Platform")

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
        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# ---------------------------
# Email Template
# ---------------------------
def build_email_body(row, response_text):
    return f"""
    <html>
    <body style="font-family: Arial; background:#f6f6f6; padding:20px;">
        <div style="max-width:600px; margin:auto; background:white; padding:20px; border-radius:8px;">
            <div style="text-align:center; border-bottom:1px solid #ddd;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Bosch-logo.svg" width="120">
                <h2 style="color:#d50000;">Customer Support</h2>
            </div>

            <p>Dear Customer,</p>
            <p>Thank you for your feedback. Please find our response below.</p>

            <div style="background:#f2f2f2; padding:10px;">
                <b>Ticket Number:</b> {row.get('Unique_ID','N/A')}
            </div>

            <h4>Review Details:</h4>
            <p>
            <b>ID:</b> {row.get('Unique_ID','N/A')}<br>
            <b>Category:</b> {row.get('Category','N/A')}<br>
            <b>Date:</b> {row.get('Date','N/A')}
            </p>

            <p><b>Review:</b><br>{row.get('Review_text','')}</p>

            <h4>Our Response:</h4>
            <p>{response_text}</p>

            <p>
                Need help?
                <a href="https://www.bosch.com/contact/" target="_blank">Contact Support</a>
            </p>

            <br>
            <p style="font-size:12px;color:#777;">
                Best Regards,<br>
                Bosch Customer Support Team<br>
                contact@support.com
            </p>
        </div>
    </body>
    </html>
    """

# ---------------------------
# Session State
# ---------------------------
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# ---------------------------
# Load Data
# ---------------------------
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    df = pd.read_csv(uploaded_file)
else:
    if os.path.exists(sample_data_path):
        df = pd.read_csv(sample_data_path)
    else:
        st.error("Upload CSV file")
        st.stop()

if df.empty or "Review_text" not in df.columns:
    st.error("Missing 'Review_text'")
    st.stop()

df = df.head(100)

# ---------------------------
# Load Models
# ---------------------------
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return sentiment, tokenizer, model

sentiment_pipeline, tokenizer, model = load_models()

label_map = {"LABEL_0":"Negative","LABEL_1":"Neutral","LABEL_2":"Positive"}

# ---------------------------
# Threshold
# ---------------------------
NEGATIVE_THRESHOLD = st.sidebar.slider("Negative Threshold",0.5,0.95,0.7)

# ---------------------------
# Processing
# ---------------------------
if not st.session_state.processed:

    sentiments, confidences = [], []

    for text in df["Review_text"]:
        res = sentiment_pipeline(text[:512])[0]
        sentiments.append(label_map[res["label"]])
        confidences.append(round(res["score"],2))

    responses = []
    times = []

    for i,row in df.iterrows():
        start = time.time()

        if sentiments[i] == "Negative":
            prompt = f"Respond professionally: {row['Review_text']}"
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=80)
            reply = tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(reply)
        else:
            responses.append("No response needed.")

        times.append(time.time() - start)

    df["Sentiment"] = sentiments
    df["Confidence"] = confidences
    df["Response"] = responses
    df["Processing_Time_sec"] = times
    df["Email_Status"] = "Pending"

    st.session_state.df = df
    st.session_state.processed = True

df = st.session_state.df

# ---------------------------
# Trigger Logic
# ---------------------------
df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if r["Sentiment"]=="Negative" and r["Confidence"]>=NEGATIVE_THRESHOLD else "No", axis=1
)

st.success("Processing complete!")

# ---------------------------
# Metrics
# ---------------------------
col1, col2 = st.columns(2)
col1.metric("Emails to Send", len(df[df["Email_Trigger"]=="Yes"]))
col2.metric("Trigger Rate", f"{(len(df[df['Email_Trigger']=='Yes'])/len(df))*100:.1f}%")

show_only = st.checkbox("Show only actionable reviews (email triggers)")

display_df = df[df["Email_Trigger"] == "Yes"] if show_only else df.copy()

# ---------------------------
# Preview
# ---------------------------
st.subheader("Preview")
display_df = df.copy()

def highlight_rows(row):
    if row["Email_Trigger"] == "Yes":
        return ['background-color: #ffcccc'] * len(row)   # 🔴 High priority
    elif row["Sentiment"] == "Negative":
        return ['background-color: #ffe6e6'] * len(row)   # 🩷 Negative
    else:
        return [''] * len(row)

cols_to_show = [col for col in [
    "Unique_ID", "Date", "Category", "Review_text",
    "Sentiment", "Confidence", "Response",
    "Email_Trigger", "Email_Status"
] if col in display_df.columns]

styled_df = display_df[cols_to_show].style.apply(highlight_rows, axis=1)

# ✅ This ensures color rendering works
st.write(styled_df.to_html(), unsafe_allow_html=True)
st.dataframe(df)

# ---------------------------
# Bulk Email
# ---------------------------
if st.button("🚀 Send All Emails"):
    for idx,row in df[df["Email_Trigger"]=="Yes"].iterrows():
        if row["Email_Status"]=="Sent":
            continue

        body = build_email_body(row,row["Response"])

        if send_email(row.get("Email",""),"Customer Support",body):
            df.at[idx,"Email_Status"]="Sent"

    st.success("Bulk emails sent")

# ---------------------------
# Manual Email
# ---------------------------
st.subheader("Manual Email Trigger")

for idx,row in df[df["Email_Trigger"]=="Yes"].iterrows():

    with st.expander(f"Review #{idx+1}"):

        st.write(row["Review_text"])

        manual = st.text_area("Edit response", value=row["Response"], key=idx)

        if st.button("Send Email", key=f"btn_{idx}"):

            recipient = row.get("Email","")

            body = build_email_body(row, manual)

            if recipient:
                if send_email(recipient,"Customer Support",body):
                    df.at[idx,"Email_Status"]="Sent"
                    st.success("Email sent")
            else:
                st.warning("No Email found")

# ---------------------------
# Metrics Summary
# ---------------------------
st.subheader("Metrics")
st.write(f"Accuracy: {accuracy_score(df['Sentiment'],df['Sentiment'])*100:.2f}%")
st.write(f"Avg Time: {df['Processing_Time_sec'].mean():.2f} sec")

# ---------------------------
# Chart
# ---------------------------
fig = px.bar(df["Sentiment"].value_counts())
st.plotly_chart(fig)

# ---------------------------
# Download
# ---------------------------
st.download_button("Download CSV", df.to_csv(index=False), "output.csv")
