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
import re 
import time
from sklearn.metrics import accuracy_score

# ---------------------------
# Fix torch Streamlit bug
# ---------------------------
try:
    del torch._classes
except AttributeError:
    pass

warnings.filterwarnings("ignore", category=warnings.FutureWarning, module=re.compile("huggingface_hub\\.file_download"))

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

# --- Base64 Encoded Bosch Logo (Replace with your actual Base64 string) ---
# You would get this by converting the SVG content from the URL to base64.
# For example, if you download the SVG and convert it:
# import base64
# with open("path/to/Bosch-logo.svg", "rb") as f:
#     BOSCH_LOGO_BASE64 = base64.b64encode(f.read()).decode('utf-8')
# BOSCH_LOGO_SRC = f"data:image/svg+xml;base64,{BOSCH_LOGO_BASE64}"

# For demonstration, I'll use a placeholder. You NEED to replace this with the actual Base64 string.
# A small SVG like the Bosch logo will result in a much longer string than this placeholder.
# You can find the raw SVG content here: https://upload.wikimedia.org/wikipedia/commons/6/6f/Bosch-logo.svg
# Then use an online Base64 encoder for SVG or a Python script to convert it.
# Example: https://base64.guru/converter/encode/image
BOSCH_LOGO_BASE64_PLACEHOLDER = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxOTYgMTYwIj48cGF0aCBmaWxsPSIjRjAwIiBkPSJNMCAxNjBoNzcuMjcxVjBoLTEzLjcyNFYxMDYuMzU2SDAuNjQ3VjE2MEgwdmEwLjA2NXYtMC4wNjVoLTYuNTR2MTYuNjQ2SDc3LjI2Okw3Ny4yNzUgMTYwSDg4Ljk5MXYtNS41NjhsMjguMDYxLTI4LjA1OEg1LjQ1M3YxNS41MzdIMTk2VjBoLjc3NXYxNTQuNjQ5aC02LjU0djExLjA0MmMwLjY0NiAwLjY1MS0wLjY0NiAwLjYzNyAwLjY0NiAwLjY1djUuNTY2SDc3LjI3NWMtNi41NCAwLjY0Ni01LjI1MSAwLjY0OS01LjQ1MyAwLjY0OUgwdjcuMjcxYy0xMS4wNDQgMC42NS0xMy43MjYgMC42NS0xNC4zNzIgMC42NUgwc3YtMC42NWgtNi41NTd2LTUuNTY3aC02LjU0MXYtMS4xMDVsLTUuNDUyLTUuNTY2VjExNy43NjlsNTguNjY2LDU4LjY3OUgxNzQuNjEzVjExNy43NTdsNTguNjQ4LDU4LjY4NlYxNjAuNjQ5SDcuMjcyVjBoLTcuMjcxdi01LjU2OEgwLjY0NlYwSDE5Ni42MjJjLTcuMjcxIDcuMjcxLTUuNTY3IDcuMjcxLDUuNTY3IDcuMjcxVjBoLTYuNTQxVjEuMzA2TDcuMjcwIDBjNi41NDctMC42NDYgMTkuNjMzLTAuNjQ1IDMwLjY3NyAwdi0xMS4wNDVoLTcuMjcyVjBIMTMyLjI3NGwtNTguNjY3LTU4LjY3di03LjI3MWgtMjcuNDEydi0yOC41NzZMNDAuMDcgNDAuMDk1VjBoLTI3LjQxNFYwSDE4My41NDNWMkgyNC41MTR2LTcuMjc1aC0xNi42NDlWMmgwLjY0NlYwSDE4My42NDhWMTE2LjY1MVYxNjBINjguNjc4VjBoLTUuNTY5VjEwNi4zNThIMTIuNTI2VjE2MC42NTFINjguNjc4VjBoLTUuNTY5VjExNy43NTdIeS4xNzJMMTY3Ljk2OSAwaC0xMy43MjJWMTU0LjY0OUgxOTZWMEgwLjYzN1YxNjBINHZIMTg4LjQ2M1YwSDc2LjYzNlYxMDYuMzU2SDkuNjE1VjE2MC42NTFINzYuNjM2VjBoLTcuMjcyVjExNy43NzdILjY0NlYyNC41MTJWMTZIMTk2LjYzOFYwSDcuMjcyVjU4LjY3OEwxMDYuMzU3IDBWMEgwLjY0NlYxMDYuMzU2SDExNy43NjJMMTg4LjQ2MyAwVjBINi43MDRWMTYwLjYzOUgwLjY0NlY2Ljc2NlY2Ljc3TDUuNDUzIDI2LjE1MUgyNy42MTZWMTIwLjgySDB2MTcuMjIxVjE2MGg2OS45Njh2LTc3LjI3MUgzMC42NzhWMjQuNTE0SDkuNjE1VjE2MGgzMC42Nzh2LTc3LjI3MUg3MC42MTR2NTguNjcxSDB2NzcuMjcxSDc1LjkyNXYtNTguNjcxVjBoLTcuMjc1VjQzLjU0OUg2LjcwNHYtMTkuMDM1SDMyLjU5NlY1LjQ1NFYwSDAuNjQ2VjBoLTUuNDUzVjBIMjcuNjE2VjBIMTk2VjBoLS4wMThVMEgxMjAuMzQ1VjBINi43MDNWNzcuMjc3VjE2MGg0OC43ODh2LTEwNi4zNkg1LjQ1NFYwSDAuNjQ2VjBoLTUuNDUzVjBIMTk2VjBoLS42NDYVMEgxNzkuOTY4VjBoLTUuNDU0VjBIMjk5LjY0NlYwSDkuNjE1VjBIMTk2VjBoLTcuMjc1VjBoLTcuMjc1VjBoLTYuNTRWMzUuMzQzTDExMi42MTcgMHoiLz48L3N2Zz4=" # Placeholder - replace with actual Base64 SVG
BOSCH_LOGO_SRC = BOSCH_LOGO_BASE64_PLACEHOLDER


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
# Session state
# ---------------------------
for key in ["processed", "last_uploaded_filename"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "last_uploaded_filename" else False

if "open_expander_index" not in st.session_state:
    st.session_state.open_expander_index = None

# Ensure df_processed is initialized in session state
if "df_processed" not in st.session_state:
    st.session_state.df_processed = pd.DataFrame()


uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# ---------------------------
# Load Data
# ---------------------------
df = pd.DataFrame() # Initialize df to avoid NameError

if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    if os.path.exists(sample_data_path):
        st.success("Using 'sample_data.csv' from directory.")
        try:
            df = pd.read_csv(sample_data_path)
        except Exception as e:
            st.error(f"Error reading 'sample_data.csv': {e}")
            st.stop()
    else:
        st.error("'sample_data.csv' not found. Please upload a CSV.")
        st.stop()

if df.empty or "Review_text" not in df.columns:
    st.error("Missing or empty 'Review_text' column.")
    st.stop()

df = df.head(100)

# Ensure 'Email' column exists for email functionality
if "Email" not in df.columns:
    df["Email"] = "" # Add an empty 'Email' column if not present
    st.warning("No 'Email' column found in the CSV. Email sending functionality will be limited.")

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

with st.spinner("Loading AI models... This might take a moment."):
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
NEGATIVE_THRESHOLD = st.sidebar.slider(
    "Negative confidence threshold (for Email Trigger)",
    min_value=0.50, max_value=0.95, value=0.70, step=0.01
)

# ---------------------------
# Functions
# ---------------------------
def analyze_all_sentiments(texts):
    results = sentiment_pipeline([t[:512] for t in texts], return_all_scores=True)
    labels, confidences = [], []
    for res in results:
        top = max(res, key=lambda x: x['score'])
        labels.append(label_map[top['label']])
        confidences.append(round(top['score'], 2))
    return labels, confidences

def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No automated response generated for non-negative reviews."
    prompt = (
        "You are a polite and helpful customer support agent. "
        "Respond professionally, briefly, and empathetically.\n"
        f"Review: {review}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad(): # Added no_grad for inference
        output = model.generate(**inputs, max_new_tokens=100)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    return f"Thank you for your review. We will look into the issue. {reply}"


def build_email_body(row, response_text):
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color:#f6f6f6; padding:20px;">
        
        <div style="max-width:600px; margin:auto; background:white; padding:20px; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,0.1);">

            <!-- Header with Logo - NOW USING BASE64 ENCODING -->
            <div style="text-align:center; border-bottom:1px solid #ddd; padding-bottom:10px; margin-bottom:20px;">
                <img src="{BOSCH_LOGO_SRC}" 
                     alt="Bosch Logo" 
                     width="120" height="auto" 
                     style="display:block; margin:0 auto;"/>
                <h2 style="color:#d50000; margin-top:10px;">Customer Support</h2>
            </div>

            <p style="color:#333;">Dear Customer,</p>

            <p style="color:#333;">Thank you for your feedback. Please find our response below.</p>

            <!-- Ticket Info -->
            <div style="background:#f2f2f2; padding:10px; border-radius:5px; margin-bottom:15px; border-left:4px solid #d50000;">
                <b style="color:#555;">Ticket Number:</b> {row.get('Unique_ID', 'N/A')}
            </div>

            <h4 style="color:#333; border-bottom:1px solid #eee; padding-bottom:5px;">Review Details:</h4>
            <p style="color:#444; line-height:1.6;">
            <b style="color:#555;">ID:</b> {row.get('Unique_ID', 'N/A')}<br>
            <b style="color:#555;">Category:</b> {row.get('Category', 'N/A')}<br>
            <b style="color:#555;">Date:</b> {row.get('Date', 'N/A')}
            </p>

            <p style="color:#444; line-height:1.6;"><b style="color:#555;">Customer's Review:</b><br>
            {row.get('Review_text', '')}</p>

            <!-- Response -->
            <h4 style="color:#333; border-bottom:1px solid #eee; padding-bottom:5px; margin-top:20px;">Our Response:</h4>
            <p style="color:#222; line-height:1.6; background:#e0f7fa; padding:15px; border-radius:5px; border-left:4px solid #00acc1;">{response_text}</p>

            <br>

            <!-- Support Link -->
            <p style="color:#555; text-align:center; margin-top:30px;">
                Need more help?  
                <a href="https://www.bosch.com/contact/" target="_blank" style="color:#d50000; text-decoration:none; font-weight:bold;">
                    Contact Support
                </a>
            </p>

            <br>

            <!-- Footer -->
            <div style="border-top:1px solid #ddd; padding-top:15px; font-size:11px; color:#888; text-align:center; margin-top:20px;">
                Best Regards,<br>
                Bosch Customer Support Team<br>
                [Your Company Contact Info]
            </div>

        </div>
    </body>
    </html>
    """
    

# ---------------------------
# Processing
# ---------------------------
if not st.session_state.processed:
    with st.spinner("Analyzing sentiments and generating responses..."):
        sentiments, confidences = analyze_all_sentiments(df["Review_text"].tolist())

        responses, times = [], []
        for i, row in df.iterrows():
            start = time.time()
            responses.append(generate_response(sentiments[i], row["Review_text"]))
            times.append(time.time() - start)

        df["Sentiment"] = sentiments
        df["Confidence"] = confidences
        df["Response"] = responses
        df["Processing_Time_sec"] = times
        df["Email_Status"] = "Pending"

        st.session_state.df_processed = df.copy()
        st.session_state.processed = True
else:
    df = st.session_state.df_processed.copy()

# ---------------------------
# Email Trigger
# ---------------------------
df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if (r["Sentiment"] == "Negative" and r["Confidence"] >= NEGATIVE_THRESHOLD) else "No",
    axis=1
)

st.success("Processing complete!")

# ---------------------------
# Metrics (FIXED)
# ---------------------------
total = len(df)
triggered = len(df[df["Email_Trigger"] == "Yes"])

col1, col2 = st.columns(2)

with col1:
    st.metric("Emails to Send", triggered)

with col2:
    trigger_rate = (triggered / total) * 100 if total > 0 else 0
    st.metric("Trigger Rate", f"{trigger_rate:.1f}%")


# Legend
st.markdown("""
<div style="display:flex; gap:25px; align-items:center; margin-top:5px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:18px; height:18px; background-color:#ffcccc; border-radius:4px;"></div>
        <span style="color:black; font-weight:500;">Triggered (High Confidence)</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:18px; height:18px; background-color:#ffe6e6; border-radius:4px;"></div>
        <span style="color:black; font-weight:500;">Negative (Low Confidence)</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Toggle
# ---------------------------
show_only = st.checkbox("Show only actionable reviews (email triggers)")

# ---------------------------
# Preview
# ---------------------------
st.subheader("Preview")

display_df = df[df["Email_Trigger"] == "Yes"].copy() if show_only else df.copy()

def highlight_negative(row):
    if row["Email_Trigger"] == "Yes":
        return ['background-color: #ffcccc'] * len(row)
    elif row["Sentiment"] == "Negative":
        return ['background-color: #ffe6e6'] * len(row)
    else:
        return [''] * len(row)

cols_to_show = [col for col in [
    "Unique_ID", "Date", "Category", "Review_text",
    "Sentiment", "Confidence", "Response", "Email_Trigger", "Email_Status"
] if col in display_df.columns]

styled_df = display_df[cols_to_show].style.apply(highlight_negative, axis=1)
st.dataframe(styled_df, use_container_width=True)



# ---------------------------
# Bulk Send
# ---------------------------
st.subheader("Bulk Email Actions")
if st.button("🚀 Send All Triggered Emails Now"):
    triggered_count = 0
    skipped_count = 0
    with st.spinner("Attempting to send all triggered emails..."):
        for idx, row in df[df["Email_Trigger"] == "Yes"].iterrows():
            if row["Email_Status"] == "Sent":
                skipped_count += 1
                continue # Skip already sent emails

            recipient_email = row.get("Email", "")
            if not recipient_email:
                st.warning(f"Skipping review {row.get('Unique_ID', 'N/A')} - no recipient email found.")
                continue

            email_subject = f"Response to your feedback (Ticket: {row.get('Unique_ID', 'N/A')})"
            body = build_email_body(row, row["Response"]) # Use auto-generated response for bulk send

            if send_email(recipient_email, email_subject, body):
                df.at[idx, "Email_Status"] = "Sent"
                triggered_count += 1
            else:
                st.error(f"Failed to send email for {row.get('Unique_ID', 'N/A')}.")
        
        st.session_state.df_processed = df.copy() # Update session state
        st.success(f"Bulk send complete! {triggered_count} emails sent, {skipped_count} skipped.")
        st.rerun() # Rerun to update table and expander states

# ---------------------------
# Trigger Email Section
# ---------------------------
st.subheader("Individual Email Actions (for Triggered Reviews)")
negative_df = df[df["Email_Trigger"] == "Yes"].reset_index(drop=True)

if negative_df.empty:
    st.info("No reviews triggered an email action based on the current threshold.")
else:
    for idx, row in negative_df.iterrows():
        uid = row.get('Unique_ID', f'Row {idx+1}')

        # Use a unique key for the expander for correct state management
        expander_key = f"expander_{uid}_{idx}" 

        with st.expander(f"Email for Review #{idx+1} - {uid} (Status: {row['Email_Status']})", expanded=False, key=expander_key):
            st.markdown(f"**ID:** {row.get('Unique_ID', 'N/A')}")
            st.markdown(f"**Date:** {row.get('Date', 'N/A')}")
            st.markdown(f"**Category:** {row.get('Category', 'N/A')}")
            st.markdown(f"**Customer Review:** {row['Review_text']}")
            st.markdown(f"**Sentiment:** {row['Sentiment']} (Confidence: {row['Confidence']:.2f})")
            
            st.markdown("---") 

            st.markdown(f"**Auto-generated Response:**")
            st.write(row['Response']) # Display auto-generated response as plain text

            st.markdown("---")

            recipient_email_display = row.get("Email", "No Email Provided")
            st.info(f"Recipient Email Address: **{recipient_email_display}**")

            use_manual = st.checkbox("Override with Manual Response", key=f"chk_manual_{idx}") # Unique key

            manual_text = ""
            if use_manual:
                manual_text = st.text_area("Enter your Manual Response Here", height=100, key=f"txt_manual_{idx}", 
                                           value=row['Response']) # Pre-fill with auto-response for easier editing

            # Determine the final response to be sent
            final_response = (
                manual_text.strip()
                if use_manual and manual_text.strip()
                else row["Response"]
            )

            # Disable send button if email already sent
            disabled_flag = (row["Email_Status"] == "Sent")

            if disabled_flag:
                st.info(f"Email for {uid} has already been sent.")
                st.button("Email Sent (Disabled)", key=f"btn_sent_{idx}", disabled=True)
            else:
                if st.button("Send Email", key=f"btn_send_{idx}"): # Unique key
                    recipient_email = row.get("Email", "")
                    
                    if not recipient_email:
                        st.warning(f"No recipient email address found for review {uid}. Cannot send.")
                    else:
                        email_subject = f"Response to your feedback (Ticket: {uid})"
                        body = build_email_body(row, final_response)

                        with st.spinner(f"Sending email for {uid} to {recipient_email}..."):
                            if send_email(recipient_email, email_subject, body):
                                # Update the original DataFrame in session state
                                # Find the index in the *original* df_processed, not the filtered negative_df
                                original_df_index = st.session_state.df_processed[st.session_state.df_processed["Unique_ID"] == uid].index[0]
                                st.session_state.df_processed.at[original_df_index, "Email_Status"] = "Sent"
                                st.success(f"Email for {uid} sent to {recipient_email}!")
                                st.rerun() # Rerun to update the button state and table
                            else:
                                st.error(f"Failed to send email for {uid}. Check logs for details.")

# ---------------------------
# Metrics Summary
# ---------------------------
st.subheader("Metrics")
# For accuracy, you'd need a 'true' sentiment column. Here, it's just comparing to itself.
# If you don't have a ground truth, consider removing this or clarifying.
# For demo purposes, we'll keep it as-is but note it's not a true accuracy.
# acc = accuracy_score(df["Sentiment"], df["Sentiment"]) # This will always be 1.0
st.write(f"Average processing time per review: {df['Processing_Time_sec'].mean():.2f} seconds")


# ---------------------------
# Chart
# ---------------------------
st.subheader("Sentiment Distribution")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

# Ensure all three sentiments are present for consistent coloring
all_sentiments = pd.DataFrame({"Sentiment": ["Negative", "Neutral", "Positive"], "Count": 0})
chart_data = pd.concat([all_sentiments, chart_data]).groupby("Sentiment", as_index=False)["Count"].sum()


fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Distribution of Sentiment Labels")
fig.update_layout(xaxis_title="Sentiment", yaxis_title="Number of Reviews")
st.plotly_chart(fig)

# ---------------------------
# Download
# ---------------------------
st.download_button(
    "Download CSV",
    df.to_csv(index=False),
    "output.csv"
)
