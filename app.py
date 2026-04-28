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

warnings.filterwarnings("ignore", category="FutureWarning", module="huggingface_hub.file_download")

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
# SENDER_PASSWORD should be configured securely, e.g., via Streamlit Secrets
# For local testing, you might hardcode it temporarily, but NOT for deployment.
# st.secrets["email_password"] will access secrets defined in .streamlit/secrets.toml
SENDER_PASSWORD = st.secrets["email_password"] 

def send_email(recipient_email, subject, body):
    """Sends an email using SMTP."""
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
        st.error(f"Failed to send email to {recipient_email}: {e}")
        return False

# ---------------------------
# Session state Initialization
# ---------------------------
# Initialize session state variables if they don't exist
if "processed" not in st.session_state:
    st.session_state["processed"] = False
if "last_uploaded_filename" not in st.session_state:
    st.session_state["last_uploaded_filename"] = None
if "open_expander_index" not in st.session_state:
    st.session_state["open_expander_index"] = None
if "df_processed" not in st.session_state:
    st.session_state["df_processed"] = pd.DataFrame()


uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")
sample_data_path = "sample_data.csv"

# ---------------------------
# Load Data
# ---------------------------
df = pd.DataFrame() # Initialize df to an empty DataFrame

if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False # Force reprocessing for new file
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
        st.stop()
else:
    if os.path.exists(sample_data_path):
        st.info("Using 'sample_data.csv' from directory as no file was uploaded.")
        try:
            df = pd.read_csv(sample_data_path)
        except Exception as e:
            st.error(f"Error reading 'sample_data.csv': {e}")
            st.stop()
    else:
        st.error("'sample_data.csv' not found. Please upload a CSV file.")
        st.stop()

if df.empty or "Review_text" not in df.columns:
    st.error("Uploaded CSV is empty or missing the 'Review_text' column. Please check your file.")
    st.stop()

# Limit the DataFrame for demonstration purposes to avoid long processing
df = df.head(100) 

# Ensure 'Email' column exists for email functionality
if "Email" not in df.columns:
    df["Email"] = "" # Add an empty 'Email' column if not present
    st.warning("No 'Email' column found in the CSV. Email sending functionality will be limited.")


# ---------------------------
# Load Models (cached for efficiency)
# ---------------------------
@st.cache_resource
def load_sentiment_pipeline():
    """Loads the pre-trained sentiment analysis model."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

@st.cache_resource
def load_llm_model():
    """Loads the tokenizer and LLM for response generation."""
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
# Threshold for Email Trigger
# ---------------------------
st.sidebar.header("Settings")
NEGATIVE_THRESHOLD = st.sidebar.slider(
    "Negative confidence threshold (for Email Trigger)",
    min_value=0.50, max_value=0.95, value=0.70, step=0.01
)

# ---------------------------
# Helper Functions for Processing
# ---------------------------
def analyze_all_sentiments(texts):
    """Analyzes sentiment for a list of texts."""
    # Truncate texts to avoid model input limits (typically 512 tokens)
    results = sentiment_pipeline([t[:512] for t in texts], return_all_scores=True)
    labels, confidences = [], []
    for res in results:
        # Get the label with the highest score
        top = max(res, key=lambda x: x['score'])
        labels.append(label_map[top['label']])
        confidences.append(round(top['score'], 2))
    return labels, confidences

def generate_response(sentiment, review):
    """Generates a polite response for negative reviews using LLM."""
    if sentiment != "Negative":
        return "No automated response generated for non-negative reviews."
    
    prompt = (
        "You are a polite and helpful customer support agent for Bosch. "
        "Respond professionally, briefly, and empathetically to the following customer review.\n"
        f"Customer Review: {review}"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    
    # Generate response, ensure it's on CPU if not using GPU
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)
    
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Add a standard opening
    return f"Dear Customer, thank you for your feedback. We are sorry to hear about your experience. {reply}"

def build_email_body(row, response_text):
    """Constructs an HTML email body for the customer response."""
    return f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color:#f6f6f6; padding:20px;">
        
        <div style="max-width:600px; margin:auto; background:white; padding:20px; border-radius:8px; box-shadow:0 0 10px rgba(0,0,0,0.1);">

            <!-- Header with Logo -->
            <div style="text-align:center; border-bottom:1px solid #ddd; padding-bottom:10px; margin-bottom:20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Bosch-logo.svg" 
                     alt="Bosch Logo" width="120"/>
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
# Core Processing Logic
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
        df["Email_Status"] = "Pending" # Default status

        st.session_state.df_processed = df.copy() # Store processed DataFrame in session state
        st.session_state.processed = True
else:
    df = st.session_state.df_processed.copy() # Load from session state if already processed

# ---------------------------
# Email Trigger Logic
# ---------------------------
df["Email_Trigger"] = df.apply(
    lambda r: "Yes" if (r["Sentiment"] == "Negative" and r["Confidence"] >= NEGATIVE_THRESHOLD) else "No",
    axis=1
)

st.success("Processing complete! Reviews are ready for analysis and response.")

# ---------------------------
# Overall Metrics
# ---------------------------
total_reviews = len(df)
triggered_emails = len(df[df["Email_Trigger"] == "Yes"])

st.subheader("Overview Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Reviews Analyzed", total_reviews)

with col2:
    st.metric("Reviews Triggering Email Action", triggered_emails)

with col3:
    trigger_rate = (triggered_emails / total_reviews) * 100 if total_reviews > 0 else 0
    st.metric("Email Trigger Rate", f"{trigger_rate:.1f}%")


# Legend for table highlighting
st.markdown("""
<div style="display:flex; gap:25px; align-items:center; margin-top:5px; margin-bottom:20px;">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:18px; height:18px; background-color:#ffcccc; border-radius:4px; border:1px solid #d50000;"></div>
        <span style="color:black; font-weight:500;">Email Triggered (High Confidence Negative)</span>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="width:18px; height:18px; background-color:#ffe6e6; border-radius:4px; border:1px solid #ff9999;"></div>
        <span style="color:black; font-weight:500;">Negative Sentiment (Below Threshold)</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------
# Interactive Data Preview
# ---------------------------
st.subheader("Review Data Preview")

show_only = st.checkbox("Show only reviews that triggered an email action")

display_df = df[df["Email_Trigger"] == "Yes"].copy() if show_only else df.copy()

def highlight_negative_reviews(row):
    """Applies conditional styling to DataFrame rows."""
    if row["Email_Trigger"] == "Yes":
        return ['background-color: #ffcccc'] * len(row) # Light red for triggered
    elif row["Sentiment"] == "Negative":
        return ['background-color: #ffe6e6'] * len(row) # Lighter red for other negatives
    else:
        return [''] * len(row)

# Define columns to display in the preview table
cols_to_show_in_preview = [col for col in [
    "Unique_ID", "Date", "Category", "Review_text",
    "Sentiment", "Confidence", "Email_Trigger", "Email_Status"
] if col in display_df.columns]

styled_df = display_df[cols_to_show_in_preview].style.apply(highlight_negative_reviews, axis=1)
st.dataframe(styled_df, use_container_width=True, height=300)


# ---------------------------
# Bulk Send Emails
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
# Individual Email Actions (with Manual Override)
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
# Sentiment Distribution Chart
# ---------------------------
st.subheader("Sentiment Distribution")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

# Ensure all three sentiments are present for consistent coloring
all_sentiments = pd.DataFrame({"Sentiment": ["Negative", "Neutral", "Positive"], "Count": 0})
chart_data = pd.concat([all_sentiments, chart_data]).groupby("Sentiment", as_index=False)["Count"].sum()


fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={
                 "Positive": "#28a745",  # Green
                 "Neutral": "#6c757d",   # Gray
                 "Negative": "#dc3545"   # Red
             },
             title="Distribution of Sentiment Labels")
fig.update_layout(xaxis_title="Sentiment", yaxis_title="Number of Reviews")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Download Processed Data
# ---------------------------
st.subheader("Download Processed Data")
st.download_button(
    label="Download Processed CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name="sentiment_analysis_output.csv",
    mime="text/csv",
    help="Download the CSV file with sentiment analysis results and email statuses."
)
