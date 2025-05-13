import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set up the page
st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# File Upload Section
uploaded_file = st.file_uploader("📁 Upload a CSV file with a column named 'Review_text'", type="csv")

# Read CSV and validate content
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or "Review_text" not in df.columns:
            st.error("❌ Uploaded file is empty or missing 'Review_text' column.")
            st.stop()
        st.success("✅ CSV uploaded and validated successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("ℹ️ Using sample CSV (sample_data.csv)")
    except Exception as e:
        st.error(f"❌ Failed to load sample CSV: {e}")
        st.stop()

# Limit number of rows for performance demo
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting processing to the first {MAX_ROWS} rows for performance.")
    df = df.head(MAX_ROWS)

# Load Sentiment Model (Pre-trained sentiment classifier)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# Load Large LLM for Response Generation
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Sentiment label map
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Helper: Run sentiment analysis
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return label_map.get(result["label"], "Unknown")
    except Exception:
        return "Unknown"

# Generate response only for negative sentiment
def generate_response_if_needed(sentiment, review):
    if sentiment != "Negative":
        return "No response needed"
    prompt = f"""
You are a helpful customer support agent. Write a clear, empathetic, and professional response to the following negative review:

Review: "{review}"

Response:
"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=150)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Processing with progress bar
st.info("🧠 Analyzing reviews and generating responses...")
progress_bar = st.progress(0)

sentiments, responses = [], []
for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    response = generate_response_if_needed(sentiment, row.Review_text)
    sentiments.append(sentiment)
    responses.append(response)
    progress_bar.progress((i + 1) / len(df))

df["Sentiment"] = sentiments
df["Response"] = responses

st.success("✅ Processing complete!")

# Display results
display_cols = ["Review_text", "Sentiment", "Response"]
st.subheader("📋 Results Preview")
st.dataframe(df[display_cols], use_container_width=True)

# Sentiment distribution chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Distribution")
st.plotly_chart(fig, use_container_width=True)

# Download results
st.download_button(
    label="⬇️ Download Results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
