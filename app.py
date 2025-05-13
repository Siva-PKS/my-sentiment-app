import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

# Set up the page
st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("\U0001F4CA Customer Review Sentiment Analyzer & Auto-Responder")

# File Upload Section
uploaded_file = st.file_uploader("\U0001F4C1 Upload a CSV file with a column named 'Review_text'", type="csv")

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

# Limit rows for demo
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting processing to the first {MAX_ROWS} reviews for faster demo performance.")
    df = df.head(MAX_ROWS)

# Filter out short or invalid reviews
def is_valid_review(review):
    return isinstance(review, str) and len(review.strip()) > 10

df = df[df["Review_text"].apply(is_valid_review)]

# Load Sentiment Model (English-only)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# Load Response Generation Model (small for performance)
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Map sentiment labels
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Run sentiment analysis
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return label_map.get(result["label"], "Unknown")
    except Exception:
        return "Unknown"

# Generate meaningful response for negative sentiment

def generate_response(sentiment, review):
    prompt = f"""
You are a customer support agent. A customer left the following {sentiment.lower()} review. Write a short, empathetic, and helpful response.

Customer Review: \"{review}\"

Support Agent Response:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=120)
    return response_tokenizer.decode(output[0], skip_special_tokens=True).strip()

# Generate response only for Negative
responses = []
sentiments = []
progress_bar = st.progress(0)

for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    sentiments.append(sentiment)
    if sentiment == "Negative":
        response = generate_response(sentiment, row.Review_text)
    else:
        response = "No response needed"
    responses.append(response)
    progress_bar.progress((i + 1) / len(df))

# Add results to DataFrame
df["Sentiment"] = sentiments
df["Response"] = responses

st.success("✅ Processing complete!")

# Show results
st.subheader("\U0001F4CB Results Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Chart: Sentiment distribution
st.subheader("\U0001F4CA Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Distribution")
st.plotly_chart(fig, use_container_width=True)

# Download processed CSV
st.download_button(
    label="⬇️ Download Results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
) 
