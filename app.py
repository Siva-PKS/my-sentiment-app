
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page setup
st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# File uploader
uploaded_file = st.file_uploader("📁 Upload a CSV file with a column named 'Review_text'", type="csv")

# Load data
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

# Limit rows for demo purposes
df = df.head(20)

# Load sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# Load response model (FLAN-T5-small)
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Label mapping
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Analyze sentiment
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return label_map.get(result["label"], "Unknown")
    except Exception:
        return "Unknown"

# Generate response
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed"
    prompt = f"""You're a helpful support agent. Generate a professional response for the customer review below.

Review: "{review}"
Sentiment: {sentiment}
Response:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=100)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Sentiment and response processing with progress bar
with st.spinner("🚀 Running sentiment analysis and generating responses..."):
    df["Sentiment"] = df["Review_text"].apply(analyze_sentiment)

    # Add progress bar
    progress_bar = st.progress(0)
    responses = []
    for i, row in enumerate(df.itertuples(index=False)):
        responses.append(generate_response(row.Sentiment, row.Review_text))
        progress_bar.progress((i + 1) / len(df))
    df["Response"] = responses

st.success("✅ Processing complete!")

# Show results
st.subheader("📋 Results Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Distribution")
st.plotly_chart(fig, use_container_width=True)

# Download
st.download_button(
    label="⬇️ Download Results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
