import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("📊 Enhanced Customer Review Sentiment Analyzer & Auto-Responder")

uploaded_file = st.file_uploader("📁 Upload a CSV file with a column named 'Review_text'", type="csv")

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

# Load improved sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# Load improved response generation model (larger FLAN)
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Analyze sentiment
@st.cache_data
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text.strip()[:512])[0]
        return result["label"].capitalize()
    except:
        return "Unknown"

# Generate empathetic auto-response for negative reviews only
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed"
    prompt = f"""You are a helpful customer support agent. Write a clear, empathetic, and professional response to the following negative review:\n\nReview: \"{review}\"\n\nResponse:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=150)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Limit number of rows for performance
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting rows to {MAX_ROWS} for demo performance.")
    df = df.head(MAX_ROWS)

# Processing with progress bar
st.subheader("🚀 Processing Reviews")
progress_bar = st.progress(0)
sentiments = []
responses = []
for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    response = generate_response(sentiment, row.Review_text)
    sentiments.append(sentiment)
    responses.append(response)
    progress_bar.progress((i + 1) / len(df))

df["Sentiment"] = sentiments
df["Response"] = responses
st.success("✅ Processing complete!")

# Preview results
st.subheader("📋 Results Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Sentiment chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Distribution")
st.plotly_chart(fig, use_container_width=True)

# CSV Download
st.download_button(
    label="⬇️ Download Results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
