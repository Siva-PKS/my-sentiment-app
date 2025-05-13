import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Streamlit page config
st.set_page_config(page_title="Sentiment & Auto-Responder", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# Upload CSV file
uploaded_file = st.file_uploader("📁 Upload a CSV file with a column named 'Review_text'", type="csv")

# Load file or fallback
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or "Review_text" not in df.columns:
            st.error("❌ CSV must contain a non-empty 'Review_text' column.")
            st.stop()
        st.success("✅ CSV uploaded and validated.")
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("ℹ️ Using fallback sample_data.csv")
    except Exception as e:
        st.error(f"❌ Failed to load fallback CSV: {e}")
        st.stop()

# Limit to 100 rows for fast demo
if len(df) > 100:
    st.warning("⚡ Limiting to 100 rows for faster processing in demo mode.")
    df = df.head(100)

# Load fast sentiment model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# Load fast T5 response model
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Sentiment function
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return result["label"].capitalize()
    except:
        return "Unknown"

# Response generator (only for Negative)
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed"
    prompt = f"""
    You're a professional customer support agent.
    Write a helpful and empathetic reply for the following customer review:

    Review: "{review}"
    Sentiment: {sentiment}
    Response:
    """
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=100)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Add progress bar
progress_bar = st.progress(0)
sentiments = []
responses = []

# Batch process with progress
for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    response = generate_response(sentiment, row.Review_text)
    sentiments.append(sentiment)
    responses.append(response)
    progress_bar.progress((i + 1) / len(df))

df["Sentiment"] = sentiments
df["Response"] = responses

st.success("✅ Processing complete!")

# Show table
st.subheader("📋 Results Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Plot sentiment chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Distribution")
st.plotly_chart(fig, use_container_width=True)

# Download option
st.download_button(
    label="⬇️ Download Results as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
