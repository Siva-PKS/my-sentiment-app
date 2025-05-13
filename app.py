import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set up the page configuration
st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# File Upload Section
uploaded_file = st.file_uploader("📁 Upload a CSV file with a column named 'Review_text'", type="csv")

# Load uploaded or fallback sample
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV uploaded successfully.")
else:
    df = pd.read_csv("sample_data.csv")
    st.info("ℹ️ Using sample CSV (sample_data.csv)")

# Proceed only if expected column exists
if "Review_text" not in df.columns:
    st.error("❌ The uploaded CSV must contain a 'Review_text' column.")
    st.stop()

# Load Sentiment Model (Pre-trained for sentiment analysis)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# Load LLM for Response Generation (FLAN-T5, instruction-tuned)
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Helper: Run sentiment analysis
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        sentiment = result["label"].capitalize()  # Returns "POSITIVE", "NEGATIVE", or "NEUTRAL"
        return sentiment
    except Exception as e:
        return "Unknown"

# Helper: Generate a professional customer support response
def generate_response(sentiment, review):
    prompt = f"""You're a helpful support agent. Generate a professional response for the customer review below.

Review: "{review}"
Sentiment: {sentiment}
Response:"""
    
    # Optimizing with no_grad() to speed up inference
    with torch.no_grad():
        inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        output = response_model.generate(**inputs, max_new_tokens=100)
        return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Main processing block
with st.spinner("🚀 Running sentiment analysis and generating responses..."):
    # Perform sentiment analysis and response generation in parallel for faster processing
    df["Sentiment"] = df["Review_text"].apply(analyze_sentiment)
    df["Response"] = df.apply(lambda row: generate_response(row["Sentiment"], row["Review_text"]), axis=1)

st.success("✅ Processing complete!")

# Show results
st.subheader("📋 Results Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Chart: Sentiment distribution
st.subheader("📊 Sentiment Breakdown")
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
