import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Page setup
st.set_page_config(page_title="Sentiment & Auto Response", layout="wide")
st.title("📊 Customer Sentiment & Auto-Responder")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload a CSV with 'Review_text' column", type="csv")

# Load or fallback to sample
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or "Review_text" not in df.columns:
            st.error("❌ CSV must contain 'Review_text' column.")
            st.stop()
        st.success("✅ CSV loaded.")
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("sample_data.csv")
        st.info("ℹ️ Using sample data.")
    except Exception as e:
        st.error(f"❌ Failed to load sample CSV: {e}")
        st.stop()

# Limit rows for performance
MAX_ROWS = 100
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting to first {MAX_ROWS} rows for faster processing.")
    df = df.head(MAX_ROWS)

# Load sentiment pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_pipeline()

# Load better response generation model (FLAN-T5-XL)
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-base")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Helper: Analyze sentiment
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return result["label"].capitalize()
    except:
        return "Unknown"

# Helper: Generate response for negative reviews
def generate_response(review_text):
    prompt = f"""You're a customer service agent. Write a thoughtful and empathetic response to this negative customer review:

Review: {review_text}
Response:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=150)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Process reviews
st.spinner("🚀 Processing reviews...")
progress = st.progress(0)

sentiments = []
responses = []

for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    sentiments.append(sentiment)
    if sentiment == "Negative":
        reply = generate_response(row.Review_text)
    else:
        reply = "No response needed"
    responses.append(reply)
    progress.progress((i + 1) / len(df))

df["Sentiment"] = sentiments
df["Response"] = responses

# Results display
st.success("✅ Done!")
st.subheader("📋 Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Sentiment chart
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Breakdown")
st.plotly_chart(fig, use_container_width=True)

# Download
st.download_button(
    label="⬇️ Download CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)  
