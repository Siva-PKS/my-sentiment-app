import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

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

# Limit the number of rows to 100 for faster processing
row_limit = 100
df = df.head(row_limit)
st.info(f"⚡ Limiting to first {len(df)} rows for faster demo processing.")

# Load Sentiment Model (English-only)
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# Load Response Generation Model
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# Label mapping for sentiment
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

# Helper: Generate a professional customer support response
def generate_response(sentiment, review):
    prompt = f"""You're a helpful support agent. Generate a professional response for the customer review below.

Review: \"{review}\"
Sentiment: {sentiment}
Response:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=100)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# Conditional response generator
def generate_response_if_needed(sentiment, review):
    if sentiment == "Negative":
        return generate_response(sentiment, review)
    else:
        return "No response needed"

# Initialize lists and progress bar
sentiments = []
responses = []
progress_bar = st.progress(0)

# Process each row
with st.spinner("🚀 Running sentiment analysis and generating responses..."):
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sentiment = analyze_sentiment(row['Review_text'])
        response = generate_response_if_needed(sentiment, row['Review_text'])
        sentiments.append(sentiment)
        responses.append(response)
        progress_bar.progress((i + 1) / len(df))

# Add to DataFrame
df["Sentiment"] = sentiments
df["Response"] = responses

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
