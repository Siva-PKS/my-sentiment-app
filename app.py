
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Hotfix for torch.classes introspection issue in Streamlit
try:
    del torch._classes
except AttributeError:
    pass

# Set page configuration
st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")

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

# Limit rows for demo purposes
MAX_ROWS = 200
if len(df) > MAX_ROWS:
    st.warning(f"⚠️ Limiting processing to first {MAX_ROWS} rows for faster performance.")
    df = df.head(MAX_ROWS)

# Load 3-class Sentiment Model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Load LLM for response generation
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

sentiment_pipeline = load_sentiment_model()
tokenizer, model = load_response_model()

# Map labels from Cardiff NLP model
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Analyze sentiment function
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        return label_map.get(result["label"], "Unknown")
    except:
        return "Unknown"

# Generate response for Negative sentiment only
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."
    
    # Construct the prompt for LLM
    prompt = (
        "You are a polite and helpful customer support agent. "
        "Write a short, professional reply to this negative customer review:\n"
        f"Review: {review}"
    )
    
    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = model.generate(**inputs, max_new_tokens=150)
    llm_reply = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # Ensure proper punctuation
    if not llm_reply.endswith(('.', '!', '?')):
        llm_reply += '.'

    # Return formatted response
    return f"Thank you for your review. We will look into the issue. {llm_reply}"

# Show progress bar
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
st.subheader("📋 Preview")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# Sentiment Distribution Chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

# Download Button
st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), "sentiment_responses.csv", "text/csv")
