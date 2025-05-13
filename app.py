import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# --- Setup ---
st.set_page_config(page_title="Sentiment & Response Generator", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# --- Upload CSV ---
uploaded_file = st.file_uploader("📁 Upload a CSV file with 'Review_text' column", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or "Review_text" not in df.columns:
            st.error("❌ The file must contain a non-empty 'Review_text' column.")
            st.stop()
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
else:
    st.warning("⚠️ Please upload a file to begin.")
    st.stop()

# --- Limit rows for speed ---
LIMIT = 100
if len(df) > LIMIT:
    st.info(f"⚡ Limiting to first {LIMIT} rows for faster processing.")
    df = df.head(LIMIT)

# --- Load Sentiment Model ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_model()

# --- Map model labels to human-readable form ---
label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# --- Load FLAN-T5-small for response generation ---
@st.cache_resource
def load_response_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    return tokenizer, model

response_tokenizer, response_model = load_response_model()

# --- Analyze Sentiment ---
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(str(text).strip()[:512])[0]
        return label_map.get(result["label"], "Unknown")
    except:
        return "Unknown"

# --- Generate response for NEGATIVE only ---
def generate_response(sentiment, review):
    if sentiment != "Negative":
        return "No response needed."
    prompt = f"""You are a helpful customer service agent. Write a professional response to the following negative customer review.

Review: "{review}"
Sentiment: Negative
Response:"""
    inputs = response_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output = response_model.generate(**inputs, max_new_tokens=100)
    return response_tokenizer.decode(output[0], skip_special_tokens=True)

# --- Run analysis with progress ---
st.subheader("🚀 Processing Reviews...")
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

st.success("✅ All reviews processed.")

# --- Show Results ---
st.subheader("📋 Preview Results")
st.dataframe(df[["Review_text", "Sentiment", "Response"]], use_container_width=True)

# --- Show Sentiment Distribution Chart ---
st.subheader("📊 Sentiment Distribution")
chart_data = df["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]

fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"},
             title="Sentiment Breakdown")
st.plotly_chart(fig, use_container_width=True)

# --- Download CSV ---
st.download_button(
    label="⬇️ Download Processed CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_response_results.csv",
    mime="text/csv"
)
