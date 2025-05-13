import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch

st.title("📊 Sentiment Analysis on Reviews (Hugging Face Edition)")

uploaded_file = st.file_uploader("Upload a CSV file with 'Review_text' column", type="csv")

# Load uploaded or fallback CSV
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully.")
else:
    df = pd.read_csv("sample_data.csv")
    st.info("Using sample CSV from repo (sample_data.csv).")

# Ensure 'Review_text' exists
if "Review_text" in df.columns:

    # Load multilingual sentiment analysis model (fast)
    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    sentiment_pipeline = load_sentiment_model()

    # Load multilingual and faster text generation model
    @st.cache_resource
    def load_response_model():
        tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
        return tokenizer, model

    response_tokenizer, response_model = load_response_model()

    # Sentiment analysis function (using multilingual model)
    def get_sentiment(text):
        result = sentiment_pipeline(text[:512])[0]
        sentiment = result["label"]
        # Map numerical labels to sentiment labels
        if sentiment == '0':
            return 'Negative'
        elif sentiment == '1':
            return 'Neutral'
        else:
            return 'Positive'

    # Feedback response generation (using multilingual model)
    def generate_feedback_response(sentiment, review):
        prompt = f"Generate a professional customer support response to the following review:\n\n\"{review}\"\n\nSentiment: {sentiment}"
        inputs = response_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = response_model.generate(**inputs, max_new_tokens=100)
        return response_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply sentiment analysis
    df["Sentiment"] = df["Review_text"].apply(get_sentiment)

    # Generate responses
    df["Feedback_Response"] = df.apply(lambda row: generate_feedback_response(row["Sentiment"], row["Review_text"]), axis=1)

    st.write("### 🧾 Sentiment Results with Feedback Responses")
    st.dataframe(df)

    # Plotly bar chart for sentiment distribution
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    st.write("### 📊 Sentiment Distribution (Interactive)")
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                 title="Review Sentiment Breakdown",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig, use_container_width=True)

    # Download results as CSV
    st.download_button(
        label="📥 Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_feedback_output.csv",
        mime="text/csv"
    )

else:
    st.error("The uploaded CSV must contain a 'Review_text' column.")
