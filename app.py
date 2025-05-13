
import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.title("📊 Sentiment Analysis on Reviews (Multilingual HF Edition)")

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

    # Load multilingual sentiment model
    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    sentiment_pipeline = load_sentiment_model()

    # Load instruction-following model (FLAN-T5)
    @st.cache_resource
    def load_response_model():
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        return tokenizer, model

    response_tokenizer, response_model = load_response_model()

    # Sentiment classifier using stars → Positive/Neutral/Negative
    def get_sentiment(text):
        try:
            if pd.isna(text):
                return "Unknown"
            result = sentiment_pipeline(str(text)[:512])[0]
            label = result["label"].lower()
            if "1" in label or "2" in label:
                return "Negative"
            elif "3" in label:
                return "Neutral"
            else:
                return "Positive"
        except Exception:
            return "Error"

    # Generate customer support response
    def generate_feedback_response(sentiment, review):
        prompt = f"Generate a professional customer support response to the following review:\n\n\"{review}\"\n\nSentiment: {sentiment}"
        inputs = response_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = response_model.generate(**inputs, max_new_tokens=100)
        return response_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Apply analysis
    df["Sentiment"] = df["Review_text"].apply(get_sentiment)
    df["Feedback_Response"] = df.apply(lambda row: generate_feedback_response(row["Sentiment"], row["Review_text"]), axis=1)

    st.write("### 🧾 Sentiment Results with Feedback Responses")
    st.dataframe(df)

    # Plot sentiment distribution
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    st.write("### 📊 Sentiment Distribution (Interactive)")
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                 title="Review Sentiment Breakdown",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig, use_container_width=True)

    # Download CSV
    st.download_button(
        label="📥 Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_feedback_output.csv",
        mime="text/csv"
    )

else:
    st.error("The uploaded CSV must contain a 'Review_text' column.")
