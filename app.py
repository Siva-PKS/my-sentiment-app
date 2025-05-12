import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px

st.title("📊 Sentiment Analysis on Reviews")

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

    def get_sentiment(text):
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Review_text"].apply(get_sentiment)

    st.write("### 🧾 Sentiment Results")
    st.dataframe(df)

    # Plotly bar chart
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    st.write("### 📊 Sentiment Distribution (Interactive)")
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                 title="Review Sentiment Breakdown",
                 color_discrete_map={
                     "Positive": "green",
                     "Neutral": "gray",
                     "Negative": "red"
                 })
    st.plotly_chart(fig, use_container_width=True)

    # Download option
    st.download_button(
        label="📥 Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_output.csv",
        mime="text/csv"
    )
else:
    st.error("The uploaded CSV must contain a 'Review_text' column.")
