import streamlit as st
import pandas as pd
import openai
from textblob import TextBlob
import plotly.express as px

# Initialize OpenAI API key
openai.api_key = "your_openai_api_key"

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

    # Sentiment analysis function using TextBlob or GPT (as per previous integration)
    def get_sentiment_from_gpt(text):
        prompt = f"Classify the sentiment of this text as Positive, Negative, or Neutral: {text}"
        response = openai.Completion.create(
            engine="gpt-4",  # Use a suitable model
            prompt=prompt,
            max_tokens=10,
            temperature=0.0  # Ensure deterministic results
        )
        sentiment = response.choices[0].text.strip()
        return sentiment

    # Function to generate customer support response
    def generate_feedback_response(sentiment, review):
        prompt = f"Generate a customer support response to the following review: {review} Sentiment: {sentiment}"
        response = openai.Completion.create(
            engine="gpt-4",  # Suitable model for response generation
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )
        response_text = response.choices[0].text.strip()
        return response_text

    # Apply sentiment analysis
    df["Sentiment"] = df["Review_text"].apply(get_sentiment_from_gpt)

    # Apply feedback response generation
    df["Feedback_Response"] = df.apply(lambda row: generate_feedback_response(row["Sentiment"], row["Review_text"]), axis=1)

    st.write("### 🧾 Sentiment Results with Feedback Responses")
    st.dataframe(df)

    # Plotly bar chart
    sentiment_counts = df["Sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    st.write("### 📊 Sentiment Distribution (Interactive)")
    fig = px.bar(sentiment_counts, x="Sentiment", y="Count", color="Sentiment",
                 title="Review Sentiment Breakdown",
                 color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
    st.plotly_chart(fig, use_container_width=True)

    # Download option
    st.download_button(
        label="📥 Download results as CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_feedback_output.csv",
        mime="text/csv"
    )

else:
    st.error("The uploaded CSV must contain a 'Review_text' column.")
