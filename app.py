
import streamlit as st
import pandas as pd
import openai
import plotly.express as px

# Initialize OpenAI API key
openai.api_key = st.secrets["openai"]["api_key"]

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

    # Sentiment analysis function using OpenAI GPT-3.5 or GPT-4 model (Completion.create)
    def get_sentiment_from_gpt(text):
        prompt = f"Classify the sentiment of this text as Positive, Negative, or Neutral: {text}"
        response = openai.Completion.create(
            model= "gpt-4",
            prompt=prompt,
            max_tokens=10,
            temperature=0.0  # Ensure deterministic results
        )
        sentiment = response['choices'][0]['text'].strip()
        return sentiment

    def generate_feedback_response(sentiment, review):
        prompt = f"Generate a professional customer support response to the following review:\n\n\"{review}\"\n\nSentiment: {sentiment}"

        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
        )

        return response['choices'][0]['text'].strip()

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
