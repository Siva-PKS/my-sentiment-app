import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os  # Import os module

# Hotfix for torch.classes introspection issue in Streamlit
try:
    del torch._classes
except AttributeError:
    pass

# Set page configuration
st.set_page_config(page_title="Sentiment Analyzer & Auto-Responder", layout="wide")
st.title("📊 Customer Review Sentiment Analyzer & Auto-Responder")

# ✅ Initialize session state variables
if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

# Check for existing sample data if no file uploaded
sample_data_path = "sample_data.csv"

# Upload CSV or use default sample data file
uploaded_file = st.file_uploader("📁 Upload CSV with 'Review_text' column", type="csv")

# ✅ Optional Enhancement: Reset session state if a new file is uploaded
if uploaded_file is not None:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.processed = False


# If no file is uploaded, check for sample_data.csv
if uploaded_file is None:
    if os.path.exists(sample_data_path):
        st.success("✅ Using 'sample_data.csv' from the directory.")
        df = pd.read_csv(sample_data_path)
    else:
        st.error("❌ 'sample_data.csv' not found. Please upload your own CSV.")
        st.stop()
else:
    # Process the uploaded file
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty or "Review_text" not in df.columns:
            st.error("❌ Missing or empty 'Review_text' column.")
            st.stop()
        st.success("✅ File uploaded successfully.")
        st.session_state.processed = False  # <--- Reset processing for new file
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

# If the data has been processed previously, skip further processing
if st.session_state.processed:
    st.info("ℹ️ The data has already been processed. Refresh the page to process again.")
    # Show only columns that exist
    display_cols = [col for col in ["Unique_ID", "Category", "Review_text", "Sentiment", "Response"] if col in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)
    st.stop()

# Limit rows for demo purposes
MAX_ROWS = 100
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

# Iterate over the DataFrame to process reviews
for i, row in enumerate(df.itertuples(index=False)):
    sentiment = analyze_sentiment(row.Review_text)
    if sentiment == "Negative":
        response = generate_response(sentiment, row.Review_text)
    else:
        response = "No response needed."
    sentiments.append(sentiment)
    responses.append(response)
    progress_bar.progress((i + 1) / len(df))

# Add the generated sentiment and response to the dataframe
df["Sentiment"] = sentiments
df["Response"] = responses
st.session_state.df_processed = df.copy()

# ✅ Ensure DataFrame is stored and reused
if "df_processed" not in st.session_state:
    df["Sentiment"] = sentiments
    df["Response"] = responses
    st.session_state.df_processed = df.copy()
    st.session_state.processed = True

# ✅ Use the stored DataFrame for display and download
df_display = st.session_state.df_processed

# ✅ Display results
st.success("✅ Processing complete!")

st.subheader("📋 Preview")
st.dataframe(df_display[["Unique_ID", "Category", "Review_text", "Sentiment", "Response"]], use_container_width=True)

# ✅ Sentiment Distribution Chart
st.subheader("📊 Sentiment Breakdown")
chart_data = df_display["Sentiment"].value_counts().reset_index()
chart_data.columns = ["Sentiment", "Count"]
fig = px.bar(chart_data, x="Sentiment", y="Count", color="Sentiment",
             color_discrete_map={"Positive": "green", "Neutral": "gray", "Negative": "red"})
st.plotly_chart(fig, use_container_width=True)

# ✅ Download Button - does not cause screen to refresh
st.download_button(
    label="⬇️ Download CSV",
    data=df_display.to_csv(index=False).encode("utf-8"),
    file_name="sentiment_responses.csv",
    mime="text/csv"
)
